"""Laya decision-model inference runtime -- ONNX + numpy only, no torch/transformers.

Purpose: run the quantized laya ONNX model inside a 512 MB / CPU-only container. The original
`laya` package pulls torch + transformers (~1 GB install, heavy import), which the target does
not have room for; this module ports exactly the inference path (prompt formatting, collation,
scoring) with only onnxruntime / tokenizers / numpy.

Every piece is a line-for-line port of laya_src/laya/common.py + agent.py so responses match the
reference torch agent:
  build_sequence    <- common.build_sequence
  render_options    <- common.render_options
  render_criterion  <- common.render_criterion
  system_one        <- agent.Agent.system_one
Tokenization equivalence with the HF tokenizer was verified token-id-for-token-id before this
module was written (the `tokenizers` Rust core is what transformers calls anyway).
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}

# From rl_agent_config.json of convaiinnovations/laya-multilingual. max_len/head_max_len are the
# prompt budgets the model was trained against; temperature is [1,1,1] for this checkpoint (no
# scaling applied), kept here so the behaviour stays identical if a checkpoint changes it.
DEFAULT_MAX_LEN = 1024
DEFAULT_HEAD_MAX_LEN = 256

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def build_sequence(tok, state, q, max_len, head_max_len, mask_token, mask_id, cls_id, sep_id):
    """Format: [CLS] <type> instructions [SEP] ⫿opt0⫿ opt1 ... [SEP] state [SEP].

    Ported from laya.common.build_sequence. ⫿ = the mask token, whose position is the marker the
    scorer reads, so the marker positions must be computed exactly as the original does.
    """
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_token, " ")
    head_ids = tok.encode("%s question: %s" % (q["t"], ins), add_special_tokens=False).ids
    opt_ids = []
    for i in range(len(opts)):
        opt_ids.append([mask_id] + tok.encode(" " + opts[i].replace(mask_token, " "),
                                              add_special_tokens=False).ids[:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [cls_id] + head_ids + [sep_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sep_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok.encode(serialize_state(state).replace(mask_token, " "), add_special_tokens=False).ids
    st = st[:room]
    ids = ids + st + [sep_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def to_internal(qdef: Dict) -> Dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


class LayaONNX:
    """Wraps one quantized ONNX checkpoint and answers typed questions exactly like the
    torch RLAgent, minus torch."""

    def __init__(self, model_path: str, tokenizer_dir: str, intra_op_threads: int = 2,
                 max_len: int = DEFAULT_MAX_LEN, head_max_len: int = DEFAULT_HEAD_MAX_LEN,
                 max_batch_questions: int = 8):
        self.tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            tcfg = json.load(f)
        self.cls_id = self.tok.token_to_id(tcfg["cls_token"])
        self.sep_id = self.tok.token_to_id(tcfg["sep_token"])
        self.mask_id = self.tok.token_to_id(tcfg["mask_token"])
        self.mask_token = tcfg["mask_token"]
        self.pad_id = self.tok.token_to_id(tcfg["pad_token"])
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.max_batch_questions = max_batch_questions

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    def _encode(self, state, questions: Dict[str, Dict[str, Any]]):
        ids, items = list(questions.keys()), []
        for qid in ids:
            q = to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len,
                                          self.mask_token, self.mask_id, self.cls_id, self.sep_id)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, self.head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        return ids, items

    def _run(self, items: List[Dict]) -> tuple:
        """Chunked forward pass. Rows are independent (each has its own attention mask), so
        chunking changes no number -- it only bounds peak activation memory for the 512 MB target.
        """
        logits_out = [None] * len(items)
        act_out = [None] * len(items)
        n_tokens, pos = 0, 0
        while pos < len(items):
            chunk = items[pos: pos + self.max_batch_questions]
            n = len(chunk)
            L = max(len(it["ids"]) for it in chunk)
            kmax = max(len(it["markers"]) for it in chunk)
            input_ids = np.full((n, L), self.pad_id, dtype=np.int64)
            att = np.zeros((n, L), dtype=np.int64)
            mpos = np.zeros((n, kmax), dtype=np.int64)
            mmask = np.zeros((n, kmax), dtype=bool)
            for i, it in enumerate(chunk):
                input_ids[i, : len(it["ids"])] = it["ids"]
                att[i, : len(it["ids"])] = 1
                k = len(it["markers"])
                mpos[i, :k] = it["markers"]
                mmask[i, :k] = True
            feed = {"input_ids": input_ids, "attention_mask": att, "marker_pos": mpos,
                    "marker_mask": mmask,
                    "qtype": np.array([it["qtype"] for it in chunk], dtype=np.int64)}
            logits, act = self.sess.run(None, feed)
            for i in range(n):
                logits_out[pos + i] = logits[i]
                act_out[pos + i] = act[i]
            n_tokens += int(att.sum())   # padded token count, matching the torch agent's usage
            pos += n
        return logits_out, act_out, n_tokens

    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ids, items = self._encode(state, questions)
        logits_list, act_list, n_tokens = self._run(items)

        answers = {}
        for r, qid in enumerate(ids):
            q = to_internal(questions[qid])
            k = len(items[r]["markers"])
            z = np.asarray(logits_list[r][:k], dtype=np.float64)
            p = np.exp(z - z.max())
            p = p / p.sum()
            act = np.asarray(act_list[r], dtype=np.float64)
            act = np.exp(act - act.max())   # softmax, overflow-safe
            act = act / act.sum()
            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }

        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
