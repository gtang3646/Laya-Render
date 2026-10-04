"""Verify the ONNX runtime matches the reference torch agent field by field.

Runs the same requests through laya.RLAgent (torch, from laya_src) and LayaONNX (this folder) and
compares every answer. This is the gate that proves the deployment is a drop-in replacement;
accuracy on the eval set was already measured for the ONNX graph itself.

Run with the `ai` conda env (needs torch for the reference side):

    set LAYA_MODEL_PATH=D:\\Laya\\quant\\laya_wo8_emb.onnx
    python verify.py
"""
import os
import sys

sys.path.insert(0, r"D:\Laya\laya_src")

MODEL_DIR = r"D:\Laya\laya-multilingual"

CASES = [
    # noul, english
    {"state": "premise: A man is playing a guitar on stage.\nhypothesis: A person is making music.",
     "questions": {"q": {"type": "noul",
                         "instructions": "Does the premise entail the hypothesis?"}}},
    # noul, chinese
    {"state": "premise: 一个人在舞台上弹吉他。\nhypothesis: 那个人正在演奏音乐。",
     "questions": {"q": {"type": "noul",
                         "instructions": "前提是否蕴含假设？"}}},
    # noul, arabic / russian
    {"state": "premise: الرجل يعزف على الجيتار.\nhypothesis: الرجل يصدر صوتا.",
     "questions": {"q": {"type": "noul", "instructions": "هل تنطوي المقدمة على الفرضية؟"}}},
    {"state": "premise: Мужчина играет на гитаре.\nhypothesis: Мужчина издаёт звуки.",
     "questions": {"q": {"type": "noul", "instructions": "Следует ли из предпосылки гипотеза?"}}},
    # choice, criteria as a mapping
    {"state": "The screen arrived cracked and the buyer wants their money back.",
     "questions": {"q": {"type": "choice", "instructions": "What does the user want?",
                         "criteria": {"refund": "wants their money back",
                                      "support": "wants technical help",
                                      "replace": "wants a replacement shipped"}}}},
    # choice, criteria as a bare list (agent._to_internal converts it)
    {"state": "Great product, fast shipping, would buy again.",
     "questions": {"q": {"type": "choice",
                         "instructions": "What is the sentiment of this review?",
                         "criteria": ["negative", "neutral", "positive"]}}},
    # score
    {"state": "The hotel was fine. Room was small but clean, breakfast was forgettable.",
     "questions": {"q": {"type": "score",
                         "instructions": "Rate the overall satisfaction.",
                         "criteria": ["very bad", "bad", "ok", "good", "very good"]}}},
    # multiple questions in one request + structured state
    {"state": {"customer": "jane", "turns": ["I was charged twice", "please help"]},
     "questions": {"sev": {"type": "choice", "instructions": "How severe is this?",
                           "criteria": {"low": "minor inconvenience", "high": "needs escalation"}},
                   "hold": {"type": "noul", "instructions": "Should the agent escalate to a human?"}}},
]


def main():
    import laya
    from laya_runtime import LayaONNX

    ref = laya.load(MODEL_DIR, device="cpu")
    onnx_path = os.environ.get("LAYA_MODEL_PATH", r"D:\Laya\quant\laya_wo8_emb.onnx")
    rt = LayaONNX(onnx_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"),
                  intra_op_threads=2)

    n_same_answer = 0
    max_prob_gap = 0.0
    for i, case in enumerate(CASES):
        a = ref.system_one(case["state"], case["questions"])
        b = rt.system_one(case["state"], case["questions"])
        assert a["answers"].keys() == b["answers"].keys(), (i, a["answers"], b["answers"])
        for qid in a["answers"]:
            qa, qb = a["answers"][qid], b["answers"][qid]
            same = qa.get(qa["type"] if qa["type"] != "noul" else "noul") == \
                qb.get(qb["type"] if qb["type"] != "noul" else "noul")
            n_same_answer += bool(same)
            pa = qa.get("probabilities") or {"noul": qa["noul"], "no": 1 - qa["noul"]}
            pb = qb.get("probabilities") or {"noul": qb["noul"], "no": 1 - qb["noul"]}
            for k in pa:
                max_prob_gap = max(max_prob_gap, abs(pa[k] - pb[k]))
            print("case %d %-6s same=%s  conf %.4f vs %.4f  act %s vs %s"
                  % (i, qa["type"], same, qa["confidence"], qb["confidence"],
                     qa["action"]["act_probability"], qb["action"]["act_probability"]))
        ta, tb = a["usage"]["input_tokens"], b["usage"]["input_tokens"]
        assert ta == tb, ("token usage differs", ta, tb)

    print()
    print("identical answers:   %d/%d" % (n_same_answer, sum(len(c["questions"]) for c in CASES)))
    print("max probability gap: %.4f" % max_prob_gap)
    print("token usage:         identical")
    print("RESULT:", "MATCH" if max_prob_gap < 0.02 else "CHECK")


if __name__ == "__main__":
    main()
