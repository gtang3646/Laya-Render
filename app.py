"""FastAPI service exposing the quantized laya model on a 512 MB / CPU-only host.

POST /v1/decide takes the same body as laya's RLAgent.system_one and returns the same response,
so this is a drop-in replacement for the torch runtime:

    {"state": "The package arrived but the screen is cracked.",
     "questions": {"q1": {"type": "choice",
                          "instructions": "What is the user's intent?",
                          "criteria": {"refund": "wants money back",
                                       "support": "wants technical help"}}}}

The model is loaded once at startup; every request reuses the single ONNX session.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union

from laya_runtime import LayaONNX
from model_files import VARIANTS, resolve_model

VARIANT = os.environ.get("LAYA_VARIANT", "wo8")
HF_REPO = os.environ.get("LAYA_HF_REPO", "convaiinnovations/laya-quantized")
INTRA_OP_THREADS = int(os.environ.get("LAYA_INTRA_OP_THREADS", "2"))
MAX_BATCH_QUESTIONS = int(os.environ.get("LAYA_MAX_BATCH_QUESTIONS", "8"))

app = FastAPI(title="laya decision model", version="1.0")
_runtime: LayaONNX = None


class DecideRequest(BaseModel):
    state: Union[str, dict, list] = Field(..., description="Text, JSON object, or conversation turn list")
    questions: dict = Field(..., description="question_id -> {type, instructions, criteria?}")


@app.on_event("startup")
def _load():
    global _runtime
    token = os.environ.get("HF_TOKEN") or None
    path = resolve_model(VARIANT, HF_REPO, token=token)
    _runtime = LayaONNX(path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"),
                        intra_op_threads=INTRA_OP_THREADS,
                        max_batch_questions=MAX_BATCH_QUESTIONS)
    print("loaded %s (%s)" % (path, VARIANTS[VARIANT]["note"]), flush=True)


@app.get("/health")
def health():
    # 503 until the ONNX session is live, so Render's health check does not route traffic to a
    # service that is still downloading/loading the model.
    if _runtime is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return {"status": "ok", "variant": VARIANT}


@app.post("/v1/decide")
def decide(req: DecideRequest):
    if _runtime is None:
        raise HTTPException(status_code=503, detail="model still loading")
    try:
        return _runtime.system_one(req.state, req.questions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")),
                log_level="info")
