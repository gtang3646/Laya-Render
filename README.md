# laya decision API — Render 部署

把量化后的 laya-multilingual 决策模型以 HTTP API 形式部署到 Render free plan
（512 MB RAM / x86_64 / 纯 CPU），与目标小设备的配置一致。

本目录零 torch / transformers 依赖，推理完全由 onnxruntime 完成，启动时只加载一份
ONNX 会话，所有请求复用它。

## 文件说明

| 文件 | 作用 |
|---|---|
| `app.py` | FastAPI 服务：`POST /v1/decide`、`GET /health` |
| `laya_runtime.py` | 推理运行时，逐行移植 laya 的 `common.py` + `agent.py` |
| `model_files.py` | 从 HuggingFace 下载模型，或用 `LAYA_MODEL_PATH` 指向本地文件 |
| `upload_model.py` | 把量化模型 + tokenizer 上传到 HF（本地执行一次） |
| `verify.py` | 与原 torch agent 逐字段对比验证 |
| `tokenizer/` | tokenizer.json + tokenizer_config.json（约 33 MB，随仓库走） |
| `Dockerfile` / `render.yaml` / `requirements.txt` | Render 部署配置 |

## 量化模型选择

| variant | 权重大小 | 准确率 | ECE | 运行时 RSS | 说明 |
|---|---|---|---|---|---|
| `wo8`（默认） | 370 MB | **0.9100**（fp32 为 0.9092，无损） | 0.0415 | ~0.30 GB | int8 weight-only |
| `wo4` | 315 MB | 0.9031（-0.61 pt） | 0.0476 | ~0.25 GB | int4 weight-only，内存更省 |

两者都是 weight-only 量化（激活保持 fp32）。动态/静态 int8 激活量化会掉到
0.87 / 0.82，不要用。详见 `../docs/quantization.md`。

默认用 `wo8`：精度无损。若实例 OOM，把 `LAYA_VARIANT` 改成 `wo4` 即可，无需改代码。

## 1. 上传模型到 HuggingFace

权重不进 git 仓库（300–370 MB），Render 启动时从 HF 拉。先在本地执行一次：

```powershell
cd D:\Laya\deploy
set HF_TOKEN=hf_xxx            # https://huggingface.co/settings/tokens，需要 write 权限
D:\anaconda3\envs\ai\python.exe upload_model.py --repo 你的用户名/laya-quantized --variant wo8
D:\anaconda3\envs\ai\python.exe upload_model.py --repo 你的用户名/laya-quantized --variant wo4
```

两个 variant 都传上去，部署时靠 `LAYA_VARIANT` 切换，不用重新上传。

## 2. 部署到 Render

1. 把本目录（`deploy/`）作为仓库根目录推到 GitHub。`tokenizer/` 要一起推，
   `.gitignore` 已排除 `__pycache__`、`.env`、`models/`。
2. 在 Render Dashboard 新建 Web Service，连接该 GitHub 仓库。
3. `render.yaml` 已定义好服务（runtime=docker、plan=free、region=oregon、
   healthCheckPath=`/health`、autoDeploy=true），Render 会自动识别。
4. 检查环境变量（见下表），把 `LAYA_HF_REPO` 改成你刚上传的 repo。模型 repo 公开时
   不需要 `HF_TOKEN`。
5. 部署。首次启动会从 HF 下载模型（300–370 MB），`/health` 在加载完成前返回 503，
   Render 不会把流量导给还没准备好的实例。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LAYA_VARIANT` | `wo8` | 量化版本，可选 `wo8` / `wo4` |
| `LAYA_HF_REPO` | `gtang0115/laya-quantized` | 模型所在 HF repo |
| `LAYA_INTRA_OP_THREADS` | `2` | ORT intra-op 线程数；512MB 实例建议 2，多了反而争抢 |
| `LAYA_MAX_BATCH_QUESTIONS` | `8` | 单次请求最多合并几个问题，超过则拆批 |
| `HF_TOKEN` | 无 | 仅当模型 repo 是私有时需要 |
| `PORT` | `8000` | Render 自动注入 |

## 3. 本地测试

```powershell
cd D:\Laya\deploy
set LAYA_MODEL_PATH=D:\Laya\quant\laya_wo8_emb.onnx
D:\anaconda3\envs\ai\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8123
```

另开一个终端：

```powershell
curl.exe -X POST http://127.0.0.1:8123/v1/decide -H "Content-Type: application/json" --data "{""state"":""The package arrived but the screen is cracked."",""questions"":{""q1"":{""type"":""choice"",""instructions"":""What is the user's intent?"",""criteria"":{""refund"":""wants money back"",""support"":""wants technical help""}}}}"
```

期望返回类似：

```json
{"model": "laya-rl-agent",
 "answers": {"q1": {"type": "choice", "choice": "refund",
                    "probabilities": {"refund": 0.7537, "support": 0.2463},
                    "confidence": 0.1947, "action": {"act_probability": 1.0}}},
 "usage": {"input_tokens": 36, "output_tokens": 0}}
```

## API 接口

### `POST /v1/decide`

入参与 laya 的 `RLAgent.system_one` 完全一致，可直接 drop-in 替换 torch 版本：

```json
{
  "state": "The package arrived but the screen is cracked.",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "What is the user's intent?",
      "criteria": {"refund": "wants money back", "support": "wants technical help"}
    }
  }
}
```

`state` 可以是文本、JSON 对象或对话轮次列表；`questions` 是 `question_id -> {type, instructions, criteria?}`，
`type` 支持 `choice` / `score` / `noul`（`score` 类型需要 `criteria` 里的描述）。
返回 `question_id -> {type, choice|score|noul, probabilities, confidence, action}`，结构与 torch agent 相同。

模型加载中或未加载时返回 503，入参格式错误返回 422（pydantic 校验）。

### `GET /health`

加载完成后返回 `{"status": "ok", "variant": "wo8"}`，加载中返回 503。

## 4. 与原 torch 实现的一致性验证

```powershell
cd D:\Laya\deploy
D:\anaconda3\envs\ai\python.exe verify.py
```

`verify.py` 默认用 `laya_wo8_emb.onnx`，改 `LAYA_MODEL_PATH` 即可换对照模型。
用无折叠 fp32 基座（`D:\Laya\quant\laya_fp32_nf.onnx`）与原 torch agent 跑同一批输入
逐字段对比：**9/9 结果完全一致，概率差 0.0000**，tokenization 逐位一致，
证明运行时移植无误（量化前基座已与 torch 完全对齐，所以任何差异都只能来自量化本身）。
在 wo8 模型上的小差异纯粹是已量化的 int8 噪声。

## 内存取舍

512 MB 是硬约束。默认 `wo8` 加载后 RSS 约 0.30 GB，离上限有余量但不算宽裕；
`wo4` 约 0.25 GB，牺牲 0.61 pt 精度换更稳的内存余量。实例出现 OOM 重启时，
改 `LAYA_VARIANT=wo4` 重新部署即可，代码无需任何改动。

启动时模型会从 HF 下载到缓存，再复制一份到 models/<variant>/ 下使 .onnx 与 .onnx.data
并排放置（HF 缓存按哈希把两个文件分到不同目录，onnxruntime 要求外置数据紧邻模型文件，
否则报 External data path escapes model directory）。因此磁盘占用约两倍模型大小
（wo8 约 740 MB），8 GB 存储无压力；models/ 已在 .gitignore 中。

Dockerfile 里只用 1 个 uvicorn worker：每个 worker 会加载自己的 ONNX 会话，
多 worker 会成倍占用内存，512 MB 扛不住。单个 worker 的事件循环足以处理并发请求，
ORT 会话内部串行执行推理，这正好匹配 512 MB 实例的预算。
