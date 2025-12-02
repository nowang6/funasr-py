# worker.py
import time
import multiprocessing as mp

print("model loading")
from funasr import AutoModel



ASR_MODEL="models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
ASR_MODEL_ONLINE="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
VAD_MODEL="models/speech_fsmn_vad_zh-cn-16k-common-onnx"
PUNC_MODEL="models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
ITN_MODEL="models/fst_itn_zh"
LM_MODEL="models/speech_ngram_lm_zh-cn-ai-wesp-fst"

# asr
model_asr = AutoModel(
    model=ASR_MODEL,
    disable_pbar=True,
    disable_log=True,
)
# asr
model_asr_streaming = AutoModel(
    model=ASR_MODEL_ONLINE,
    disable_pbar=True,
    disable_log=True,
)
# vad
model_vad = AutoModel(
    model=VAD_MODEL,
    disable_pbar=True,
    disable_log=True,
    # chunk_size=60,
)

# punc
model_punc = AutoModel(
    model=PUNC_MODEL,
    disable_pbar=True,
    disable_log=True,
)




def infer(data):
    # 模型推理逻辑（替换）
    time.sleep(0.1)
    return f"model_output_for_{data}"

def worker_loop(task_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
    print(f"Worker {worker_id} started.")
    while True:
        task = task_queue.get()
        if task is None:
            break

        req_id, payload = task
        output = infer(payload)

        # 回传给主进程
        result_queue.put((req_id, output, worker_id))
