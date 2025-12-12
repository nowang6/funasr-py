
# Process
VAD_PROCESS_NUM = 1
NUM_WORKERS = 1


VAD_MODEL_PATH="models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
OFFLINE_MODEL_PATH="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ONLINE_MODEL_ONLINE_PATH="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
VAD_MODEL_PATH="models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_PATH="models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
ITN_MODEL_PATH="models/fst_itn_zh"
LM_MODEL_PATH="models/speech_ngram_lm_zh-cn-ai-wesp-fst"



BASE_CHUNK_SIZE = 60 * 16 * 2  # 60ms 每毫秒采样16次 16bit=2byte
AUDIO_CHUNK_SIZE = BASE_CHUNK_SIZE * 8 # 480ms
VAD_CHUNK_SIZE = BASE_CHUNK_SIZE * 4 # 240ms
