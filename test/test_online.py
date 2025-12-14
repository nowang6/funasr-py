from funasr import AutoModel
from src.const import ONLINE_MODEL_PATH

chunk_size = [0, 8, 0] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model=ONLINE_MODEL_PATH, model_revision="v2.0.4")

import soundfile
import os

wav_file = "data/张三丰.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 480ms

total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache={}, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=0, decoder_chunk_look_back=0)
    print(res)
