from funasr import AutoModel
from src.const import OFFLINE_MODEL_PATH, VAD_MODEL_PATH, PUNC_MODEL_PATH
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model=OFFLINE_MODEL_PATH, model_revision="v2.0.4",
                  vad_model=VAD_MODEL_PATH, vad_model_revision="v2.0.4",
                  punc_model=PUNC_MODEL_PATH, punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )
res = model.generate(input="data/张三丰.wav", 
                     batch_size_s=300, 
                     hotword='张三丰')
print(res)
