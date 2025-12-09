import copy
import os

import soundfile
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from funasr.download.download_model_from_hub import download_model
from funasr.models.paraformer_streaming.model import ParaformerStreaming
from funasr.register import tables
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.utils.misc import deep_update


def _to_python(obj):
    """Convert OmegaConf containers to native python types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def load_paraformer_streaming(model_path, model_revision="master", device=None):
    """Build ParaformerStreaming the same way AutoModel would."""
    kwargs = download_model(model=model_path, model_revision=model_revision)
    kwargs = _to_python(kwargs)
    set_all_random_seed(kwargs.get("seed", 0))

    if device is None:
        device = kwargs.get("device", "cuda")
    if (
        (device == "cuda" and not torch.cuda.is_available())
        or (device == "xpu" and not torch.xpu.is_available())
        or (device == "mps" and not torch.backends.mps.is_available())
        or kwargs.get("ngpu", 1) == 0
    ):
        device = "cpu"
    kwargs["device"] = device
    torch.set_num_threads(kwargs.get("ncpu", 4))

    tokenizer = kwargs.get("tokenizer", None)
    kwargs["vocab_size"] = -1
    if tokenizer is not None:
        tokenizer_names = tokenizer.split(",") if isinstance(tokenizer, str) else tokenizer
        tokenizers_conf = kwargs.get("tokenizer_conf", {})
        if not isinstance(tokenizers_conf, (list, tuple, ListConfig)):
            tokenizers_conf = [tokenizers_conf] * len(tokenizer_names)
        built_tokenizers = []
        token_lists = []
        vocab_sizes = []
        for tokenizer_name, tokenizer_conf in zip(tokenizer_names, tokenizers_conf):
            tokenizer_conf = _to_python(tokenizer_conf)
            tokenizer_class = tables.tokenizer_classes.get(tokenizer_name)
            tokenizer_inst = tokenizer_class(**tokenizer_conf)
            built_tokenizers.append(tokenizer_inst)
            token_list = tokenizer_inst.token_list if hasattr(tokenizer_inst, "token_list") else None
            if token_list is None and hasattr(tokenizer_inst, "get_vocab"):
                token_list = tokenizer_inst.get_vocab()
            token_lists.append(token_list)
            vocab_sizes.append(len(token_list) if token_list is not None else -1)
        if len(built_tokenizers) == 1:
            built_tokenizers = built_tokenizers[0]
            token_lists = token_lists[0]
            vocab_sizes = vocab_sizes[0]
        kwargs["tokenizer"] = built_tokenizers
        kwargs["token_list"] = token_lists
        kwargs["vocab_size"] = vocab_sizes

    frontend = kwargs.get("frontend", None)
    kwargs["input_size"] = None
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend_conf = _to_python(kwargs.get("frontend_conf", {}))
        frontend = frontend_class(**frontend_conf)
        kwargs["input_size"] = frontend.output_size() if hasattr(frontend, "output_size") else None
    kwargs["frontend"] = frontend

    model_conf = {}
    deep_update(model_conf, _to_python(kwargs.get("model_conf", {})))
    deep_update(model_conf, kwargs)
    if kwargs.get("model") != ParaformerStreaming.__name__:
        raise ValueError(f"Expected ParaformerStreaming model, but got {kwargs.get('model')}")
    model_class = tables.model_classes.get(kwargs["model"])
    model = model_class(**model_conf)

    init_param = kwargs.get("init_param", None)
    if init_param is not None and os.path.exists(init_param):
        load_pretrained_model(
            model=model,
            path=init_param,
            ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
            oss_bucket=kwargs.get("oss_bucket", None),
            scope_map=kwargs.get("scope_map", []),
            excludes=kwargs.get("excludes", None),
        )
    model.to(device)
    model.eval()
    return model, kwargs


chunk_size = [0, 8, 4]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

model_local_path = "models/speech_paraformer-large_asr"

paraformer_model, paraformer_cfg = load_paraformer_streaming(
    model_local_path, model_revision="v2.0.4", device="cuda"
)
inference_base_cfg = {
    k: v
    for k, v in paraformer_cfg.items()
    if k
    not in [
        "model_conf",
        "init_param",
        "model",
        "model_path",
    ]
}

wav_file = "wav/创建警单.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960  # 480ms

cache = {}
key = ["stream"]
total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    inference_kwargs = dict(inference_base_cfg)
    inference_kwargs.update(
        {
        "chunk_size": chunk_size,
        "encoder_chunk_look_back": encoder_chunk_look_back,
        "decoder_chunk_look_back": decoder_chunk_look_back,
        "is_final": is_final,
    }
    )
    result, _ = paraformer_model.inference(
        data_in=[speech_chunk],
        key=key,
        cache=cache,
        **inference_kwargs,
    )
    print(result)