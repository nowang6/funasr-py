from model_infer_only import FunASRNano


def main():
    model_dir = "models/Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0",disalbe_update=True)
    m.eval()

    wav_path = f"data/近远场测试.wav"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()

