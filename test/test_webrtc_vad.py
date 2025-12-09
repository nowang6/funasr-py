import webrtcvad
import wave

vad = webrtcvad.Vad()
vad.set_mode(1)

# 使用 wave 库读取 WAV 文件
with wave.open("data/创建警单.wav", "rb") as wf:
    sample_rate = wf.getframerate()
    frame_duration = 30  # 毫秒，只能是 10, 20, 或 30
    
    # 计算每帧的样本数
    frame_length = int(sample_rate * frame_duration / 1000)
    
    print(f"Sample rate: {sample_rate}, Frame duration: {frame_duration}ms, Frame length: {frame_length} samples")
    
    while True:
        frame = wf.readframes(frame_length)
        if len(frame) < frame_length * 2:  # 不足一帧时退出
            break
        res = vad.is_speech(frame, sample_rate)
        print(res)
        