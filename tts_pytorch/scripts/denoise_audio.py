import os
import json
import torchaudio
raw_audio_dir = "./raw_audio/"
denoise_audio_dir = "./denoised_audio/"
filelist = list(os.walk(raw_audio_dir))[0][2]
# 2023/4/21: Get the target sampling rate
with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    hps = json.load(f)
target_sr = hps['data']['sampling_rate']
# 对于以.wav结尾的文件，使用demucs工具对其进行分离处理，仅保留人声（vocals）部分
for file in filelist:
    if file.endswith(".wav"):
        os.system(f"demucs --two-stems=vocals {raw_audio_dir}{file}")
# 对于每个文件，将文件名中的.wav替换为空，得到基本文件名
for file in filelist:
    file = file.replace(".wav", "")
    # 使用torchaudio.load加载之前分离得到的音频文件,对音频进行了一些处理，如截取、归一化等
    wav, sr = torchaudio.load(f"./separated/htdemucs/{file}/vocals.wav", frame_offset=0, num_frames=-1, normalize=True,
                              channels_first=True)
    # merge two channels into one
    # 将音频的两个通道取平均，变成单通道音频
    wav = wav.mean(dim=0).unsqueeze(0)
    # 如果采样率与目标采样率不一致，使用torchaudio.transforms.Resample对音频进行重采样
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    # 使用torchaudio.save将处理后的音频保存到denoise_audio_dir目录下，保存的文件名是基本文件名加上.wav后缀
    torchaudio.save(denoise_audio_dir + file + ".wav", wav, target_sr, channels_first=True)
    # os.remove(f"./separated/htdemucs/{file}/vocals.wav")