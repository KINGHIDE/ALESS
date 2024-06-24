import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# 録音設定
duration = 5.0  # 秒
sample_rate = 44100  # サンプリングレート（Hz）

print("Recording...")
# 録音
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
sd.wait()  # 録音完了まで待機
print("Recording finished")

# FFTの計算
frequencies = np.fft.rfftfreq(len(recording), d=1/sample_rate)
fft_magnitudes = np.abs(np.fft.rfft(recording[:, 0]))

# 10Hz刻みでの振幅を得る
resolution = 10
frequency_indices = np.where((frequencies % resolution == 0))[0]
frequencies_10hz = frequencies[frequency_indices]
fft_magnitudes_10hz = fft_magnitudes[frequency_indices]

# 1000Hzの振幅を取得
index_1000hz = np.where(frequencies_10hz == 1000)[0][0]
amplitude_1000hz = fft_magnitudes_10hz[index_1000hz]

print(f"The amplitude at 1000Hz is {amplitude_1000hz}")

# 結果をプロット（オプション）
plt.plot(frequencies_10hz, fft_magnitudes_10hz)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT Amplitudes at 10Hz Intervals')
plt.show()
