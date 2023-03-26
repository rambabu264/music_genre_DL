import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import librosa

class Feature_Extractor:

    def __init__(self, path):
        self.path = path
        self.freq_min = 0
        self.mel_filter_num = 10
        self.hop_size = 15  # ms
        self.FFT_size = 2048
        self.sample_rate = 44100

    def normalize_audio(self, audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    def frame_audio(self, audio):
        # hop_size in ms

        audio = np.pad(audio, int(self.FFT_size / 2), mode='reflect')
        frame_len = np.round(self.sample_rate * self.hop_size / 1000).astype(int)
        frame_num = int((len(audio) - self.FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, self.FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n * frame_len:n * frame_len + self.FFT_size]

        return frames

    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(self, mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def get_filter_points(self):
        fmin_mel = self.freq_to_mel(self.freq_min)
        fmax_mel = self.freq_to_mel(self.freq_max)

        mels = np.linspace(fmin_mel, fmax_mel, num=self.mel_filter_num + 2)
        freqs = self.met_to_freq(mels)

        return np.floor((self.FFT_size + 1) / self.sample_rate * freqs).astype(int), freqs

    def get_filters(self, filter_points):
        filters = np.zeros((len(filter_points) - 2, int(self.FFT_size / 2 + 1)))

        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1,
                                                                             filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0,
                                                                                 filter_points[n + 2] - filter_points[
                                                                                     n + 1])

        return filters

    def dct(self, dct_filter_num):
        basis = np.empty((dct_filter_num, self.mel_filter_num))
        basis[0, :] = 1.0 / np.sqrt(self.mel_filter_num)

        samples = np.arange(1, 2 * self.mel_filter_num, 2) * np.pi / (2.0 * self.mel_filter_num)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / self.mel_filter_num)

        return basis

    def Audio_Power(self):
        sample_rate, audio = wavfile.read(self.path)
        self.freq_max = sample_rate/2

        audio = self.normalize_audio(audio)
        audio_framed = self.frame_audio(audio)

        window = get_window("hann", self.FFT_size, fftbins=True)
        audio_win = audio_framed * window

        audio_winT = np.transpose(audio_win)

        audio_fft = np.empty((int(1 + self.FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

        audio_fft = np.transpose(audio_fft)
        audio_power = np.square(np.abs(audio_fft))

        return audio_power, audio

    def MFCC(self):
        print(self.path)
        audio_power, audio = self.Audio_Power()

        filter_points, mel_freqs = self.get_filter_points()
        filters = self.get_filters(filter_points)

        enorm = 2.0 / (mel_freqs[2:self.mel_filter_num + 2] - mel_freqs[:self.mel_filter_num])
        filters *= enorm[:, np.newaxis]

        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered)

        dct_filters = self.dct(dct_filter_num=20)

        cepstral_coefficents = np.dot(dct_filters, audio_log)

        return cepstral_coefficents, audio, self.sample_rate




    def MFCCS(self):

        hop_size = 15
        sample_rate = 44100

        audio, sr = librosa.load(self.path)
        # Extract MFCC coefficients
        frame_len = int(round(sample_rate * hop_size / 1000))
        frames = librosa.util.frame(audio, frame_length=20, hop_length=frame_len).T

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, hop_length=frame_len)

        return mfccs.T