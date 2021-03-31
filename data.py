import os
import random
import librosa
import numpy as np
import albumentations
import soundfile as sf
import colorednoise as cn
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import BasicTransform


class AudioTransform(BasicTransform):
    """
    Transform for audio task. This is the main class where we
    override the targets and update params function for our need
    """

    @property
    def targets(self):
        return {"data": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class AddGaussianNoise(AudioTransform):
    """
    Do time shifting of audio
    """

    def __init__(self, always_apply=False, p=0.5):
        super(AddGaussianNoise, self).__init__(always_apply, p)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        noise = np.random.randn(len(data))
        ratio = 0.0001 * random.randint(20, 500)
        data_wn = data + ratio * noise
        return data_wn


class Gain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False, p=0.5):
        super(Gain, self).__init__(always_apply, p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db

    def apply(self, data, **args):
        amplitude_ratio = 10 ** (random.uniform(self.min_gain_in_db, self.max_gain_in_db) / 20)
        return data * amplitude_ratio


class StretchAudio(AudioTransform):
    """
    Do stretching of audio file
    """

    def __init__(self, always_apply=False, p=0.5, rate=None):
        super(StretchAudio, self).__init__(always_apply, p)

        if rate:
            self.rate = rate
        else:
            self.rate = np.random.uniform(0.5, 1.5)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        input_length = len(data)

        data = librosa.effects.time_stretch(data, self.rate)

        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


def get_train_transforms():
    return albumentations.Compose([
        # TimeShifting(p=0.3),  # here not p=1.0 because your nets should get some difficulties
        # albumentations.OneOf([
        # AddCustomNoise(file_dir='noise', p=0.8),
        # SpeedTuning(p=0.2),
        # ]),
        AddGaussianNoise(p=0.5),
        # PitchShift(p=0.2,n_steps=4),
        Gain(p=0.3),
        # PolarityInversion(p=0.9),
        StretchAudio(p=0.1),
    ])


class SLUDataset(Dataset):
    def __init__(self, config, df, transform=None, test=False):
        self.df = df
        self.config = config
        self.transform = transform
        self.test = test
        self.labels = []
        with open(os.path.join(config.data_dir, 'data/label.txt')) as f:
            for line in f:
                self.labels.append(line.strip())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        audio, _ = sf.read(os.path.join(self.config.data_dir, self.df.loc[index, 'path']))
        x = np.zeros(16000 * 5)
        if not self.test and self.transform is not None:
            audio = self.transform(data=audio)['data']
        if len(audio) > 16000 * 5:
            audio = audio[:16000 * 5]
        x[:len(audio)] = audio

        label = np.zeros(len(self.labels))
        label[self.labels.index(self.df.loc[index, 'action'])] = 1
        label[self.labels.index(self.df.loc[index, 'object'])] = 1
        label[self.labels.index(self.df.loc[index, 'location'])] = 1

        return x, label
