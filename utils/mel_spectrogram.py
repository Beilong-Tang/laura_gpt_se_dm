import torch
import librosa
import numpy as np
from utils.hinter import hint_once
from funcodec.modules.nets_utils import pad_list

def rms_normalize(audio):
    rms = np.sqrt(np.mean(np.square(audio)))  # Calculate the RMS value
    normalized_audio = audio / rms  # Normalize audio to unit RMS
    return normalized_audio

class MelSpec:
    def __init__(
        self,
        fs=16000,
        n_fft=2048,
        hop_size=640,
        normalization = True,
    ):
        """
        normalization: Whether to normalize audio using rms
        """
        self.fs = fs
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.normalization = normalization
        pass

    def mel(self, audio: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            audio: audio shape [B, T]
            mask: shape [B,]
        Returns:
            mel: [B, T', D]
            mel_mask: [B]
        """
        mel_tensor = []
        mel_mask = []
        for a, m in zip(audio, mask):
            m = m.item()
            a = a[:m].cpu().numpy()
            if self.normalization:
                hint_once("normalization applied", "normalization", 0)
                a = rms_normalize(a)
            mel = np.transpose(
                librosa.feature.melspectrogram(
                    y=a,
                    sr=self.fs,
                    n_fft=self.n_fft,
                    hop_length=self.hop_size,
                ),
                (1, 0),
            )
            mel_len = len(mel)
            mel_mask.append(mel_len)
            mel_tensor.append(torch.from_numpy(mel))
        mel_tensor = pad_list(mel_tensor, 0.0)
        mel_mask = torch.tensor(mel_mask, dtype=torch.long)
        return mel_tensor, mel_mask
