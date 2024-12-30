import torch
import librosa
import numpy as np
from funcodec.modules.nets_utils import pad_list


class MelSpec:
    def __init__(
        self,
        fs=16000,
        n_fft=2048,
        hop_size=640,
    ):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_size = hop_size
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
