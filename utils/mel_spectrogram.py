import torch
import librosa
import numpy as np
from utils.hinter import hint_once
from funcodec.modules.nets_utils import pad_list

def rms_normalize(audio, target_rms=0.1):
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Calculate scaling factor
    scaling_factor = target_rms / (rms + 1e-8)  # Add epsilon to avoid division by zero
    
    # Apply scaling
    normalized_audio = audio * scaling_factor
    
    # Ensure no clipping
    peak = np.max(np.abs(normalized_audio))
    if peak > 1.0:
        normalized_audio /= peak  # Scale down to avoid clipping
    
    return normalized_audio

class MelSpec:
    def __init__(
        self,
        fs=16000,
        n_fft=2048,
        hop_size=640,
        normalization = False,
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
