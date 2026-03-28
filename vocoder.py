from __future__ import annotations

import numpy as np
from scipy.signal import resample
import soundfile as sf
import torch

from config import get_default_config


def griffin_lim_vocoder(log_mel: np.ndarray) -> np.ndarray:
    cfg = get_default_config()
    # Step 11: Spectrogram ko waveform/audio me convert karne ka basic vocoder.
    n_fft = cfg.audio.n_fft
    hop_length = cfg.audio.hop_length
    win_length = cfg.audio.win_length
    iterations = cfg.audio.griffin_lim_iters
    target_bins = (n_fft // 2) + 1

    mel_np = np.asarray(log_mel, dtype=np.float32)
    resized = np.stack(
        [resample(frame, target_bins) for frame in mel_np],
        axis=0,
    ).astype(np.float32)
    magnitude = torch.tensor(np.exp(resized).T, dtype=torch.float32)

    phase = 2 * np.pi * torch.rand_like(magnitude)
    complex_spec = torch.polar(magnitude, phase)
    window = torch.hann_window(win_length)

    for _ in range(iterations):
        audio = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        rebuilt = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        phase = rebuilt / rebuilt.abs().clamp_min(1e-8)
        complex_spec = magnitude * phase

    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    return audio.cpu().numpy().astype(np.float32)


def save_waveform(audio: np.ndarray, output_path: str) -> None:
    cfg = get_default_config()
    # Step 12: Final audio file save karna.
    sf.write(output_path, audio, cfg.audio.sample_rate)


if __name__ == "__main__":
    dummy_log_mel = np.random.randn(100, 128).astype(np.float32)
    waveform = griffin_lim_vocoder(dummy_log_mel)
    print(f"Waveform shape: {waveform.shape}")
