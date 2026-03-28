from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 200
    fmin: int = 20
    fmax: int = 8000
    prompt_seconds: float = 3.0
    max_audio_seconds: float = 12.0
    griffin_lim_iters: int = 32


@dataclass
class ModelConfig:
    llm_name: str = "gpt2"
    speech_hidden_size: int = 256
    llm_hidden_size: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 4
    dropout: float = 0.1
    recon_weight: float = 0.1
    derivative_order: int = 3


@dataclass
class TrainConfig:
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 2e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cuda"


@dataclass
class DataConfig:
    root_dir: Path = Path("data")
    librispeech_dir: Path = Path("data/LibriSpeech")
    webquestions_dir: Path = Path("data/webquestions")
    spoken_webquestions_manifest: Path = Path("data/webquestions/spoken_train.jsonl")


@dataclass
class ExperimentConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


if __name__ == "__main__":
    cfg = get_default_config()
    print(cfg)
