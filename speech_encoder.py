from __future__ import annotations

import torch
from torch import nn

from config import get_default_config


class SpeechEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Step 3: Speech encoder ka first part speech features ko compact temporal tokens me badalta hai.
        self.conv_subsampler = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        # Step 3: Encoder hidden size me projection.
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        # Step 3: Main transformer speech encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _sinusoidal_positions(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / self.hidden_size)
        )
        embeddings = torch.zeros(length, self.hidden_size, device=device, dtype=dtype)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings.unsqueeze(0)

    def _downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp_min(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Step 3: Input speech features ko encoded speech representation me badalna.
        x = x.transpose(1, 2)
        x = self.conv_subsampler(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self._sinusoidal_positions(x.size(1), x.device, x.dtype)
        padding_mask = None
        if lengths is not None:
            encoded_lengths = self._downsample_lengths(lengths.to(x.device))
            time_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            padding_mask = time_positions >= encoded_lengths.unsqueeze(1)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return encoded, padding_mask


def build_speech_encoder() -> SpeechEncoder:
    cfg = get_default_config()
    return SpeechEncoder(
        input_dim=cfg.audio.n_mels,
        hidden_size=cfg.model.speech_hidden_size,
        num_layers=cfg.model.encoder_layers,
        num_heads=cfg.model.encoder_heads,
        dropout=cfg.model.dropout,
    )


if __name__ == "__main__":
    model = build_speech_encoder()
    dummy_input = torch.randn(2, 120, 128)
    output = model(dummy_input)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
