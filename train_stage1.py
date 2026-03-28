from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import get_default_config
from dataset import LibriSpeechItem, load_librispeech_items
from model import SpokenQAModel


def load_log_mel(audio_path: str | Path) -> torch.Tensor:
    cfg = get_default_config()
    # Step 2: Stage 1 training ke liye speech audio ko log-mel features me badalna.
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != cfg.audio.sample_rate:
        audio = signal.resample_poly(audio, cfg.audio.sample_rate, sr).astype(np.float32)
    max_samples = int(cfg.audio.max_audio_seconds * cfg.audio.sample_rate)
    audio = audio[:max_samples]
    waveform = torch.from_numpy(audio).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.audio.sample_rate,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
        n_mels=cfg.audio.n_mels,
        f_min=cfg.audio.fmin,
        f_max=cfg.audio.fmax,
        power=2.0,
    )
    mel = mel_transform(waveform).clamp_min(1e-5)
    log_mel = torch.log10(mel).squeeze(0).transpose(0, 1).contiguous()
    return log_mel


def split_prompt_and_continuation(log_mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = get_default_config()
    prompt_frames = max(1, int(round(cfg.audio.prompt_seconds * cfg.audio.sample_rate / cfg.audio.hop_length)))
    if log_mel.size(0) <= prompt_frames + 1:
        raise ValueError("Audio too short for prompt/continuation split.")
    return log_mel[:prompt_frames], log_mel[prompt_frames:]


def truncate_continuation_mel(continuation_mel: torch.Tensor, max_frames: int = 400) -> torch.Tensor:
    if continuation_mel.size(0) <= max_frames:
        return continuation_mel
    return continuation_mel[:max_frames]


def split_transcript_for_prompt(transcript: str, prompt_ratio: float) -> tuple[str, str]:
    words = transcript.strip().split()
    if len(words) < 2:
        raise ValueError("Transcript too short for split.")
    prompt_word_count = max(1, min(len(words) - 1, int(round(len(words) * prompt_ratio))))
    prompt_text = " ".join(words[:prompt_word_count]).strip()
    continuation_text = " ".join(words[prompt_word_count:]).strip()
    if not continuation_text:
        raise ValueError("Continuation transcript empty after split.")
    return prompt_text, continuation_text


class LibriSpeechStage1Dataset(Dataset):
    def __init__(self, root_dir: str | Path, max_items: int | None = None) -> None:
        all_items: list[LibriSpeechItem] = load_librispeech_items(root_dir)
        selected_items = all_items[:max_items] if max_items is not None else all_items
        self.items: list[LibriSpeechItem] = []
        for item in selected_items:
            try:
                log_mel = load_log_mel(item.audio_path)
                split_prompt_and_continuation(log_mel)
                self.items.append(item)
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        log_mel = load_log_mel(item.audio_path)
        # Step 1/2: Input speech ko prompt aur continuation me todna.
        prompt_mel, continuation_mel = split_prompt_and_continuation(log_mel)
        continuation_mel = truncate_continuation_mel(continuation_mel)
        effective_total_frames = prompt_mel.size(0) + continuation_mel.size(0)
        prompt_ratio = prompt_mel.size(0) / effective_total_frames
        prompt_text, continuation_text = split_transcript_for_prompt(item.transcript, prompt_ratio)
        return {
            "prompt_mel": prompt_mel,
            "continuation_mel": continuation_mel,
            "prompt_text": prompt_text,
            "continuation_text": continuation_text,
        }


def collate_fn(batch):
    prompt_mel = pad_sequence([item["prompt_mel"] for item in batch], batch_first=True)
    continuation_mel = pad_sequence([item["continuation_mel"] for item in batch], batch_first=True)
    prompt_lengths = torch.tensor([item["prompt_mel"].size(0) for item in batch], dtype=torch.long)
    prompt_texts = [item["prompt_text"] for item in batch]
    continuation_texts = [item["continuation_text"] for item in batch]
    continuation_lengths = torch.tensor([item["continuation_mel"].size(0) for item in batch], dtype=torch.long)
    return {
        "prompt_mel": prompt_mel,
        "prompt_lengths": prompt_lengths,
        "continuation_mel": continuation_mel,
        "prompt_texts": prompt_texts,
        "continuation_texts": continuation_texts,
        "continuation_lengths": continuation_lengths,
    }


def reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    losses = []
    for row, length in enumerate(lengths.tolist()):
        pred_slice = predicted[row, -length:, :]
        target_slice = target[row, :length, :]
        losses.append(F.l1_loss(pred_slice, target_slice))
    return torch.stack(losses).mean()


def build_stage1_text_targets(
    model: SpokenQAModel,
    prompt_texts: list[str],
    continuation_texts: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Step 6: Transcript continuation ko text target ke roop me banana.
    sequences = [
        f"Transcript: {prompt_text}\nContinuation: {continuation_text}".strip()
        for prompt_text, continuation_text in zip(prompt_texts, continuation_texts)
    ]
    tokenized = model.tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    prefix_lengths = []
    for prompt_text in prompt_texts:
        prefix_text = f"Transcript: {prompt_text}\nContinuation:"
        prefix_ids = model.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        prefix_lengths.append(len(prefix_ids))

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = input_ids.clone()
    for row_index, prefix_len in enumerate(prefix_lengths):
        labels[row_index, :prefix_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def main() -> None:
    cfg = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_items = 256
    num_epochs = 2
    batch_size = 2 if device == "cuda" else cfg.train.batch_size

    dataset = LibriSpeechStage1Dataset(cfg.data.librispeech_dir, max_items=max_items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = SpokenQAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(loader, start=1):
            prompt_mel = batch["prompt_mel"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            continuation_mel = batch["continuation_mel"].to(device)
            continuation_lengths = batch["continuation_lengths"].to(device)
            text_input_ids, text_attention_mask, text_labels = build_stage1_text_targets(
                model,
                batch["prompt_texts"],
                batch["continuation_texts"],
                device,
            )

            outputs = model(
                prompt_mel,
                prompt_lengths=prompt_lengths,
                continuation_mel=continuation_mel,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
            )
            # Step 7/10: Acoustic continuation loss.
            recon_loss = reconstruction_loss(outputs["acoustic_output"], continuation_mel, continuation_lengths)
            text_logits = outputs["text_logits"]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = text_labels[:, 1:].contiguous()
            text_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            # Stage 1: Text + acoustic joint training.
            loss = text_loss + (cfg.model.recon_weight * recon_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(
                f"epoch={epoch + 1} step={step} total_loss={loss.item():.4f} "
                f"text_loss={text_loss.item():.4f} recon_loss={recon_loss.item():.4f}"
            )

        torch.save(model.state_dict(), checkpoint_dir / f"stage1_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
