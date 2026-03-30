from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from scipy import signal

from config import get_default_config
from model import SpokenQAModel


def audio_to_log_mel(audio_path: str | Path) -> torch.Tensor:
    cfg = get_default_config()
    # Step 2: Spoken question audio ko log-mel speech features me badalna.
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != cfg.audio.sample_rate:
        audio = signal.resample_poly(audio, cfg.audio.sample_rate, sample_rate).astype(np.float32)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.audio.sample_rate,
        n_fft=cfg.audio.n_fft,
        win_length=cfg.audio.win_length,
        hop_length=cfg.audio.hop_length,
        f_min=cfg.audio.fmin,
        f_max=cfg.audio.fmax,
        n_mels=cfg.audio.n_mels,
        power=2.0,
    )
    mel = mel_transform(waveform).clamp_min(1e-5)
    log_mel = torch.log10(mel).squeeze(0).transpose(0, 1).contiguous()
    return log_mel


class SpokenWebQuestionsStage2Dataset(Dataset):
    def __init__(self, manifest_path: str | Path, max_items: int | None = None) -> None:
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]

        valid_records = []
        for record in records:
            audio_path = Path(record["question_audio_path"])
            if audio_path.exists() and audio_path.is_file() and audio_path.stat().st_size > 0:
                valid_records.append(record)

        self.items = valid_records[:max_items] if max_items is not None else valid_records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        return {
            # Step 1/2: Spoken question input aur uski speech features.
            "prompt_mel": audio_to_log_mel(item["question_audio_path"]),
            "question": item["question_text"],
            "answer": item["answer_text"],
        }


def collate_fn(batch):
    prompt_mel = pad_sequence([item["prompt_mel"] for item in batch], batch_first=True)
    prompt_lengths = torch.tensor([item["prompt_mel"].size(0) for item in batch], dtype=torch.long)
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    return {
        "prompt_mel": prompt_mel,
        "prompt_lengths": prompt_lengths,
        "questions": questions,
        "answers": answers,
    }


def build_answer_targets(
    model: SpokenQAModel,
    questions: list[str],
    answers: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Step 6: QA answer text ko training target banana.
    answer_prefixes = [f"Question: {question}\nAnswer:" for question in questions]
    formatted_sequences = [
        f"Question: {question}\nAnswer: {answer}".strip()
        for question, answer in zip(questions, answers)
    ]
    tokenized = model.tokenizer(
        formatted_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=96,
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = input_ids.clone()
    for row_index, prefix_text in enumerate(answer_prefixes):
        prefix_len = len(model.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        labels[row_index, :prefix_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def build_question_targets(
    model: SpokenQAModel,
    questions: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Step 6: Spoken question se expected question text target banana.
    prefix = "Question:"
    sequences = [f"{prefix} {question}".strip() for question in questions]
    tokenized = model.tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    prefix_ids = model.tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_ids)

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def build_alignment_targets(
    model: SpokenQAModel,
    questions: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Step 6: Speech-question embedding ko text-question embedding ke saath align karne ke liye.
    tokenized = model.tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


def contrastive_alignment_loss(
    pooled_prompt: torch.Tensor,
    pooled_text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    # Stage 2 helper: Sahi speech-question aur sahi text-question ko paas laana,
    # aur galat pairs ko door rakhna.
    normalized_prompt = F.normalize(pooled_prompt, dim=-1)
    normalized_text = F.normalize(pooled_text, dim=-1)
    logits = torch.matmul(normalized_prompt, normalized_text.transpose(0, 1)) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    prompt_to_text = F.cross_entropy(logits, targets)
    text_to_prompt = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (prompt_to_text + text_to_prompt)


def main() -> None:
    cfg = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_path = cfg.data.spoken_webquestions_manifest
    max_items = 64
    num_epochs = 2
    batch_size = 2 if device == "cuda" else cfg.train.batch_size

    dataset = SpokenWebQuestionsStage2Dataset(manifest_path, max_items=max_items)
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
    stage1_candidates = sorted(checkpoint_dir.glob("stage1_epoch_*.pt"), key=lambda path: path.stat().st_mtime)
    stage1_checkpoint = stage1_candidates[-1] if stage1_candidates else None
    if stage1_checkpoint is not None and stage1_checkpoint.exists():
        state_dict = torch.load(stage1_checkpoint, map_location=device)
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)
        print(f"Loaded compatible stage-1 weights: {len(compatible_state)} tensors from {stage1_checkpoint}")

    print(f"Using device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(loader, start=1):
            # Step 1/2: Spoken question batch input.
            prompt_mel = batch["prompt_mel"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            # Step 6: Question text understanding target.
            question_ids, question_mask, question_labels = build_question_targets(
                model,
                batch["questions"],
                device,
            )
            # Step 6: Speech-question aur text-question alignment target.
            alignment_ids, alignment_mask = build_alignment_targets(
                model,
                batch["questions"],
                device,
            )
            # Step 6: Final answer text generation target.
            answer_ids, answer_mask, labels = build_answer_targets(
                model,
                batch["questions"],
                batch["answers"],
                device,
            )

            question_outputs = model(
                prompt_mel,
                prompt_lengths=prompt_lengths,
                text_input_ids=question_ids,
                text_attention_mask=question_mask,
            )
            # Step 6: Question understanding / question-text loss.
            question_logits = question_outputs["text_logits"]
            question_shift_logits = question_logits[:, :-1, :].contiguous()
            question_shift_labels = question_labels[:, 1:].contiguous()
            question_loss = F.cross_entropy(
                question_shift_logits.reshape(-1, question_shift_logits.size(-1)),
                question_shift_labels.reshape(-1),
                ignore_index=-100,
            )
            _, pooled_text = model.encode_text_sequence(alignment_ids, alignment_mask)
            pooled_prompt = question_outputs["pooled_prompt"]
            alignment_loss = contrastive_alignment_loss(pooled_prompt, pooled_text)

            answer_outputs = model(
                prompt_mel,
                prompt_lengths=prompt_lengths,
                text_input_ids=answer_ids,
                text_attention_mask=answer_mask,
            )
            # Step 6: Answer text generation loss.
            answer_logits = answer_outputs["text_logits"]
            answer_shift_logits = answer_logits[:, :-1, :].contiguous()
            answer_shift_labels = labels[:, 1:].contiguous()
            answer_loss = F.cross_entropy(
                answer_shift_logits.reshape(-1, answer_shift_logits.size(-1)),
                answer_shift_labels.reshape(-1),
                ignore_index=-100,
            )
            # Stage 2 total loss: question understanding + answer generation + alignment.
            loss = question_loss + answer_loss + (0.5 * alignment_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(
                f"epoch={epoch + 1} step={step} total_loss={loss.item():.4f} "
                f"question_loss={question_loss.item():.4f} answer_loss={answer_loss.item():.4f} "
                f"align_loss={alignment_loss.item():.4f}"
            )

        torch.save(model.state_dict(), checkpoint_dir / f"stage2_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
