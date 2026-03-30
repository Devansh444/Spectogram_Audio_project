from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import difflib
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from config import get_default_config
from model import SpokenQAModel
from vocoder import griffin_lim_vocoder, save_waveform


def record_audio(output_path: str | Path, duration_seconds: int = 5) -> Path:
    cfg = get_default_config()
    # Step 1: User ka spoken question record hota hai.
    print("Recording started... Speak now")
    audio = sd.rec(
        int(duration_seconds * cfg.audio.sample_rate),
        samplerate=cfg.audio.sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(str(output_path), audio, cfg.audio.sample_rate)
    print(f"Recorded audio saved to {output_path}")
    return Path(output_path)


def play_audio(audio: np.ndarray) -> None:
    cfg = get_default_config()
    sd.play(audio, cfg.audio.sample_rate)
    sd.wait()


def audio_to_log_mel(audio_path: str | Path) -> torch.Tensor:
    cfg = get_default_config()
    # Step 2: Audio ko log-mel spectrogram features me badalna.
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != cfg.audio.sample_rate:
        audio = signal.resample_poly(audio, cfg.audio.sample_rate, sample_rate).astype(np.float32)

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
    return log_mel.unsqueeze(0)


def is_low_quality_text(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return True
    words = cleaned.split()
    if len(words) <= 1:
        return True
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.45:
        return True
    if len(cleaned) < 6:
        return True
    punctuation_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in cleaned) / max(len(cleaned), 1)
    return punctuation_ratio > 0.35


def synthesize_response_audio(text: str, output_path: str | Path) -> bool:
    import asyncio
    import edge_tts

    # Step 12: Final answer text ko spoken audio response me badalna.
    safe_text = " ".join(text.strip().split())
    if not safe_text:
        return False
    output_path = Path(output_path)

    async def _speak() -> None:
        communicate = edge_tts.Communicate(safe_text, voice="en-US-AriaNeural")
        await communicate.save(str(output_path))

    try:
        asyncio.run(_speak())
    except Exception:
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def load_spoken_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_retrieval_bank(
    model: SpokenQAModel,
    device: str,
    max_items: int = 256,
) -> tuple[torch.Tensor, list[str], list[str]]:
    # Helper: Known spoken questions ka bank banana for fallback question understanding.
    cfg = get_default_config()
    records = load_spoken_manifest(cfg.data.spoken_webquestions_manifest)[:max_items]
    embeddings = []
    questions = []
    answers = []
    for record in records:
        audio_path = Path(record["question_audio_path"])
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            continue
        prompt_mel = audio_to_log_mel(audio_path).to(device)
        prompt_lengths = torch.tensor([prompt_mel.size(1)], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, pooled = model.encode_prompt(prompt_mel, prompt_lengths)
        embeddings.append(pooled.squeeze(0).cpu())
        questions.append(record["question_text"])
        answers.append(record["answer_text"])
    if not embeddings:
        return torch.empty(0), [], []
    return torch.stack(embeddings, dim=0), questions, answers


def retrieve_answer_from_speech(
    model: SpokenQAModel,
    prompt_mel: torch.Tensor,
    prompt_lengths: torch.Tensor,
    bank_embeddings: torch.Tensor,
    bank_questions: list[str],
    bank_answers: list[str],
) -> tuple[str, str, float]:
    # Helper: Spoken question embedding ka nearest known question-answer pair nikalna.
    if bank_embeddings.numel() == 0:
        return "", "", 0.0
    with torch.no_grad():
        _, _, pooled = model.encode_prompt(prompt_mel, prompt_lengths)
    query = torch.nn.functional.normalize(pooled.squeeze(0).cpu(), dim=0)
    bank = torch.nn.functional.normalize(bank_embeddings, dim=1)
    scores = torch.matmul(bank, query)
    best_index = int(torch.argmax(scores).item())
    return bank_questions[best_index], bank_answers[best_index], float(scores[best_index].item())


def maybe_replace_with_retrieval(
    generated_question_text: str,
    generated_answer_text: str,
    retrieved_question_text: str,
    retrieved_answer_text: str,
    retrieval_score: float,
) -> tuple[str, str]:
    if retrieval_score >= 0.90 and retrieved_answer_text:
        return retrieved_question_text, retrieved_answer_text
    if is_low_quality_text(generated_answer_text) and retrieved_answer_text:
        return retrieved_question_text, retrieved_answer_text
    if generated_question_text and retrieved_question_text:
        similarity = difflib.SequenceMatcher(
            None,
            generated_question_text.lower(),
            retrieved_question_text.lower(),
        ).ratio()
        if similarity >= 0.65:
            return retrieved_question_text, retrieved_answer_text
    return generated_question_text, generated_answer_text


def select_understood_question(
    generated_question_text: str,
    retrieved_question_text: str,
    retrieval_score: float,
) -> tuple[str, bool]:
    if retrieval_score >= 0.90 and retrieved_question_text:
        return retrieved_question_text, True
    if not is_low_quality_text(generated_question_text):
        return generated_question_text, True
    return "I could not understand the question clearly.", False


def main() -> None:
    cfg = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpokenQAModel().to(device)
    checkpoint_candidates = sorted(Path("checkpoints").glob("stage2_epoch_*.pt"), key=lambda path: path.stat().st_mtime)
    checkpoint_path = checkpoint_candidates[-1] if checkpoint_candidates else None
    if checkpoint_path is not None and checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    model.eval()
    bank_embeddings, bank_questions, bank_answers = build_retrieval_bank(model, device=device)

    session_dir = Path("sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    question_path = session_dir / f"question_{timestamp}.wav"
    answer_wav_path = session_dir / f"answer_{timestamp}.wav"
    answer_mp3_path = session_dir / f"answer_{timestamp}.mp3"

    record_audio(question_path)

    # Step 2: Recorded speech ko feature space me convert karna.
    prompt_mel = audio_to_log_mel(question_path).to(device)
    prompt_lengths = torch.tensor([prompt_mel.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        # Step 6: Question understanding / text branch.
        predicted_question_text = model.generate_question_text(prompt_mel, prompt_lengths=prompt_lengths)
        # Step 6: Answer text generation.
        predicted_text = model.generate_answer_from_question_text(
            prompt_mel,
            prompt_lengths=prompt_lengths,
            question_text=predicted_question_text,
        )
        # Step 7/10: Acoustic output branch.
        outputs = model(prompt_mel, prompt_lengths=prompt_lengths)
        predicted_mel = outputs["acoustic_output"][0].cpu().numpy()

    retrieved_question_text, retrieved_answer_text, retrieval_score = retrieve_answer_from_speech(
        model,
        prompt_mel,
        prompt_lengths,
        bank_embeddings,
        bank_questions,
        bank_answers,
    )
    final_question_text, final_answer_text = maybe_replace_with_retrieval(
        predicted_question_text,
        predicted_text,
        retrieved_question_text,
        retrieved_answer_text,
        retrieval_score,
    )
    recognized_question_text, question_understood = select_understood_question(
        predicted_question_text,
        retrieved_question_text,
        retrieval_score,
    )
    if not question_understood:
        final_answer_text = "I could not understand the question clearly."

    predicted_mel = np.asarray(predicted_mel, dtype=np.float32)
    if predicted_mel.ndim == 1:
        predicted_mel = predicted_mel.reshape(1, -1)
    if predicted_mel.ndim != 2 or predicted_mel.shape[0] == 0 or predicted_mel.shape[1] == 0:
        predicted_mel = np.zeros((10, cfg.audio.n_mels), dtype=np.float32)
    predicted_mel = np.nan_to_num(predicted_mel, nan=0.0, posinf=0.0, neginf=0.0)
    if predicted_mel.shape[0] < 10:
        pad_rows = 10 - predicted_mel.shape[0]
        predicted_mel = np.pad(predicted_mel, ((0, pad_rows), (0, 0)), mode="constant")

    used_text_audio = synthesize_response_audio(final_answer_text, answer_mp3_path)
    if used_text_audio:
        answer_audio, _ = sf.read(str(answer_mp3_path), dtype="float32", always_2d=False)
        answer_path = answer_mp3_path
        if isinstance(answer_audio, np.ndarray) and answer_audio.ndim > 1:
            answer_audio = answer_audio.mean(axis=1)
    else:
        try:
            # Step 11: Spectrogram se waveform fallback.
            answer_audio = griffin_lim_vocoder(predicted_mel)
        except Exception:
            answer_audio = np.zeros(cfg.audio.sample_rate, dtype=np.float32)
        save_waveform(answer_audio, str(answer_wav_path))
        answer_path = answer_wav_path
    play_audio(np.asarray(answer_audio, dtype=np.float32))

    print(f"Answer audio saved to {answer_path}")
    print(f"Final answer text: {final_answer_text}")


if __name__ == "__main__":
    main()
