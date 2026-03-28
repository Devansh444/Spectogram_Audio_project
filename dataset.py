from __future__ import annotations

import bz2
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LibriSpeechItem:
    audio_path: Path
    transcript: str


@dataclass
class WebQuestionsItem:
    question: str
    answers: list[str]


def load_librispeech_items(root_dir: str | Path) -> list[LibriSpeechItem]:
    root = Path(root_dir)
    items: list[LibriSpeechItem] = []

    for transcript_file in root.rglob("*.trans.txt"):
        transcript_lookup: dict[str, str] = {}
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcript_lookup[parts[0]] = parts[1]

        for audio_path in transcript_file.parent.glob("*.flac"):
            transcript = transcript_lookup.get(audio_path.stem)
            if transcript:
                items.append(LibriSpeechItem(audio_path=audio_path, transcript=transcript))

    return items


def load_webquestions_items(file_path: str | Path) -> list[WebQuestionsItem]:
    path = Path(file_path)
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    items: list[WebQuestionsItem] = []
    for record in records:
        question = record.get("utterance", "").strip()
        target_value = record.get("targetValue", "")
        answers = []
        if target_value:
            answers = re.findall(r'"([^"]+)"', target_value)
        if question:
            items.append(WebQuestionsItem(question=question, answers=answers))

    return items


if __name__ == "__main__":
    libri_root = Path("data/LibriSpeech")
    webq_train = Path("data/webquestions/webquestions.examples.train.json.bz2")

    libri_items = load_librispeech_items(libri_root)
    webq_items = load_webquestions_items(webq_train)

    print(f"LibriSpeech items: {len(libri_items)}")
    print(f"WebQuestions items: {len(webq_items)}")
