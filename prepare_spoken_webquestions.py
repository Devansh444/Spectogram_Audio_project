from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import edge_tts

from dataset import load_webquestions_items


async def synthesize_question(text: str, output_path: Path, voice: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(str(output_path))


async def main_async(max_items: int, voice: str) -> None:
    data_dir = Path("data/webquestions")
    train_path = data_dir / "webquestions.examples.train.json.bz2"
    audio_dir = data_dir / "spoken_questions"
    manifest_path = data_dir / "spoken_train.jsonl"

    items = [item for item in load_webquestions_items(train_path) if item.answers][:max_items]
    written = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index, item in enumerate(items):
            audio_path = audio_dir / f"question_{index:05d}.mp3"
            try:
                if not audio_path.exists():
                    await synthesize_question(item.question, audio_path, voice=voice)
            except Exception as exc:
                skipped += 1
                print(f"skipped={skipped} index={index} reason={type(exc).__name__}")
                continue
            record = {
                "question_audio_path": str(audio_path),
                "question_text": item.question,
                "answer_text": item.answers[0],
            }
            manifest_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
            print(f"written={written} audio={audio_path.name}")

    print(f"Manifest saved to {manifest_path}")
    print(f"Total written={written} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=512)
    parser.add_argument("--voice", type=str, default="en-US-AriaNeural")
    args = parser.parse_args()
    asyncio.run(main_async(max_items=args.max_items, voice=args.voice))


if __name__ == "__main__":
    main()
