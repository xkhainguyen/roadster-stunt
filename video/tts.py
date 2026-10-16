"""Narration with Kokoro-82M (local, Apache-2.0): one clip per sentence, per-section WAVs and timings."""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24_000
# American-born-sounding Asian male voice: American English base blended with a Mandarin voice's timbre.
VOICE = "am_eric:0.4,zm_yunxi:0.6"
LEAD, GAP, TAIL = 0.12, 0.22, 0.40  # seconds of silence


@dataclass
class Section:
    name: str
    wav: str
    duration: float
    sentences: list = field(default_factory=list)  # (text, t0, t1) within the section


def parse_script(path: str) -> list[tuple[str, list[str]]]:
    sections, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            if name:
                sections.append((name, buf))
            name, buf = line[3:].strip(), []
        elif line.strip():
            buf += [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s]
    if name:
        sections.append((name, buf))
    return sections


# Spelled out for the voice only; captions keep the written form. ("TinyMPC" was heard as "tiny NPC".)
PRONOUNCE = {"TinyMPC": "Tiny M P C"}


def spoken(text: str) -> str:
    for written, said in PRONOUNCE.items():
        text = text.replace(written, said)
    return text


def parse_voice(spec: str) -> list[tuple[str, float]]:
    """'name' or 'name:w,name:w' (Kokoro voice embeddings mixed by weight; weights sum to 1)."""
    parts = [(p.split(":")[0], float(p.split(":")[1]) if ":" in p else 1.0) for p in spec.split(",")]
    if abs(sum(w for _, w in parts) - 1.0) > 1e-6:
        raise ValueError(f"voice weights must sum to 1: {spec}")
    return parts


def synthesize(script_path: str, out_dir: str, voice: str = VOICE, speed: float = 1.2) -> list[Section]:
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    voice = sum(w * pipe.load_single_voice(name) for name, w in parse_voice(voice))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for name, sentences in parse_script(script_path):
        chunks, timings, t = [np.zeros(int(LEAD * SR))], [], LEAD
        for i, text in enumerate(sentences):
            audio = np.concatenate([np.asarray(r.audio) for r in pipe(spoken(text), voice=voice, speed=speed,
                                                                     split_pattern=None)])
            timings.append((text, t, t + len(audio) / SR))
            t += len(audio) / SR
            chunks.append(audio)
            pad = GAP if i < len(sentences) - 1 else TAIL
            chunks.append(np.zeros(int(pad * SR)))
            t += pad
        wav = np.concatenate(chunks).astype(np.float32)
        path = out / f"{name}.wav"
        sf.write(path, wav, SR)
        result.append(Section(name=name, wav=str(path), duration=len(wav) / SR, sentences=timings))
    (out / "sections.json").write_text(json.dumps([asdict(s) for s in result], indent=1))
    return result


def load_sections(out_dir: str) -> list[Section]:
    return [Section(**d) for d in json.loads((Path(out_dir) / "sections.json").read_text())]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    secs = synthesize(str(root / "video" / "script.md"), str(root / "audio"))
    for s in secs:
        print(f"{s.name:10s} {s.duration:5.1f}s")
    print(f"total {sum(s.duration for s in secs):.1f}s")
