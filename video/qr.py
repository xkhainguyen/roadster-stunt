"""QR code for the public repo, shown on the end card."""
from pathlib import Path

import segno

REPO_URL = "https://github.com/xkhainguyen/roadster-stunt"
REPO_LABEL = "github.com/xkhainguyen/roadster-stunt"


def make_qr(path: str, scale: int = 16) -> str:
    """Black modules on white with a 4-module quiet zone: the most reliable form for phone cameras."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    segno.make(REPO_URL, error="m").save(path, scale=scale, border=4, dark="#000000", light="#ffffff")
    return path
