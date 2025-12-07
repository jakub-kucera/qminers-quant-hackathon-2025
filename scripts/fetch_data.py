"""Download the competition data from the organisers' public repository.

The CSVs belong to the organisers (IDA-CTU/QQH2025) and are not redistributed here.
This fetches them at a pinned commit into data/ and checks their hashes.

    uv run python scripts/fetch_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "334d224d20fe9a44200f2a475cb8677b16d1dd59"
BASE = f"https://raw.githubusercontent.com/IDA-CTU/QQH2025/{COMMIT}/"
DATA = Path(__file__).resolve().parent.parent / "data"

# path in their repo -> (local name, sha256)
FILES = {
    "data/games.csv": (
        "games.csv",
        "941dde6837b4b25350631e245fb62156286f34d1e6028c7ed4c1c1b96eef9a50",
    ),
    "data/bookie_rookie_games.csv": (
        "bookie_rookie_games.csv",
        "2844eb4c883248435a207a266cba2f05a9b87f0857cc4d6a63fe05564e4a7415",
    ),
    "data/finals/train_games.csv": (
        "finals/train_games.csv",
        "5193c3362d218313c0ccd51971786825205a39671cf246b5dd36c5f3676a94ad",
    ),
    "data/finals/round1_games.csv": (
        "finals/round1_games.csv",
        "3cfd65b7001f010018afe8beae04802f4ef30c5b8a53b5b460e4b3e5ea31fe53",
    ),
    "data/finals/round2_games.csv": (
        "finals/round2_games.csv",
        "124f9289783881c06b9588d0202514a686a5ada949e90641c87338deb86a3e16",
    ),
    "data/finals/round3_games.csv": (
        "finals/round3_games.csv",
        "62a16b0b8f5bfb3e4b804c652ab8af28494d34d02ee1f2073770e84442e930f0",
    ),
    "src/environment.py": (
        "reference/environment.py",
        "b747c28e1eb6a7e2ac3c67d6c3f5b47e776c9e944ee6bd80169d4663183cb8dd",
    ),
}


def main() -> int:
    failed = False
    for remote, (local, digest) in FILES.items():
        target = DATA / local
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetching {remote}")
            with urllib.request.urlopen(BASE + remote) as response:
                target.write_bytes(response.read())
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest is not None and actual != digest:
            print(f"hash mismatch: {target} is {actual}, expected {digest}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
