import argparse, hashlib, json
from itertools import islice
from pathlib import Path
from posixpath import split
from pydoc import text
from datasets import load_dataset
DATASET = "roneneldan/TinyStories"



def clean(text: str) -> str:

    text = text.replace("\r\n", "\n").strip()
    return "\n".join(line.rstrip() for line in text.splitlines())


def dump(split: str, limit: int, out: Path):

    ds = load_dataset(DATASET, split=split, streaming=True)

    seen, kept = set(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:

        for row in islice(ds, limit * 2):

            text = clean(row.get("text", ""))
            if len(text) < 40 or len(text) > 12000: continue

            digest = hashlib.sha256(text.encode()).hexdigest()

            if digest in seen: continue

            seen.add(digest)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= limit: break
            
    print(split, "kept=", kept)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200_000)
    ap.add_argument("--valid", type=int, default=5_000)
    args = ap.parse_args()
    dump("train", args.train, Path("data/train.jsonl"))
    dump("validation", args.valid, Path("data/valid.jsonl"))