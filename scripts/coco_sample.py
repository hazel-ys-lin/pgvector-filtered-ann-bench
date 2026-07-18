#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""COCO Captions 參照樣本抽樣(DECISIONS #18)。

規則:官方 2017 annotations zip 的 val split(captions_val2017.json),
1000 distinct images × 每 image 1 caption,固定 seed。zip sha256 記入 manifest。

用法:
    uv run scripts/coco_sample.py --zip <path>/annotations_trainval2017.zip \
        --out corpus/coco_ref_1k.jsonl --n 1000 --seed 42

輸出:
  - coco_ref_1k.jsonl:{id, text}(id = COCO image id;embed loader 直接可吃)
  - coco_ref_1k.manifest.json:zip sha256、抽樣參數、caption 選取規則
  - coco_ref_1k.ATTRIBUTION.md:CC BY 4.0 標註(隨 release assets 發佈)
"""

import argparse
import hashlib
import json
import pathlib
import random
import zipfile

CAPTIONS_MEMBER = "annotations/captions_val2017.json"


def sample(zip_path: str, n: int, seed: int) -> tuple[list, str]:
    sha = hashlib.sha256(pathlib.Path(zip_path).read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read(CAPTIONS_MEMBER))
    by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    # 決定性:image id 排序後 seeded 抽 n;每 image 取 annotation id 最小的 caption
    # (caption 選取不用 rng——少一個隨機源,重現面更窄)
    rng = random.Random(seed)
    image_ids = sorted(by_image)
    chosen = rng.sample(image_ids, n)
    out = []
    for img in sorted(chosen):
        ann = min(by_image[img], key=lambda a: a["id"])
        out.append({"id": img, "text": ann["caption"].strip()})
    return out, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="corpus/coco_ref_1k.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, zip_sha = sample(args.zip, args.n, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "source": "COCO 2017 annotations (official zip), val split",
        "zip_file": "annotations_trainval2017.zip",
        "zip_sha256": zip_sha,
        "member": CAPTIONS_MEMBER,
        "n_images": args.n,
        "seed": args.seed,
        "caption_rule": "per selected image, caption with smallest annotation id",
        "license": "CC BY 4.0 (annotations); see ATTRIBUTION.md",
    }
    pathlib.Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    pathlib.Path(out.parent / (out.stem + ".ATTRIBUTION.md")).write_text(
        """# Attribution — COCO Captions reference sample

This file set includes 1,000 captions sampled from the **COCO 2017
Captions annotations** (val split), © the COCO Consortium, licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Source: annotations_trainval2017.zip (official distribution,
  images.cocodataset.org), member `annotations/captions_val2017.json`
- Sampling: 1,000 distinct images, one caption each (smallest
  annotation id), fixed seed — see `coco_ref_1k.manifest.json` and
  `scripts/coco_sample.py`
- Changes: text extracted verbatim; no modifications to caption text
- Use here: reference distribution for corpus geometry validation
  (see repo README §4)
""")
    print(f"{len(records)} captions → {out}")
    print(f"zip sha256: {zip_sha}")


if __name__ == "__main__":
    main()
