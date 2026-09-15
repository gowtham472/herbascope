"""Dataset lock: verify, curate and split Mikrobat; select the DIMPSAR far-OOD subset.

Steps
  1. Inventory every Mikrobat file: readability, format, size, dimensions, SHA-256, pHash.
  2. Exact duplicates: within a class keep one copy; a byte-identical image filed under more
     than one class is a label conflict and moves to the ambiguity-probe set.
  3. Near-duplicate groups. Two images from different species cannot share a physical
     specimen, so cross-class pHash distances form a null distribution of distinct images.
     Pairs closer than its configured low percentile are linked, and links are merged
     transitively. Groups (not images) are the split unit; a group spanning classes is a
     label conflict and is excluded.
  4. Held-out known-material set: per class, the smallest fragment type with at least
     `heldout_min_images` images is withheld from training and the reference library.
  5. Remaining groups: stratified (class x fragment type) train/validation/test split.
  6. DIMPSAR: verify counts, split classes into disjoint calibration/evaluation halves and
     sample a fixed number of images per class (seeded).

Outputs: data/splits/*.csv, data/metadata/{dataset_manifest.json, image_metadata.csv,
split_metadata.json}, data/processed/dimpsar/, docs/reports/dataset_report.md

Usage:  python -m ml.training.prepare_dataset
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from ml import paths
from ml.pipeline_config import PipelineConfig, load_pipeline_config
from ml.preprocessing.image_io import InvalidImageError, decode_image
from ml.training import datasets as ds

MIKROBAT = "Mikrobat"
DIMPSAR = "DIMPSAR"
EXCLUDED = "excluded"
_TRAILING_INDEX = re.compile(r"(\s*\(\d+\)|\s+\d+)+$")


def perceptual_hash(image: Image.Image) -> int:
    """64-bit DCT perceptual hash (32x32 grayscale -> 8x8 low frequencies vs. median)."""
    small = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def parse_fragment_type(filename: str, aliases: dict[str, str]) -> str:
    stem = re.sub(r"^cropped_", "", Path(filename).stem, flags=re.IGNORECASE)
    normalized = " ".join(_TRAILING_INDEX.sub("", stem).lower().split())
    return aliases.get(normalized, normalized)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def inventory_mikrobat(config: PipelineConfig) -> pd.DataFrame:
    root = paths.RAW_DIR / "mikrobat"
    if not (root / "SOURCE.json").is_file():
        raise FileNotFoundError("Mikrobat not downloaded; run `python -m scripts.download_assets`.")
    aliases = config.mikrobat_curation.fragment_type_aliases
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in sorted(p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith(".")):
            data = file.read_bytes()
            row = {
                "path": file.relative_to(paths.DATA_DIR).as_posix(),
                "filename": file.name,
                "class_name": class_dir.name,
                "fragment_type": parse_fragment_type(file.name, aliases),
                "file_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            try:
                decoded = decode_image(data)
                pixels = np.asarray(decoded.image.convert("RGB"))
                row.update(
                    format=decoded.format,
                    mode=decoded.image.mode,
                    width=decoded.width,
                    height=decoded.height,
                    phash=f"{perceptual_hash(decoded.image):016x}",
                    readable=True,
                    grayscale_content=bool(
                        (pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all()
                    ),
                )
            except InvalidImageError as exc:
                row.update(
                    format="",
                    mode="",
                    width=0,
                    height=0,
                    phash="",
                    readable=False,
                    grayscale_content=False,
                    error=str(exc),
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["image_id"] = "mikrobat-" + frame["sha256"].str[:12]
    frame["role"] = ""
    frame["exclusion_reason"] = ""
    frame["group_id"] = ""
    frame["conflicting_classes"] = ""
    return frame


def _exclude(frame: pd.DataFrame, index, reason: str) -> None:
    frame.loc[index, ["role", "exclusion_reason"]] = [EXCLUDED, reason]


def curate_mikrobat(frame: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Assign each inventory row a role (split or exclusion) and a near-duplicate group."""
    frame = frame.copy()
    for index in frame.index[~frame["readable"].astype(bool)]:
        _exclude(frame, index, "unreadable")

    # 1. Byte-identical images filed under more than one class: one probe copy per hash.
    readable = frame[frame["role"] == ""]
    classes_per_hash = readable.groupby("sha256")["class_name"].nunique()
    conflict_hashes = set(classes_per_hash[classes_per_hash > 1].index)
    for sha, rows in readable[readable["sha256"].isin(conflict_hashes)].sort_values("path").groupby("sha256"):
        keep = rows.index[0]
        frame.loc[keep, ["role", "group_id", "conflicting_classes"]] = [
            ds.AMBIGUITY_PROBE,
            f"conflict-{sha[:12]}",
            "|".join(sorted(rows["class_name"].unique())),
        ]
        for index in rows.index[1:]:
            _exclude(frame, index, "label-conflict duplicate copy")

    # 2. Within-class exact duplicates: keep the first path.
    pending = frame[frame["role"] == ""].sort_values("path")
    for index in pending.index[pending.duplicated("sha256", keep="first")]:
        _exclude(frame, index, "exact duplicate within class")

    # 3. Near-duplicate groups, bounded by the cross-class null distribution.
    eligible = frame[frame["role"] == ""].sort_values("path")
    hashes = [int(value, 16) for value in eligible["phash"]]
    labels = eligible["class_name"].tolist()
    pairs = [(i, j, hamming(hashes[i], hashes[j])) for i, j in combinations(range(len(hashes)), 2)]
    cross_class = np.array([d for i, j, d in pairs if labels[i] != labels[j]])
    percentile = config.mikrobat_curation.near_duplicate_null_percentile
    bound = float(np.percentile(cross_class, percentile))
    union = _UnionFind(len(hashes))
    linked = [(i, j, d) for i, j, d in pairs if d < bound]
    for i, j, _ in linked:
        union.union(i, j)
    for position, index in enumerate(eligible.index):
        frame.loc[index, "group_id"] = f"g{union.find(position):04d}"
    eligible = frame[frame["role"] == ""]
    group_classes = eligible.groupby("group_id")["class_name"].nunique()
    mixed_groups = set(group_classes[group_classes > 1].index)
    for index in eligible.index[eligible["group_id"].isin(mixed_groups)]:
        _exclude(frame, index, "near-duplicate group spans classes")

    # 4. Held-out known-material fragment types (whole groups move together).
    eligible = frame[frame["role"] == ""]
    minimum = config.mikrobat_curation.heldout_min_images
    heldout_types = {}
    for class_name, rows in eligible.groupby("class_name"):
        candidates = sorted(
            (count, name) for name, count in rows["fragment_type"].value_counts().items() if count >= minimum
        )
        if len(candidates) < 2:
            raise ValueError(
                f"{class_name}: need >= 2 fragment types with >= {minimum} images to hold one out"
            )
        heldout_types[class_name] = candidates[0][1]
    is_heldout = eligible["fragment_type"] == eligible["class_name"].map(heldout_types)
    heldout_groups = set(eligible.loc[is_heldout, "group_id"])
    frame.loc[eligible.index[eligible["group_id"].isin(heldout_groups)], "role"] = ds.HELDOUT_KNOWN

    # 5. Stratified group split of everything left.
    eligible = frame[frame["role"] == ""]
    rng = np.random.default_rng(config.seed)
    ratios = {
        ds.TRAIN: config.splits.train,
        ds.VALIDATION: config.splits.validation,
        ds.TEST: config.splits.test,
    }
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, rows in eligible.groupby("group_id"):
        strata[(rows["class_name"].mode()[0], rows["fragment_type"].mode()[0])].append(group)
    group_size = eligible.groupby("group_id").size()
    assignment = {}
    for stratum in sorted(strata):
        groups = sorted(strata[stratum])
        rng.shuffle(groups)
        total = int(group_size[groups].sum())
        filled = dict.fromkeys(ratios, 0)
        for group in groups:
            # Greedy: the split furthest below its target share receives the group.
            chosen = max(ratios, key=lambda name: (ratios[name] * total - filled[name], ratios[name]))
            assignment[group] = chosen
            filled[chosen] += int(group_size[group])
    frame.loc[eligible.index, "role"] = eligible["group_id"].map(assignment)

    sizes = frame[frame["group_id"].str.startswith("g")].groupby("group_id").size()
    summary = {
        "exact_label_conflict_hashes": len(conflict_hashes),
        "near_duplicate_rule": (
            f"pHash Hamming distance < {bound:g}, the {percentile:g}th percentile of {cross_class.size} "
            "cross-class pairs (images of different species cannot share a specimen)"
        ),
        "near_duplicate_bound": bound,
        "near_duplicate_links": len(linked),
        "near_duplicate_groups_with_multiple_images": int((sizes > 1).sum()),
        "largest_near_duplicate_group": int(sizes.max()),
        "cross_class_near_duplicate_groups": len(mixed_groups),
        "heldout_fragment_types": heldout_types,
    }
    return frame, summary


def prepare_dimpsar(config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    source = config.sources.dimpsar
    parquet = paths.RAW_DIR / "dimpsar" / Path(source.file).name
    if not parquet.is_file():
        raise FileNotFoundError("DIMPSAR not downloaded; run `python -m scripts.download_assets`.")
    table = pd.read_parquet(parquet)
    features = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"]
    names = features["label"]["names"]
    counts = table["label"].value_counts().sort_index()
    rng = np.random.default_rng(config.seed)
    class_order = rng.permutation(len(names)).tolist()
    n_calibration = round(config.dimpsar_ood.calibration_class_fraction * len(names))
    calibration_classes = set(class_order[:n_calibration])

    target_root = paths.PROCESSED_DIR / "dimpsar"
    rows = []
    for label in sorted(counts.index):
        subset = table[table["label"] == label].reset_index(drop=True)
        role = ds.OOD_CALIBRATION if label in calibration_classes else ds.OOD_EVALUATION
        seen: set[str] = set()
        for position in rng.permutation(len(subset)):
            data = subset.loc[position, "image"]["bytes"]
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen:
                continue
            try:
                decoded = decode_image(data)
            except InvalidImageError:
                continue
            seen.add(sha)
            destination = target_root / role / names[label] / f"{sha[:12]}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            decoded.image.save(buffer, format="PNG")
            destination.write_bytes(buffer.getvalue())
            rows.append(
                {
                    "image_id": f"dimpsar-{sha[:12]}",
                    "dataset": DIMPSAR,
                    "class_name": names[label],
                    "fragment_type": "",
                    "group_id": f"dimpsar-{names[label]}",
                    "sha256": sha,
                    "path": destination.relative_to(paths.DATA_DIR).as_posix(),
                    "role": role,
                }
            )
            if len(seen) == config.dimpsar_ood.images_per_class:
                break
    summary = {
        "actual_images": len(table),
        "actual_classes": int(counts.size),
        "images_per_class": {names[i]: int(c) for i, c in counts.items()},
        "calibration_classes": sorted(names[i] for i in calibration_classes),
        "evaluation_classes": sorted(names[i] for i in set(range(len(names))) - calibration_classes),
    }
    return pd.DataFrame(rows), summary


def _counts(frame: pd.DataFrame, by: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame.groupby(by).size().items()}


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def write_outputs(
    config: PipelineConfig,
    mikrobat: pd.DataFrame,
    curation: dict,
    dimpsar: pd.DataFrame,
    dimpsar_summary: dict,
) -> dict:
    for directory in (paths.SPLITS_DIR, paths.METADATA_DIR, paths.DOCS_REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    mikrobat = mikrobat.assign(dataset=MIKROBAT)
    # The probe split carries the conflicting labels, since no single class is correct.
    probe = mikrobat["role"] == ds.AMBIGUITY_PROBE
    split_rows = mikrobat.assign(
        class_name=mikrobat["class_name"].where(~probe, mikrobat["conflicting_classes"])
    )
    splits = pd.concat([split_rows[split_rows["role"].isin(ds.ALL_SPLITS)], dimpsar], ignore_index=True)
    for name in ds.ALL_SPLITS:
        splits[splits["role"] == name].sort_values("path")[ds.SPLIT_COLUMNS].to_csv(
            ds.split_path(name), index=False
        )
    mikrobat.to_csv(paths.METADATA_DIR / "image_metadata.csv", index=False)

    receipt = json.loads((paths.RAW_DIR / "mikrobat" / "SOURCE.json").read_text(encoding="utf-8"))
    dimpsar_receipt = json.loads((paths.RAW_DIR / "dimpsar" / "SOURCE.json").read_text(encoding="utf-8"))
    readable = mikrobat[mikrobat["readable"].astype(bool)]
    usable = mikrobat[mikrobat["role"].isin([ds.TRAIN, ds.VALIDATION, ds.TEST, ds.HELDOUT_KNOWN])]
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "generated_at": generated_at,
        "pipeline_config_version": config.version,
        "seed": config.seed,
        "primary": {
            "dataset": MIKROBAT,
            "source": config.sources.mikrobat.repository,
            "commit": receipt["commit"],
            "archive_sha256": receipt["archive_sha256"],
            "license_note": config.sources.mikrobat.license_note,
            "download_date": receipt["downloaded_at"][:10],
            "reported_images": config.sources.mikrobat.reported_images,
            "actual_images": len(mikrobat),
            "readable_images": len(readable),
            "unique_images": int(readable["sha256"].nunique()),
            "usable_images": len(usable),
            "classes": sorted(mikrobat["class_name"].unique()),
            "images_per_class": _counts(mikrobat, "class_name"),
            "usable_images_per_class": _counts(usable, "class_name"),
            "formats": _counts(readable, "format"),
            "colour_modes": _counts(readable, "mode"),
            "grayscale_content_images": int(readable["grayscale_content"].astype(bool).sum()),
            "dimensions": _counts(
                readable.assign(size=readable["width"].astype(str) + "x" + readable["height"].astype(str)),
                "size",
            ),
            "fragment_types": {
                c: _counts(rows, "fragment_type") for c, rows in readable.groupby("class_name")
            },
            "fragment_types_in_multiple_classes_after_curation": sorted(
                name for name, n in usable.groupby("fragment_type")["class_name"].nunique().items() if n > 1
            ),
            "exclusions": _counts(mikrobat[mikrobat["role"] == EXCLUDED], "exclusion_reason"),
            "ambiguity_probe_images": int(probe.sum()),
            "verified": True,
        },
        "far_ood": {
            "dataset": DIMPSAR,
            "source": config.sources.dimpsar.repository,
            "revision": dimpsar_receipt["revision"],
            "sha256": dimpsar_receipt["sha256"],
            "license": config.sources.dimpsar.license,
            "download_date": dimpsar_receipt["downloaded_at"][:10],
            "reported_images": config.sources.dimpsar.reported_images,
            **dimpsar_summary,
            "selected_calibration_images": int((dimpsar["role"] == ds.OOD_CALIBRATION).sum()),
            "selected_evaluation_images": int((dimpsar["role"] == ds.OOD_EVALUATION).sum()),
            "verified": True,
        },
    }
    split_metadata = {
        "generated_at": generated_at,
        "seed": config.seed,
        "ratios": config.splits.model_dump(),
        "split_unit": "near-duplicate group",
        "stratification": "class x fragment type",
        "heldout_rule": (
            f"per class, the smallest fragment type with >= {config.mikrobat_curation.heldout_min_images} eligible "
            "images is withheld from training and the reference library"
        ),
        **curation,
        "counts": {name: _counts(splits[splits["role"] == name], "class_name") for name in ds.ALL_SPLITS},
        "dimpsar_images_per_class": config.dimpsar_ood.images_per_class,
        "limitation": ds.OOD_CALIBRATION_LIMITATION,
        "leakage_note": (
            "Mikrobat publishes no specimen or source-micrograph identifiers. Exact duplicates and pHash "
            "near-duplicates are grouped before splitting, but visual inspection shows that some images are "
            "overlapping crops of one source micrograph with large pHash distances; such pairs cannot be "
            "detected reliably and may still cross split boundaries, which can make test metrics optimistic."
        ),
    }
    (paths.METADATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (paths.METADATA_DIR / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2), encoding="utf-8"
    )
    (paths.DOCS_REPORTS_DIR / "dataset_report.md").write_text(
        render_report(manifest, split_metadata), encoding="utf-8"
    )
    return split_metadata


def render_report(manifest: dict, split_metadata: dict) -> str:
    primary, ood = manifest["primary"], manifest["far_ood"]
    classes = primary["classes"]
    counts = split_metadata["counts"]
    split_rows = [
        [name] + [counts[name].get(c, 0) for c in classes] + [sum(counts[name].values())]
        for name in (ds.TRAIN, ds.VALIDATION, ds.TEST, ds.HELDOUT_KNOWN)
    ]
    fragment_rows = [
        [
            c,
            fragment,
            count,
            "held out" if split_metadata["heldout_fragment_types"].get(c) == fragment else "",
        ]
        for c, fragments in primary["fragment_types"].items()
        for fragment, count in fragments.items()
    ]
    return "\n".join(
        [
            "# Dataset Report",
            "",
            f"_Generated by `python -m ml.training.prepare_dataset` at {manifest['generated_at']} "
            f"(config {manifest['pipeline_config_version']}, seed {manifest['seed']}). Do not edit by hand._",
            "",
            "## Primary dataset: Mikrobat",
            "",
            _table(
                ["Property", "Value"],
                [
                    ["Source", f"{primary['source']} @ `{primary['commit'][:12]}`"],
                    ["License", primary["license_note"]],
                    ["Downloaded", primary["download_date"]],
                    ["Reported images (repository README)", primary["reported_images"]],
                    ["Actual files", primary["actual_images"]],
                    ["Readable", primary["readable_images"]],
                    ["Unique (SHA-256)", primary["unique_images"]],
                    ["Usable (train/validation/test/held-out)", primary["usable_images"]],
                    ["Ambiguity-probe images (label conflicts)", primary["ambiguity_probe_images"]],
                    ["Formats", primary["formats"]],
                    ["Colour modes", primary["colour_modes"]],
                    [
                        "Images with identical R/G/B channels (grayscale content)",
                        primary["grayscale_content_images"],
                    ],
                    ["Dimensions", primary["dimensions"]],
                ],
            ),
            "",
            "### Images per class",
            "",
            _table(
                ["Class", "Files", "Usable"],
                [
                    [c, primary["images_per_class"][c], primary["usable_images_per_class"].get(c, 0)]
                    for c in classes
                ],
            ),
            "",
            "### Exclusions",
            "",
            _table(["Reason", "Images"], [[k, v] for k, v in primary["exclusions"].items()]),
            "",
            "### Fragment types (readable files, before exclusions)",
            "",
            _table(["Class", "Fragment type", "Files", "Role"], fragment_rows),
            "",
            "Fragment types present in more than one class after curation: "
            f"{', '.join(primary['fragment_types_in_multiple_classes_after_curation']) or 'none'}. "
            "Where this is none, species and fragment type are confounded and the classifier may partly learn "
            "fragment appearance rather than species-specific traits.",
            "",
            "## Splits",
            "",
            f"- Split unit: {split_metadata['split_unit']}; stratified by {split_metadata['stratification']}; "
            f"target ratios {split_metadata['ratios']}.",
            f"- Near-duplicate rule: {split_metadata['near_duplicate_rule']}.",
            f"- Near-duplicate links: {split_metadata['near_duplicate_links']}; multi-image groups: "
            f"{split_metadata['near_duplicate_groups_with_multiple_images']} (largest "
            f"{split_metadata['largest_near_duplicate_group']}); groups spanning classes: "
            f"{split_metadata['cross_class_near_duplicate_groups']}.",
            f"- Held-out known-material rule: {split_metadata['heldout_rule']}.",
            "",
            _table(["Split", *classes, "Total"], split_rows),
            "",
            f"Ambiguity probe: {sum(counts[ds.AMBIGUITY_PROBE].values())} byte-identical images filed under "
            f"{', '.join(counts[ds.AMBIGUITY_PROBE])}. They are excluded from training, references and test, and are "
            "used only to observe REVIEW_REQUIRED behaviour.",
            "",
            f"**Leakage note.** {split_metadata['leakage_note']}",
            "",
            "## Far-OOD negative set: DIMPSAR",
            "",
            _table(
                ["Property", "Value"],
                [
                    ["Source", f"{ood['source']} @ `{ood['revision'][:12]}`"],
                    ["License", ood["license"]],
                    ["Reported images", ood["reported_images"]],
                    ["Actual images", ood["actual_images"]],
                    ["Actual classes", ood["actual_classes"]],
                    ["Calibration classes", ", ".join(ood["calibration_classes"])],
                    ["Evaluation classes", ", ".join(ood["evaluation_classes"])],
                    ["Selected calibration images", ood["selected_calibration_images"]],
                    ["Selected evaluation images", ood["selected_evaluation_images"]],
                ],
            ),
            "",
            "DIMPSAR is never a training class. Calibration and evaluation subsets use disjoint DIMPSAR classes.",
            "",
            "## Scientific limitation",
            "",
            split_metadata["limitation"],
            "",
            "## Dataset lock checklist",
            "",
            "- [x] Sources verified and pinned (commit / revision / SHA-256)",
            "- [x] License and usage terms reviewed",
            "- [x] Files downloaded and readable",
            "- [x] Actual counts and class list recorded",
            "- [x] Duplicate analysis (SHA-256 exact, pHash near-duplicate)",
            "- [x] Specimen leakage considered (group-aware split, leakage note)",
            "- [x] Train / validation / test / held-out split generated",
            "",
            "Image-quality bounds are calibrated in a later step; see `evaluation_report.md`.",
            "",
        ]
    )


def main() -> int:
    config = load_pipeline_config()
    curated, curation = curate_mikrobat(inventory_mikrobat(config), config)
    dimpsar, dimpsar_summary = prepare_dimpsar(config)
    split_metadata = write_outputs(config, curated, curation, dimpsar, dimpsar_summary)
    print(
        json.dumps(
            {
                key: split_metadata[key]
                for key in (
                    "near_duplicate_rule",
                    "near_duplicate_links",
                    "cross_class_near_duplicate_groups",
                    "heldout_fragment_types",
                    "counts",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
