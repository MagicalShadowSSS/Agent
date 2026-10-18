"""Evaluate simple, training-free global fusion variants on stored view vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unified_raw_qwen_candidates import (
    DATASETS,
    TEXT_FIELDS,
    VIEWS,
    load_text_vectors,
    read_jsonl,
    visual_indexes,
    dump,
)


def load_views(root: Path, dataset: str):
    rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
    n = sum(row["role"] == "query" for row in rows)
    sources, targets = rows[:n], rows[n:]
    scores, valid = {}, {}
    source_has, target_has = {}, {}
    for field in TEXT_FIELDS:
        vectors, present = load_text_vectors(root, dataset, field, len(rows), 4096)
        scores[field] = vectors[:n] @ vectors[n:].T
        source_has[field] = present[:n]
        target_has[field] = present[n:]
        valid[field] = source_has[field][:, None] & target_has[field][None, :]
    sv, tv = visual_indexes(dataset)
    image_dim = len(next(iter(sv.values()), next(iter(tv.values()))))
    image_s = np.stack([sv[row["uri"]] if row["uri"] in sv else np.zeros(image_dim, np.float32) for row in sources])
    image_t = np.stack([tv[row["uri"]] if row["uri"] in tv else np.zeros(image_dim, np.float32) for row in targets])
    scores["image"] = image_s @ image_t.T
    source_has["image"] = np.linalg.norm(image_s, axis=1) > 1e-8
    target_has["image"] = np.linalg.norm(image_t, axis=1) > 1e-8
    valid["image"] = source_has["image"][:, None] & target_has["image"][None, :]
    return rows, n, scores, valid, source_has, target_has


def positions(root: Path, dataset: str, matrix: np.ndarray, k: int = 20):
    rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
    n = sum(row["role"] == "query" for row in rows)
    targets = rows[n:]
    labels = {r["source"]: r["gold"] for r in read_jsonl(root / "labels" / f"{dataset}.jsonl") if r["gold"]}
    pos = []
    take = min(k, matrix.shape[1])
    # Full sorting every target row dominates the sweep; partition first, then
    # stably sort only the retained Top-k candidates.
    candidate_idx = np.argpartition(-matrix, take - 1, axis=1)[:, :take]
    for i, row in enumerate(rows[:n]):
        local = candidate_idx[i]
        order = local[np.argsort(-matrix[i, local], kind="stable")]
        ranked = [targets[j]["uri"] for j in order]
        gold = labels.get(row["uri"], "")
        pos.append(ranked.index(gold) + 1 if gold in ranked else 999)
    return pos


def metric(pos):
    n = len(pos)
    return {
        "queries": n,
        "hits1": sum(x == 1 for x in pos) / n,
        "hits10": sum(x <= 10 for x in pos) / n,
        "hits20": sum(x <= 20 for x in pos) / n,
        "gap_pp": 100 * sum(1 < x <= 10 for x in pos) / n,
    }


def make_score(scores, valid, source_has, target_has, image_weight=1.0, mode="sqrt"):
    weights = {"name": 1.0, "attribute": 1.0, "relation": 1.0, "image": image_weight}
    fused = None
    valid_count = None
    for view in VIEWS:
        term = weights[view] * np.where(valid[view], scores[view], 0.0)
        fused = term if fused is None else fused + term
        count = weights[view] * valid[view].astype(np.float32)
        valid_count = count if valid_count is None else valid_count + count
    if mode == "sqrt":
        count_s = sum(weights[v] * source_has[v].astype(np.float32) for v in VIEWS)
        count_t = sum(weights[v] * target_has[v].astype(np.float32) for v in VIEWS)
        return fused / np.maximum(np.sqrt(count_s[:, None] * count_t[None, :]), 1.0)
    if mode == "pair_mean":
        return fused / np.maximum(valid_count, 1.0)
    if mode == "source_mean":
        count_s = sum(weights[v] * source_has[v].astype(np.float32) for v in VIEWS)
        return fused / np.maximum(count_s[:, None], 1.0)
    if mode == "sum":
        return fused
    raise ValueError(mode)


def make_rank_score(scores, valid, image_weight=1.0, rrf_k=60.0):
    """Fuse per-view reciprocal ranks without training or dataset-specific rules."""
    fused = None
    for view in VIEWS:
        matrix = np.where(valid[view], scores[view], -np.inf)
        order = np.argsort(-matrix, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[np.arange(order.shape[0])[:, None], order] = np.arange(1, order.shape[1] + 1, dtype=np.float32)
        contribution = (1.0 / (rrf_k + ranks)) * valid[view]
        if view == "image":
            contribution *= image_weight
        fused = contribution if fused is None else fused + contribution
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = {"root": str(root), "variants": {}}
    for dataset in args.datasets:
        _, _, scores, valid, source_has, target_has = load_views(root, dataset)
        for mode in ("sqrt", "pair_mean", "source_mean", "sum"):
            for image_weight in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.25, 1.5):
                name = f"{mode}_image{image_weight:g}"
                matrix = make_score(scores, valid, source_has, target_has, image_weight, mode)
                report["variants"].setdefault(name, {})[dataset] = metric(positions(root, dataset, matrix))
        for image_weight in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            name = f"rrf_image{image_weight:g}"
            matrix = make_rank_score(scores, valid, image_weight)
            report["variants"].setdefault(name, {})[dataset] = metric(positions(root, dataset, matrix))
    dump(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
