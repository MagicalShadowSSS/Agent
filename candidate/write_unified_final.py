"""Write/evaluate one globally fixed raw-field candidate protocol.

Ranking reads only persisted view vectors and availability masks.  Labels are
opened after the Top-20 files have been written, solely for the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sweep_unified_fusion import load_views
from unified_raw_qwen_candidates import DATASETS, VIEWS, read_jsonl


def rank_matrix(data, weights, use_image=True, normalization="sqrt"):
    scores, valid, source_has, target_has = data
    fused = None
    denominator = None
    for view in VIEWS:
        current_valid = valid[view]
        if view == "image" and not use_image:
            current_valid = np.zeros_like(current_valid, dtype=bool)
        term = weights[view] * np.where(current_valid, scores[view], 0.0).astype(np.float32)
        count = weights[view] * current_valid.astype(np.float32)
        fused = term if fused is None else fused + term
        denominator = count if denominator is None else denominator + count
    if normalization == "pair_mean":
        return fused / np.maximum(denominator, 1.0)
    source_count = None
    target_count = None
    for view in VIEWS:
        source = source_has[view].copy()
        target = target_has[view].copy()
        if view == "image" and not use_image:
            source[:] = False
            target[:] = False
        source_count = weights[view] * source.astype(np.float32) if source_count is None else source_count + weights[view] * source
        target_count = weights[view] * target.astype(np.float32) if target_count is None else target_count + weights[view] * target
    return fused / np.maximum(np.sqrt(source_count[:, None] * target_count[None, :]), 1.0)


def rank_rrf_matrix(data, weights, use_image=True, rrf_k=60.0):
    """Fuse view rankings instead of raw cosine scales.

    Each available view casts one reciprocal-rank vote.  ``weights`` are
    fixed global evidence priors, not learned or query-dependent gates.  This
    keeps all available views in the decision while avoiding the assumption
    that Qwen text cosine and VGG image cosine have the same scale.
    """
    scores, valid, _, _ = data
    fused = None
    for view in VIEWS:
        current_valid = valid[view]
        if view == "image" and not use_image:
            current_valid = np.zeros_like(current_valid, dtype=bool)
        current = np.where(current_valid, scores[view], -np.inf)
        order = np.argsort(-current, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[np.arange(order.shape[0])[:, None], order] = np.arange(
            1, order.shape[1] + 1, dtype=np.float32
        )
        contribution = weights[view] / (rrf_k + ranks)
        contribution = np.where(current_valid, contribution, 0.0).astype(np.float32)
        fused = contribution if fused is None else fused + contribution
    return fused


def rank_agreement_matrix(data, use_image=True, agreement_lambda=0.2):
    """Equal-view late interaction with a fixed cross-view agreement term.

    Scores are mapped from cosine range to [0, 1].  The first term is the
    mean evidence score and the second is the mean product over every pair of
    available views.  The product rewards candidates supported jointly by
    several views; it is not a reliability gate and every available view is
    still included in the mean.  No parameter is fitted on labels.
    """
    scores, valid, _, _ = data
    values = []
    masks = []
    for view in VIEWS:
        current_valid = valid[view]
        if view == "image" and not use_image:
            current_valid = np.zeros_like(current_valid, dtype=bool)
        values.append(np.where(current_valid, (scores[view] + 1.0) / 2.0, 0.0).astype(np.float32))
        masks.append(current_valid)
    stack = np.stack(values, axis=0)
    mask = np.stack(masks, axis=0)
    count = mask.sum(axis=0).astype(np.float32)
    mean = stack.sum(axis=0) / np.maximum(count, 1.0)
    pair_sum = np.zeros_like(mean)
    pair_count = np.zeros_like(mean)
    for left in range(len(VIEWS)):
        for right in range(left + 1, len(VIEWS)):
            pair_valid = mask[left] & mask[right]
            pair_sum += np.where(pair_valid, stack[left] * stack[right], 0.0)
            pair_count += pair_valid.astype(np.float32)
    agreement = pair_sum / np.maximum(pair_count, 1.0)
    return mean + float(agreement_lambda) * agreement


def rank_csls_matrix(data, weights, use_image=True, normalization="sqrt", csls_k=10.0):
    """Apply standard CSLS hubness correction after four-view fusion."""
    base = rank_matrix(data, weights, use_image, normalization)
    k = max(1, min(int(csls_k), base.shape[0], base.shape[1]))
    source_density = np.partition(base, -k, axis=1)[:, -k:].mean(axis=1)
    target_density = np.partition(base, -k, axis=0)[-k:, :].mean(axis=0)
    return 2.0 * base - source_density[:, None] - target_density[None, :]


def _view_csls(scores, valid, source_has, target_has, csls_k=10):
    """Apply CSLS inside one view's available source/target subgraph.

    Missing entities are excluded from the density neighborhoods rather than
    being treated as zero-similarity neighbors.  The returned matrix keeps
    the original global shape and uses zero for unavailable pairs so that the
    caller can combine views with the same availability normalization.
    """
    result = np.zeros_like(scores, dtype=np.float32)
    source_idx = np.flatnonzero(source_has)
    target_idx = np.flatnonzero(target_has)
    if not len(source_idx) or not len(target_idx):
        return result
    sub = scores[np.ix_(source_idx, target_idx)].astype(np.float32, copy=False)
    k = max(1, min(int(csls_k), sub.shape[0], sub.shape[1]))
    source_density = np.partition(sub, -k, axis=1)[:, -k:].mean(axis=1)
    target_density = np.partition(sub, -k, axis=0)[-k:, :].mean(axis=0)
    corrected = 2.0 * sub - source_density[:, None] - target_density[None, :]
    result[np.ix_(source_idx, target_idx)] = corrected
    return result


def rank_per_view_csls_matrix(data, weights, use_image=True, normalization="sqrt", csls_k=10):
    """Correct hubness per modality, then perform equal-view late fusion."""
    scores, valid, source_has, target_has = data
    fused = None
    denominator = None
    source_count = None
    target_count = None
    for view in VIEWS:
        current_source = source_has[view].copy()
        current_target = target_has[view].copy()
        current_valid = valid[view].copy()
        if view == "image" and not use_image:
            current_source[:] = False
            current_target[:] = False
            current_valid[:] = False
        corrected = _view_csls(scores[view], current_valid, current_source, current_target, csls_k)
        term = weights[view] * corrected
        count = weights[view] * current_valid.astype(np.float32)
        fused = term if fused is None else fused + term
        denominator = count if denominator is None else denominator + count
        source_count = (weights[view] * current_source.astype(np.float32)
                        if source_count is None else source_count + weights[view] * current_source)
        target_count = (weights[view] * current_target.astype(np.float32)
                        if target_count is None else target_count + weights[view] * current_target)
    if normalization == "pair_mean":
        return fused / np.maximum(denominator, 1.0)
    return fused / np.maximum(np.sqrt(source_count[:, None] * target_count[None, :]), 1.0)


def write_rankings(root, dataset, matrix, out_dir, top_k=20):
    rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
    n = sum(row["role"] == "query" for row in rows)
    targets = rows[n:]
    out_dir.mkdir(parents=True, exist_ok=True)
    take = min(top_k, matrix.shape[1])
    candidate_idx = np.argpartition(-matrix, take - 1, axis=1)[:, :take]
    written = []
    with (out_dir / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows[:n]):
            local = candidate_idx[i]
            order = local[np.argsort(-matrix[i, local], kind="stable")]
            ranked = [targets[j]["uri"] for j in order]
            item = {"source": row["uri"], "rank": ranked,
                    "scores": [float(matrix[i, j]) for j in order]}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            written.append(item)
    return written


def metric(rows, labels):
    positions = []
    for row in rows:
        gold = labels.get(row["source"], "")
        try:
            positions.append(row["rank"].index(gold) + 1)
        except ValueError:
            positions.append(999)
    n = len(positions)
    return {"queries": n, "hits1": sum(x == 1 for x in positions) / n,
            "hits10": sum(x <= 10 for x in positions) / n,
            "hits20": sum(x <= 20 for x in positions) / n,
            "gap_pp": 100 * sum(1 < x <= 10 for x in positions) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", type=float, default=1.0)
    ap.add_argument("--attribute", type=float, default=1.0)
    ap.add_argument("--relation", type=float, default=1.0)
    ap.add_argument("--image", type=float, default=1.0)
    ap.add_argument("--normalization", choices=("sqrt", "pair_mean"), default="sqrt")
    ap.add_argument("--fusion", choices=("cosine", "rrf", "agreement", "csls", "per_view_csls"), default="cosine")
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--agreement-lambda", type=float, default=0.2)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse complete persisted rankings after checking the fixed configuration")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    weights = {"name": args.name, "attribute": args.attribute,
               "relation": args.relation, "image": args.image}
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "config.json").exists():
        previous = json.loads((out / "config.json").read_text(encoding="utf-8"))
        for key, value in (("weights", weights), ("normalization", args.normalization),
                           ("fusion", args.fusion), ("top_k", 20)):
            if previous.get(key) != value:
                raise ValueError(f"Cannot resume a different configuration: {key}")
        if args.fusion in ("csls", "per_view_csls"):
            recorded_k = previous.get("csls_k")
            # Earlier per-view output omitted this metadata while always using
            # the default k=10. Permit only that legacy default to be repaired.
            if recorded_k is None and args.fusion == "per_view_csls" and args.csls_k == 10:
                pass
            elif recorded_k != args.csls_k:
                raise ValueError("Cannot resume a different CSLS neighborhood")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    protocol.update({
        "fusion": args.fusion,
        "weights": weights,
        "normalization": args.normalization,
        "rrf_k": args.rrf_k if args.fusion == "rrf" else None,
        "agreement_lambda": args.agreement_lambda if args.fusion == "agreement" else None,
        "csls_k": args.csls_k if args.fusion in ("csls", "per_view_csls") else None,
        "reliability_gate": False,
        "ranking_uses_labels": False,
    })
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    config = {
        "datasets": list(DATASETS), "views": list(VIEWS), "top_k": 20,
        "weights": weights, "normalization": args.normalization,
        "fusion": args.fusion, "rrf_k": args.rrf_k if args.fusion == "rrf" else None,
        "agreement_lambda": args.agreement_lambda if args.fusion == "agreement" else None,
        "csls_k": args.csls_k if args.fusion in ("csls", "per_view_csls") else None,
        "training": False, "task_checkpoint": None,
        "text_encoder": "frozen public Qwen3-Embedding-8B",
        "input_protocol": str(out / "protocol.json"),
        "ranking_uses_labels": False,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    report = {"config": config, "variants": {}}
    for dataset in DATASETS:
        data = None
        for variant, use_image in (("all", True), ("no_image", False)):
            ranking_path = out / "rankings" / variant / dataset / "rankings.jsonl"
            if args.resume and ranking_path.exists():
                rows = read_jsonl(ranking_path)
                input_rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
                query_uris = [r["uri"] for r in input_rows if r["role"] == "query"]
                complete = [r.get("source") for r in rows] == query_uris
                complete = complete and all(len(r.get("rank", [])) == 20 for r in rows)
                if complete:
                    labels = {r["source"]: r["gold"] for r in read_jsonl(root / "labels" / f"{dataset}.jsonl")}
                    report["variants"].setdefault(variant, {})[dataset] = metric(rows, labels)
                    print(f"Reused {dataset}/{variant}: {len(rows)} rankings", flush=True)
                    continue
            if data is None:
                _, _, scores, valid, source_has, target_has = load_views(root, dataset)
                data = (scores, valid, source_has, target_has)
            if args.fusion == "rrf":
                matrix = rank_rrf_matrix(data, weights, use_image, args.rrf_k)
            elif args.fusion == "agreement":
                matrix = rank_agreement_matrix(data, use_image, args.agreement_lambda)
            elif args.fusion == "csls":
                matrix = rank_csls_matrix(data, weights, use_image, args.normalization, args.csls_k)
            elif args.fusion == "per_view_csls":
                matrix = rank_per_view_csls_matrix(data, weights, use_image, args.normalization, args.csls_k)
            else:
                matrix = rank_matrix(data, weights, use_image, args.normalization)
            rows = write_rankings(root, dataset, matrix, out / "rankings" / variant / dataset)
            # This is the first point at which benchmark labels are read.
            labels = {r["source"]: r["gold"] for r in read_jsonl(root / "labels" / f"{dataset}.jsonl")}
            report["variants"].setdefault(variant, {})[dataset] = metric(rows, labels)
            del matrix
            print(f"Completed {dataset}/{variant}: {report['variants'][variant][dataset]}", flush=True)
        (out / "metrics.partial.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        del data
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
