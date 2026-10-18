"""Raw-only four-view retrieval with one direct score.

This is intentionally a small end-to-end baseline: raw names, attributes and
relations are serialized separately, encoded on this run with a frozen LaBSE,
and averaged with the released independent image feature.  No old embedding,
ranking, alignment or checkpoint is read during ``generate``.
"""
from __future__ import annotations
import os

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from unified_raw_qwen_candidates import (DATASETS, MMKG, dataset_dir, label,
    parse_fact, read_map, translated_names, write_jsonl, read_jsonl)

MODEL = Path(os.environ.get("SGMEA_TEXT_MODEL", "models/LaBSE"))
VIEWS = ("name", "attribute", "relation", "image")


def original_relation_paths(dataset, side):
    # The files are part of the local raw release, not generated candidate
    # artifacts. Their labels are used only to render the original relation.
    if dataset in DATASETS[:3]:
        language = dataset[:2] if side == 1 else "en"
        return [Path(os.environ.get("SGMEA_DBP_ROOT", "data/DBP15K")) /
                dataset / f"{language}_rel_triples"]
    graph = "FB15K" if side == 1 else ("DB15K" if dataset == "FBDB15K" else "YAGO15K")
    return [Path(os.environ.get("SGMEA_MMKGB_ROOT", "data/mmkb")) / graph / f"{graph}_EntityTriples.txt"]


def recover_relation_labels(dataset, side):
    root = dataset_dir(dataset)
    ids = read_map(root / f"ent_ids_{side}")
    wanted = set(ids.values())
    pairs = defaultdict(set)
    for path in original_relation_paths(dataset, side):
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                bits = line.split()
                if len(bits) < 3:
                    continue
                h, p, t = [x.strip().strip("<>") for x in bits[:3]]
                if h in wanted and t in wanted:
                    pairs[(h, t)].add(p)
    # Match each normalized edge to a unique original predicate by endpoint.
    labels_by_subject = defaultdict(list)
    with (root / f"triples_{side}").open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            fact = parse_fact(line)
            if not fact:
                continue
            h, numeric_p, t = fact
            h, t = ids.get(h, h), ids.get(t, t)
            options = pairs.get((h, t), set())
            predicate = next(iter(options)) if len(options) == 1 else numeric_p
            neighbor = label(t) or t
            readable = label(predicate) or predicate
            labels_by_subject[h].append(f"{readable}: {neighbor}")
    return labels_by_subject


def raw_records(dataset):
    root = dataset_dir(dataset)
    source_ids, target_ids = read_map(root / "ent_ids_1"), read_map(root / "ent_ids_2")
    all_ids = {**source_ids, **target_ids}
    records = {uri: {"uri": uri, "name": "", "attribute": [], "relation": []}
               for uri in set(source_ids.values()) | set(target_ids.values())}
    names = translated_names(dataset)
    for entity_id, uri in all_ids.items():
        if dataset in DATASETS[:3]:
            records[uri]["name"] = names.get(entity_id, "") or label(uri)
        elif not uri.startswith("/m/"):
            records[uri]["name"] = label(uri)
    for filename in ("att_triples1", "att_triples2"):
        path = root / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                fact = parse_fact(line)
                if fact and fact[0] in records:
                    records[fact[0]]["attribute"].append(
                        f"{label(fact[1]) or fact[1]}: {fact[2]}")
    for side in (1, 2):
        semantic = recover_relation_labels(dataset, side)
        for uri, values in semantic.items():
            if uri in records:
                records[uri]["relation"].extend(values)
    for rec in records.values():
        rec["attribute"] = sorted(set(rec["attribute"]))
        rec["relation"] = sorted(set(rec["relation"]))
        rec["attribute_text"] = "\n".join(rec["attribute"])
        rec["relation_text"] = "\n".join(rec["relation"])
        rec["role"] = "source" if rec["uri"] in set(source_ids.values()) else "target"
    return [records[u] for u in source_ids.values()], [records[u] for u in target_ids.values()]


def images(dataset, sources, targets):
    filename = (f"{dataset}_GA_id_img_feature_dict.pkl" if dataset in DATASETS[:3]
                else f"{dataset}_id_img_feature_dict.pkl")
    with (MMKG / "pkls" / filename).open("rb") as stream:
        features = pickle.load(stream)
    result = []
    for side, rows in ((1, sources), (2, targets)):
        mapping = {uri: int(eid) for eid, uri in read_map(dataset_dir(dataset) / f"ent_ids_{side}").items()}
        values = []
        for row in rows:
            value = features.get(mapping[row["uri"]])
            if value is None:
                values.append(None)
                continue
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(value)
            values.append(value / norm if np.isfinite(value).all() and norm > 1e-12 else None)
        result.append(values)
    dim = len(next(v for v in result[0] + result[1] if v is not None))
    out = []
    for values in result:
        has = np.array([v is not None for v in values], dtype=bool)
        mat = np.zeros((len(values), dim), dtype=np.float32)
        for i, value in enumerate(values):
            if value is not None:
                mat[i] = value
        out.append((mat, has))
    return out


def encode_fields(sources, targets, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL), device=device, local_files_only=True)
    model.eval(); model.max_seq_length = 512
    result = {}
    for field in ("name", "attribute", "relation"):
        key = field if field == "name" else field + "_text"
        texts = [r[key] for r in sources + targets]
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        result[field] = (vectors[:len(sources)], vectors[len(sources):],
                         np.array([bool(x.strip()) for x in texts[:len(sources)]], bool),
                         np.array([bool(x.strip()) for x in texts[len(sources):]], bool))
    del model
    return result


def generate(dataset, root, device, limit):
    started = time.time()
    sources, targets = raw_records(dataset)
    all_sources = sources
    if limit > 0:
        sources = sources[:limit]
    write_jsonl(root / "catalog" / f"{dataset}_sources.jsonl", sources)
    write_jsonl(root / "catalog" / f"{dataset}_targets.jsonl", targets)
    encoded = encode_fields(sources, targets, device)
    img = images(dataset, sources, targets)
    scores = {}
    valid = {}
    for field in ("name", "attribute", "relation"):
        left, right, sh, th = encoded[field]
        scores[field] = left @ right.T
        valid[field] = sh[:, None] & th[None, :]
    scores["image"] = img[0][0] @ img[1][0].T
    valid["image"] = img[0][1][:, None] & img[1][1][None, :]
    total = np.zeros_like(scores["name"])
    counts_s = np.zeros(len(sources), dtype=np.float32)
    counts_t = np.zeros(len(targets), dtype=np.float32)
    for field in VIEWS:
        total += np.where(valid[field], scores[field], 0)
        counts_s += valid[field].any(1)
        counts_t += valid[field].any(0)
    total /= np.maximum(np.sqrt(counts_s[:, None] * counts_t[None, :]), 1)
    ranked = []
    for i, source in enumerate(sources):
        order = np.argsort(-total[i], kind="stable")[:20]
        ranked.append({"source": source["uri"], "rank": [targets[j]["uri"] for j in order],
                       "scores": [float(total[i, j]) for j in order],
                       "views_present": {k: bool(valid[k][i].any()) for k in VIEWS}})
    write_jsonl(root / "rankings" / "all" / f"{dataset}.jsonl", ranked)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / f"{dataset}.json").write_text(json.dumps({
        "dataset": dataset, "queries": len(sources), "targets": len(targets),
        "all_raw_sources": len(all_sources), "device": device,
        "encoder": "frozen LaBSE", "fields": list(VIEWS),
        "labels_read": False, "ill_used_for_generation": False,
        "elapsed_sec": time.time() - started}, indent=2) + "\n")
    print(json.dumps({"dataset": dataset, "queries": len(sources), "targets": len(targets),
                      "elapsed_sec": time.time() - started}), flush=True)


def evaluate(dataset, root, labels_root):
    ranks = {r["source"]: r["rank"] for r in read_jsonl(root / "rankings" / "all" / f"{dataset}.jsonl")}
    labels = [json.loads(x) for x in (labels_root / "labels" / f"{dataset}.jsonl").open()]
    missing = sum(r["source"] not in ranks for r in labels)
    if missing:
        raise ValueError(f"{dataset}: {missing}/{len(labels)} labelled queries have no ranking; accuracy is invalid")
    pos = np.array([ranks.get(r["source"], []).index(r["gold"]) + 1
                    if r["gold"] in ranks.get(r["source"], []) else 10**9 for r in labels])
    out = {"dataset": dataset, "queries": len(pos), "H1": float(np.mean(pos <= 1) * 100),
           "H10": float(np.mean(pos <= 10) * 100), "H20": float(np.mean(pos <= 20) * 100)}
    out["gap_H20_H1"] = out["H20"] - out["H1"]
    out["gap_H10_H1"] = out["H10"] - out["H1"]
    (root / "evaluation").mkdir(parents=True, exist_ok=True)
    (root / "evaluation" / f"{dataset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--stage", choices=("generate", "evaluate"), required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--labels-root", default="output/candidate_e2e_raw_qwen_simple_20261009")
    a = p.parse_args(); root = Path(a.root)
    if a.stage == "generate":
        for ds in a.datasets: generate(ds, root, a.device, a.limit)
    else:
        for ds in a.datasets: evaluate(ds, root, Path(a.labels_root))


if __name__ == "__main__": main()
