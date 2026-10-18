"""Build a predicate-only relation text view from the local raw release.

This is a single-factor control for the degree-cosine generator.  It uses
only original relation endpoints and predicate labels, never entity names,
alignment pairs, candidate ranks, or a trained/checkpointed model.
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from unified_raw_qwen_candidates import (DATASETS, MODEL, dataset_dir, dump,
    label, parse_fact, read_map, sha, write_jsonl, QwenEncoder)


def original_relation_paths(dataset: str, side: int):
    if dataset in DATASETS[:3]:
        language = dataset[:2] if side == 1 else "en"
        return [Path(os.environ.get('SGMEA_DBP_ROOT', 'data/DBP15K')) /
                dataset / f"{language}_rel_triples"]
    graph = "FB15K" if side == 1 else ("DB15K" if dataset == "FBDB15K" else "YAGO15K")
    return [Path(os.environ.get('SGMEA_MMKGB_ROOT', 'data/mmkb')) / graph /
            f"{graph}_EntityTriples.txt"]


def token(value: str) -> str:
    value = str(value).strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value.rstrip(".")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def side_profiles(dataset: str, side: int):
    root = dataset_dir(dataset)
    mapping = read_map(root / f"ent_ids_{side}")
    wanted = set(mapping.values())
    predicates = defaultdict(set)
    input_hashes = {}
    endpoint_edges = 0
    for path in original_relation_paths(dataset, side):
        if not path.exists():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = file_sha(path)
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                bits = line.split()
                if len(bits) < 3:
                    continue
                h, p, t = (token(x) for x in bits[:3])
                if h in wanted and t in wanted:
                    predicates[(h, t)].add(p)
    triples = root / f"triples_{side}"
    input_hashes[str(triples)] = file_sha(triples)
    mapped_edges = 0
    for line in triples.open(encoding="utf-8", errors="ignore"):
        fact = parse_fact(line)
        if not fact:
            continue
        h_id, _, t_id = fact
        h, t = mapping.get(h_id, h_id), mapping.get(t_id, t_id)
        options = predicates.get((h, t), ())
        if not options:
            continue
        mapped_edges += 1
        for predicate in options:
            readable = label(predicate) or predicate
            predicates[("__profile__", h)].add(f"out {readable}")
            predicates[("__profile__", t)].add(f"in {readable}")
    profiles = {}
    for uri in mapping.values():
        profiles[uri] = "\n".join(sorted(predicates.get(("__profile__", uri), ())))
    endpoint_edges = sum(1 for key in predicates if isinstance(key, tuple) and len(key) == 2
                         and key[0] != "__profile__")
    return profiles, {
        "side": side,
        "entity_count": len(mapping),
        "entities_with_relation_text": sum(bool(v) for v in profiles.values()),
        "raw_endpoint_edges": endpoint_edges,
        "normalized_edges_with_predicate": mapped_edges,
        "mapping_sha256": file_sha(root / f"ent_ids_{side}"),
        "triples_sha256": file_sha(triples),
        "raw_relation_files": input_hashes,
    }


def build_dataset(root: Path, dataset: str):
    started = time.time()
    data_root = dataset_dir(dataset)
    maps = [read_map(data_root / f"ent_ids_{side}") for side in (1, 2)]
    side_data = [side_profiles(dataset, side) for side in (1, 2)]
    rows = []
    for side, mapping in ((1, maps[0]), (2, maps[1])):
        profiles = side_data[side - 1][0]
        role = "query" if side == 1 else "target"
        for uri in mapping.values():
            rows.append({"uri": uri, "relation_text": profiles.get(uri, ""), "role": role})
    inputs = root / "generated" / "inputs" / f"{dataset}_relation.jsonl"
    write_jsonl(inputs, rows)
    dump(root / "generated" / "relation_profiles" / f"{dataset}.json", {
        "dataset": dataset, "source_count": len(maps[0]), "target_count": len(maps[1]),
        "predicate_only": True, "neighbor_names": False, "alignment_labels": False,
        "sides": [side_data[0][1], side_data[1][1]],
        "input_sha256": {str(data_root / f"ent_ids_{s}"): file_sha(data_root / f"ent_ids_{s}")
                         for s in (1, 2)},
        "labels_read": False,
    })
    return rows, time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = Path(args.root)
    encoder = QwenEncoder()
    for dataset in args.datasets:
        out = root / "generated" / "embeddings" / f"{dataset}_relation.npy"
        receipt = root / "receipts" / f"{dataset}_relation.json"
        if out.exists() and receipt.exists():
            print(dataset, "relation vectors already complete; preserving", flush=True)
            continue
        rows, elapsed = build_dataset(root, dataset)
        texts = [row["relation_text"] for row in rows]
        roles = [row["role"] for row in rows]
        vectors = encoder.encode(texts, roles).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.save(stream, vectors)
        tmp.replace(out)
        profile = json.loads((root / "generated" / "relation_profiles" / f"{dataset}.json").read_text())
        dump(receipt, {
            "dataset": dataset, "model": str(MODEL), "rows": len(rows),
            "dim": int(vectors.shape[1]), "nonempty_rows": int(np.sum(np.linalg.norm(vectors, axis=1) > 1e-8)),
            "embedding_sha256": file_sha(out), "input_sha256": profile["input_sha256"],
            "profile": profile["sides"], "labels_read": False,
            "alignment_or_candidate_inputs": False, "elapsed_seconds": elapsed,
        })
        print(json.dumps({"dataset": dataset, "rows": len(rows),
                          "nonempty": int(np.sum(np.linalg.norm(vectors, axis=1) > 1e-8)),
                          "elapsed": elapsed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
