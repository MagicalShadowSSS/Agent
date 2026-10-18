"""Unified, training-free raw-field candidate generation for DBP/FB datasets.

The generator has one protocol for every dataset.  It reads raw entity,
attribute, relation and original-image files only.  Alignment files are
loaded only by ``evaluate`` after rankings have been written.  No EA model,
task checkpoint, alias expansion, anchor file or gold-based name repair is
used.
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import pickle
import re
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MMKG = Path(os.environ.get("SGMEA_MMKGB_ROOT", "data/mmkg"))
MODEL = Path(os.environ.get("SGMEA_EMBEDDING_MODEL", "models/Qwen3-Embedding-8B"))
DATASETS = ("zh_en", "ja_en", "fr_en", "FBDB15K", "FBYG15K")
TEXT_FIELDS = ("name", "attribute", "relation")
VIEWS = TEXT_FIELDS + ("image",)
TASK = "Given an entity record from one knowledge graph, retrieve a record describing exactly the same real-world entity in another knowledge graph."
CONTENT_TOKENS = 384
MAX_LENGTH = 512
DEFAULT_LIMIT = 128


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate workers can write the identical shared protocol concurrently.
    # A fixed .tmp pathname let one worker replace another worker's file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        tmp = Path(stream.name)
    tmp.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_uri(value: str) -> str:
    value = str(value).strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value.strip()


def clean_literal(value: str) -> str:
    value = str(value).strip().rstrip(" .")
    if value.startswith('"'):
        value = value[1:]
        quote = value.rfind('"')
        if quote >= 0:
            value = value[:quote]
    value = value.replace("\\\"", '"').replace("\\n", " ")
    return " ".join(value.split())


def label(value: str) -> str:
    value = strip_uri(value)
    if value.startswith("/m/"):
        return ""
    tail = value.rsplit("/", 1)[-1]
    tail = tail.rsplit("#", 1)[-1]
    return re.sub(r"[_+]+", " ", tail).strip()


def read_map(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            bits = line.rstrip("\n").split("\t", 1)
            if len(bits) == 2:
                out[bits[0].strip()] = strip_uri(bits[1])
    return out


def parse_fact(line: str):
    line = line.strip()
    if not line:
        return None
    bits = line.split("\t")
    if len(bits) >= 3:
        return strip_uri(bits[0]), strip_uri(bits[1]), clean_literal("\t".join(bits[2:]))
    bits = line.split()
    if len(bits) >= 3:
        return strip_uri(bits[0]), strip_uri(bits[1]), clean_literal(" ".join(bits[2:]))
    return None


def dataset_dir(dataset: str) -> Path:
    return MMKG / "DBP15K" / dataset if dataset in DATASETS[:3] else MMKG / dataset / "norm"


def translated_names(dataset: str) -> dict[str, str]:
    if dataset not in DATASETS[:3]:
        return {}
    path = MMKG / "DBP15K" / "translated_ent_name" / f"dbp_{dataset}.json"
    if not path.exists():
        return {}
    result = {}
    for entity_id, values in json.loads(path.read_text(encoding="utf-8")):
        if isinstance(values, list):
            result[str(entity_id)] = " ".join(str(x) for x in values if str(x).strip())
    return result


def read_pairs(path: Path, source_ids: dict[str, str], target_ids: dict[str, str]):
    pairs = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            bits = line.split()
            if len(bits) >= 2 and bits[0] in source_ids and bits[1] in target_ids:
                pairs.append((source_ids[bits[0]], target_ids[bits[1]]))
    return pairs


def load_raw(dataset: str, limit: int) -> tuple[list[dict], list[dict], list[dict]]:
    root = dataset_dir(dataset)
    source_ids = read_map(root / "ent_ids_1")
    target_ids = read_map(root / "ent_ids_2")
    all_ids = {**source_ids, **target_ids}
    by_uri = {uri: entity_id for entity_id, uri in all_ids.items()}
    source_uris = list(source_ids.values())
    target_uris = list(target_ids.values())
    pairs = read_pairs(root / "ill_ent_ids", source_ids, target_ids)
    queries = []
    seen = set()
    for source, target in pairs:
        if source not in seen:
            queries.append({"source": source, "gold": target})
            seen.add(source)
        if len(queries) >= limit:
            break
    if len(queries) < limit:
        for source in source_uris:
            if source not in seen:
                queries.append({"source": source, "gold": ""})
            if len(queries) >= limit:
                break
    names_by_id = translated_names(dataset)
    records = {uri: {"uri": uri, "name": "", "attribute": [], "relation": []}
               for uri in set(source_uris + target_uris)}
    for entity_id, uri in all_ids.items():
        if dataset in DATASETS[:3]:
            records[uri]["name"] = names_by_id.get(entity_id, "") or label(uri)
        elif uri.startswith("/m/"):
            # Raw Freebase IDs are opaque.  This is deliberately missing input.
            records[uri]["name"] = ""
        else:
            # The original target URI tail is the only readable target name.
            records[uri]["name"] = label(uri)

    for filename in ("att_triples1", "att_triples2"):
        path = root / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                fact = parse_fact(line)
                if not fact:
                    continue
                subject, predicate, value = fact
                record = records.get(subject)
                if record is not None:
                    record["attribute"].append(f"{label(predicate) or predicate}: {value}")

    for filename in ("triples_1", "triples_2"):
        path = root / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                fact = parse_fact(line)
                if not fact:
                    continue
                subject, relation, object_id = fact
                # Relation triples use integer IDs in all five raw releases.
                subject_uri = all_ids.get(subject, subject)
                object_uri = all_ids.get(object_id, object_id)
                record = records.get(subject_uri)
                if record is not None:
                    neighbor = label(object_uri) or object_uri
                    record["relation"].append(f"relation {relation}: {neighbor}")

    for record in records.values():
        record["attribute"] = sorted(record["attribute"])
        record["relation"] = sorted(record["relation"])
        record["attribute_text"] = "\n".join(record["attribute"])
        record["relation_text"] = "\n".join(record["relation"])
        record["role"] = "query" if record["uri"] in {q["source"] for q in queries} else "target"
    query_records = [records[q["source"]] for q in queries]
    target_records = [records[uri] for uri in target_uris]
    return query_records, target_records, queries


def visual_indexes(dataset: str):
    # The locally crawled FB CLIP cache is NOT safe input: its target images
    # were assigned from source entities using ill_ent_ids.  Use the released
    # per-entity visual features for all datasets through this same loader.
    filename = f"{dataset}_GA_id_img_feature_dict.pkl" if dataset in DATASETS[:3] else f"{dataset}_id_img_feature_dict.pkl"
    path = MMKG / "pkls" / filename
    with path.open("rb") as stream:
        features = pickle.load(stream)
    indexes = []
    for side in ("1", "2"):
        by_id = read_map(dataset_dir(dataset) / f"ent_ids_{side}")
        index = {}
        for entity_id, uri in by_id.items():
            value = features.get(int(entity_id))
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm <= 1e-12:
                continue
            index[uri] = vector / norm
        indexes.append(index)
    return tuple(indexes)


def prepare(args) -> None:
    root = Path(args.root)
    protocol = {
        "datasets": list(args.datasets), "limit": args.limit, "target_catalog": "all raw ent_ids_2",
        "views": list(VIEWS), "top_k": 20, "downstream_top_k": 10,
        "text_encoder": "frozen public Qwen3-Embedding-8B; no EA checkpoint and no training",
        "text_input": "raw names, raw attributes and raw outgoing relations; all records use the same serializer",
        "fb_name_rule": "source /m/... names are empty; target URI tails are preserved; no processed_names or ILL repair",
        "image_input": "released per-entity image feature files under mmkg/pkls; no ILL-derived image assignment",
        "missing_rule": "a view is valid only when both source and target have it",
        "fusion": "sum(valid cosine view scores)/sqrt(number of valid source views * number of valid target views)",
        "labels": "read only by evaluate after rankings are persisted",
    }
    dump(root / "protocol.json", protocol)
    input_hashes = {}
    for dataset in args.datasets:
        query, target, labels = load_raw(dataset, args.limit)
        write_jsonl(root / "inputs" / f"{dataset}.jsonl", query + target)
        write_jsonl(root / "labels" / f"{dataset}.jsonl", labels)
        qv, tv = visual_indexes(dataset)
        coverage = {
            "queries": len(query), "targets": len(target), "labels_with_gold": sum(bool(x["gold"]) for x in labels),
            "source_name": sum(bool(x["name"]) for x in query),
            "target_name": sum(bool(x["name"]) for x in target),
            "source_attribute": sum(bool(x["attribute_text"]) for x in query),
            "target_attribute": sum(bool(x["attribute_text"]) for x in target),
            "source_relation": sum(bool(x["relation_text"]) for x in query),
            "target_relation": sum(bool(x["relation_text"]) for x in target),
            "source_image": sum(x["uri"] in qv for x in query),
            "target_image": sum(x["uri"] in tv for x in target),
        }
        dump(root / "coverage" / f"{dataset}.json", coverage)
        input_hashes[dataset] = {
            "records": sha(root / "inputs" / f"{dataset}.jsonl"),
            "labels": sha(root / "labels" / f"{dataset}.jsonl"),
        }
        print(json.dumps({"dataset": dataset, **coverage}, ensure_ascii=False), flush=True)
    dump(root / "input_hashes.json", input_hashes)


def split_complete(tokenizer, text: str, width: int = CONTENT_TOKENS):
    if not text.strip():
        return []
    result, start = [], 0
    while start < len(text):
        remaining = text[start:]
        encoded = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)
        if len(encoded["input_ids"]) <= width:
            result.append((remaining, len(encoded["input_ids"])))
            break
        end = encoded["offset_mapping"][width - 1][1]
        while end > 0 and len(tokenizer.encode(remaining[:end], add_special_tokens=False)) > width:
            end -= 1
        if end <= 0:
            raise RuntimeError("Tokenizer produced an empty chunk")
        piece = remaining[:end]
        result.append((piece, len(tokenizer.encode(piece, add_special_tokens=False))))
        start += end
    assert "".join(piece for piece, _ in result) == text
    return result


class QwenEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        assert (MODEL / "download_complete.json").exists(), MODEL
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa", device_map={"": "cuda:0"},
                                               local_files_only=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.dim = int(self.model.config.hidden_size)
        self.cache = {}

    def encode(self, texts, roles):
        torch = self.torch
        output = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Encode each distinct (role, text) once.  The old implementation
        # mixed record indices with chunk indices, which could silently attach
        # a chunk to the wrong record when records had multiple chunks.
        unique_keys, unique_records = [], []
        key_to_uid, record_uid = {}, [None] * len(texts)
        for i, (text, role) in enumerate(zip(texts, roles)):
            key = (role, hashlib.sha256(text.encode("utf-8")).hexdigest())
            if key in self.cache:
                output[i] = self.cache[key]
                continue
            if key in key_to_uid:
                record_uid[i] = key_to_uid[key]
                continue
            uid = len(unique_keys)
            key_to_uid[key] = uid
            unique_keys.append(key)
            unique_records.append((text, role))
            record_uid[i] = uid

        chunks, chunk_owners, weights, lengths = [], [], [], []
        for uid, (text, role) in enumerate(unique_records):
            for piece, count in split_complete(self.tokenizer, text):
                formatted = f"Instruct: {TASK}\nQuery: {piece}" if role == "query" else piece
                length = len(self.tokenizer.encode(formatted, add_special_tokens=True))
                if length > MAX_LENGTH:
                    raise AssertionError((length, role))
                chunks.append(formatted)
                chunk_owners.append(uid)
                weights.append(count)
                lengths.append(length)

        ordered = sorted(range(len(chunks)), key=lambda j: lengths[j])
        sums = np.zeros((len(unique_keys), self.dim), dtype=np.float32)
        cursor = 0
        while cursor < len(ordered):
            chosen = ordered[cursor:cursor + getattr(self, 'batch_sequences', 12)]
            while len(chosen) > 1 and max(lengths[j] for j in chosen) * len(chosen) > getattr(self, 'batch_tokens', 1536):
                chosen = chosen[: max(1, len(chosen) // 2)]
            batch = [chunks[j] for j in chosen]
            with torch.inference_mode():
                tokenized = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to("cuda:0")
                hidden = self.model(**tokenized, use_cache=False).last_hidden_state[:, -1].float()
                hidden = torch.nn.functional.normalize(hidden, dim=1).cpu().numpy()
            for j, vector in zip(chosen, hidden):
                sums[chunk_owners[j]] += vector * weights[j]
            cursor += len(chosen)

        if unique_keys:
            norms = np.linalg.norm(sums, axis=1, keepdims=True)
            unique_vectors = sums / np.maximum(norms, 1e-12)
        else:
            unique_vectors = np.zeros((0, self.dim), dtype=np.float32)
        for i, (text, role) in enumerate(zip(texts, roles)):
            key = (role, hashlib.sha256(text.encode("utf-8")).hexdigest())
            if key in self.cache:
                # The first occurrence may have populated the cache earlier
                # in this loop. Every duplicate still needs its output row.
                output[i] = self.cache[key]
                continue
            uid = record_uid[i]
            if uid is not None:
                output[i] = unique_vectors[uid]
                self.cache[key] = output[i].copy()
        nonempty = np.array([bool(text.strip()) for text in texts])
        if nonempty.any() and not np.all(np.linalg.norm(output[nonempty], axis=1) > 1e-8):
            raise RuntimeError("A nonempty record received a zero embedding")
        return output


def encode(args, encoder=None) -> None:
    root = Path(args.root)
    import torch
    torch.set_num_threads(2)
    encoder = encoder if encoder is not None else QwenEncoder()
    for dataset in args.datasets:
        rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
        fields = args.fields or list(TEXT_FIELDS)
        for field in fields:
            texts = [row["name"] if field == "name" else row[field + "_text"] for row in rows]
            roles = [row["role"] for row in rows]
            indices = np.arange(len(rows))[args.shard::args.nshards]
            for block, start in enumerate(range(0, len(indices), 128)):
                selected_indices = indices[start:start + 128]
                path = root / "embeddings" / dataset / field / f"part_{args.shard}_{block:04d}.npz"
                if path.exists():
                    continue
                selected = [texts[i] for i in selected_indices]
                selected_roles = [roles[i] for i in selected_indices]
                vectors = encoder.encode(selected, selected_roles)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, indices=selected_indices, vectors=vectors,
                             present=np.array([bool(texts[i].strip()) for i in selected_indices]))
                temporary.replace(path)
                status = {"dataset": dataset, "field": field, "shard": args.shard,
                          "done_in_field": min(start + 128, len(indices)),
                          "total_in_field": len(indices), "elapsed": time.time()}
                dump(root / "status" / f"encode_{dataset}_{field}_{args.shard}.json", status)
                print(json.dumps(status, ensure_ascii=False), flush=True)


def load_text_vectors(root: Path, dataset: str, field: str, n: int, dim: int):
    vectors = np.zeros((n, dim), np.float32); present = np.zeros(n, bool); seen = np.zeros(n, bool)
    for path in sorted((root / "embeddings" / dataset / field).glob("part_*.npz")):
        with np.load(path) as block:
            indices = block["indices"]
            if seen[indices].any():
                raise RuntimeError((dataset, field, path))
            vectors[indices] = block["vectors"]; present[indices] = block["present"]; seen[indices] = True
    if not seen.all():
        raise RuntimeError(f"Missing embedding shards for {dataset}/{field}: {seen.sum()}/{n}")
    return vectors, present


def retrieve(args) -> None:
    root = Path(args.root)
    for dataset in args.datasets:
        rows = read_jsonl(root / "inputs" / f"{dataset}.jsonl")
        n = sum(row["role"] == "query" for row in rows); m = len(rows) - n
        records = {row["uri"]: row for row in rows}
        sources = rows[:n]; targets = rows[n:]
        score = {}; valid = {}; source_has = {}; target_has = {}
        for field in TEXT_FIELDS:
            vectors, present = load_text_vectors(root, dataset, field, n + m, 4096)
            score[field] = vectors[:n] @ vectors[n:].T
            source_has[field] = present[:n]; target_has[field] = present[n:]
            valid[field] = source_has[field][:, None] & target_has[field][None, :]
        source_visual, target_visual = visual_indexes(dataset)
        image_dim = len(next(iter(source_visual.values()), next(iter(target_visual.values()))))
        image_s = np.stack([source_visual[row["uri"]] if row["uri"] in source_visual else np.zeros(image_dim, np.float32) for row in sources])
        image_t = np.stack([target_visual[row["uri"]] if row["uri"] in target_visual else np.zeros(image_dim, np.float32) for row in targets])
        source_has["image"] = np.linalg.norm(image_s, axis=1) > 1e-8
        target_has["image"] = np.linalg.norm(image_t, axis=1) > 1e-8
        score["image"] = image_s @ image_t.T
        valid["image"] = source_has["image"][:, None] & target_has["image"][None, :]
        count_s = sum(source_has.values()); count_t = sum(target_has.values())
        fused = np.zeros((n, m), np.float32)
        for view in VIEWS:
            fused += np.where(valid[view], score[view], 0.0)
        no_image = fused - np.where(valid["image"], score["image"], 0.0)
        fused /= np.maximum(np.sqrt(count_s[:, None] * count_t[None, :]), 1.0)
        no_image /= np.maximum(np.sqrt((count_s - source_has["image"])[:, None] * (count_t - target_has["image"])[None, :]), 1.0)
        for name, matrix in (("all", fused), ("no_image", no_image)):
            output = []
            for i in range(n):
                order = np.argsort(-matrix[i], kind="stable")[:20]
                output.append({"source": sources[i]["uri"], "rank": [targets[j]["uri"] for j in order],
                               "scores": [float(matrix[i, j]) for j in order],
                               "views_present": {view: bool(source_has[view][i]) for view in VIEWS},
                               "target_view_counts": {view: int(target_has[view].sum()) for view in VIEWS}})
            write_jsonl(root / "rankings" / name / f"{dataset}.jsonl", output)
        dump(root / "status" / f"retrieve_{dataset}.json", {"complete": True, "queries": n, "targets": m,
             "source_views": {view: int(source_has[view].sum()) for view in VIEWS},
             "target_views": {view: int(target_has[view].sum()) for view in VIEWS}})
        print(json.dumps({"dataset": dataset, "queries": n, "targets": m,
                          "source_views": {view: int(source_has[view].sum()) for view in VIEWS}}, ensure_ascii=False), flush=True)


def evaluate(args) -> None:
    root = Path(args.root)
    report = {"protocol": json.loads((root / "protocol.json").read_text()), "datasets": {}}
    for dataset in args.datasets:
        labels = {row["source"]: row["gold"] for row in read_jsonl(root / "labels" / f"{dataset}.jsonl") if row["gold"]}
        methods = {}
        for method in ("all", "no_image"):
            rows = read_jsonl(root / "rankings" / method / f"{dataset}.jsonl")
            positions = []
            for row in rows:
                gold = labels.get(row["source"], "")
                rank = row["rank"]
                positions.append(rank.index(gold) + 1 if gold in rank else 999)
            n = len(positions)
            methods[method] = {"queries": n, "gold_queries": sum(bool(labels.get(r["source"])) for r in rows),
                               "hits1": sum(x == 1 for x in positions) / max(n, 1),
                               "hits10": sum(x <= 10 for x in positions) / max(n, 1),
                               "hits20": sum(x <= 20 for x in positions) / max(n, 1),
                               "gap_pp": 100 * sum(1 < x <= 10 for x in positions) / max(n, 1)}
        report["datasets"][dataset] = methods
    dump(root / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("prepare", "encode", "retrieve", "evaluate"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--fields", nargs="+", choices=TEXT_FIELDS,
                        help="Encode only these text views; default is all three.")
    args = parser.parse_args()
    if args.stage == "prepare": prepare(args)
    elif args.stage == "encode": encode(args)
    elif args.stage == "retrieve": retrieve(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
