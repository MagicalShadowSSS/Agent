#!/usr/bin/env python3
"""Bridge frozen relation-cosine rankings into the MAG-EA candidate interface.

The bridge has no access to ent_ILLs: it only canonicalizes URI rankings against
the entity catalog and writes the first ten of each saved Top20 list.  Labels
remain in the downstream Agent data and are read only by the evaluation code.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path


ROOT = Path(os.environ.get('SGMEA_BRIDGE_ROOT', str(Path(__file__).resolve().parents[1] / 'runs' / 'agent_bridge')))
PROJECT = Path(os.environ.get('SGMEA_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
RANK_ROOT = Path(os.environ.get('SGMEA_RANK_ROOT', str(PROJECT / 'outputs' / 'candidate' / 'rankings' / 'all')))
INTEGRATION = ROOT / "agent_data"
MMKG = Path(os.environ.get('SGMEA_MMKGB_ROOT', str(PROJECT / 'data' / 'mmkg')))


def read_map(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split("\t", 1)
        if len(p) >= 2:
            out[p[0].strip()] = p[1].strip()
    return out


def norm(value: str) -> str:
    return str(value).strip().strip("<>").strip()


def canonical_lookup(id2uri: dict[str, str]) -> dict[str, str]:
    out = {}
    for uri in id2uri.values():
        out[norm(uri)] = uri
        out[str(uri).strip()] = uri
    return out


def prepare_fbyg(out: Path) -> dict:
    raw = MMKG / "FBYG15K" / "norm"
    out.mkdir(parents=True, exist_ok=True)
    ent1, ent2 = read_map(raw / "ent_ids_1"), read_map(raw / "ent_ids_2")
    all_map = {**ent1, **ent2}
    uri_to_id = {}
    for eid, uri in all_map.items():
        uri_to_id[eid] = eid
        uri_to_id[norm(uri)] = eid
        uri_to_id[f"<{norm(uri)}>"] = eid

    def parse_attr(line: str):
        line = line.strip()
        if not line:
            return None
        p = line.split("\t", 2) if "\t" in line else line.split(" ", 2)
        if len(p) < 3:
            return None
        value = p[2].rsplit(" .", 1)[0].strip()
        value = value.split("^^", 1)[0].strip().strip('"').strip("<>").replace("_", " ")
        return p[0].strip(), p[1].strip(), value

    def attrs(paths, allowed, filename):
        seen = set(); count = 0
        with (out / filename).open("w", encoding="utf-8") as dst:
            for path in paths:
                with path.open("r", encoding="utf-8", errors="ignore") as src:
                    for line in src:
                        row = parse_attr(line)
                        if row is None:
                            continue
                        ent, pred, value = row
                        eid = uri_to_id.get(ent) or uri_to_id.get(norm(ent))
                        if eid not in allowed:
                            continue
                        key = (all_map[eid], pred, value)
                        if key in seen:
                            continue
                        seen.add(key)
                        dst.write("\t".join(key) + "\n")
                        count += 1
        return count

    relation_map = json.loads((PROJECT / "processed_names" / "FBYG15K_id2relation.json").read_text(encoding="utf-8"))
    def triples(path, filename):
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as src, (out / filename).open("w", encoding="utf-8") as dst:
            for line in src:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                s, r, o = p[:3]
                if s not in all_map or o not in all_map:
                    continue
                dst.write(f"{all_map[s]}\t{relation_map.get(r, 'r_' + r)}\t{all_map[o]}\n")
                count += 1
        return count

    def ill():
        with (out / "ent_ILLs").open("w", encoding="utf-8") as dst, (raw / "ill_ent_ids").open(encoding="utf-8") as src:
            n = 0
            for line in src:
                p = line.split()
                if len(p) >= 2 and p[0] in ent1 and p[1] in ent2:
                    dst.write(f"{ent1[p[0]]}\t{ent2[p[1]]}\n"); n += 1
            return n

    source_attrs = attrs([raw / "att_triples1"], set(ent1), "fb_att_triples")
    target_attrs = attrs([raw / "att_triples2"], set(ent2), "yg_att_triples")
    source_rels = triples(raw / "triples_1", "fb_rel_triples")
    target_rels = triples(raw / "triples_2", "yg_rel_triples")
    pairs = ill()
    names = json.loads((PROJECT / "processed_names" / "FBYG15K_norm_names.json").read_text(encoding="utf-8"))
    (out / "id2uri.json").write_text(json.dumps(all_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "id2name.json").write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "uri2id.json").write_text(json.dumps({uri: eid for eid, uri in all_map.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"source_entities": len(ent1), "target_entities": len(ent2), "pairs": pairs,
            "source_attrs": source_attrs, "target_attrs": target_attrs,
            "source_rels": source_rels, "target_rels": target_rels}


def bridge_candidates(dataset: str, data_dir: Path, out_path: Path) -> dict:
    id2uri = json.loads((data_dir / "id2uri.json").read_text(encoding="utf-8"))
    lookup = canonical_lookup(id2uri)
    rank_path = RANK_ROOT / f"{dataset}.jsonl"
    output = {}; rows = 0; dropped = 0
    with rank_path.open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line); source = lookup.get(norm(row.get("source", "")))
            if source is None:
                dropped += 1; continue
            ranks, scores = row.get("rank", []), row.get("scores", [])
            cand, sim = [], []
            for target, score in zip(ranks[:10], scores[:10]):
                target_uri = lookup.get(norm(target))
                try: score = float(score)
                except (TypeError, ValueError): continue
                if target_uri is None or not math.isfinite(score) or target_uri in cand:
                    continue
                cand.append(target_uri); sim.append(score)
            if cand:
                output[source] = {"candidates": cand, "distance": [-x for x in sim],
                                  "similarity": sim, "rank": list(range(1, len(cand) + 1))}
                rows += 1
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rank_rows": rows, "dropped_sources": dropped, "candidate_file": str(out_path),
            "min_k": min((len(x["candidates"]) for x in output.values()), default=0),
            "max_k": max((len(x["candidates"]) for x in output.values()), default=0)}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def main():
    if INTEGRATION.exists():
        # Preserve prior bridge output for resumability; generated FBYG files are
        # deterministic and are only rebuilt if the directory is absent.
        pass
    INTEGRATION.mkdir(parents=True, exist_ok=True)
    for ds in ("zh_en", "ja_en", "fr_en"):
        src = PROJECT / "dataset" / "DBP15K" / ds
        dst = INTEGRATION / "DBP15K" / ds
        link_or_copy(src, dst)
        # dataset_registry keeps the historical DBP layout when data_root is
        # supplied, while FB families use root/<dataset>.  Mirror the DBP
        # split at the root so one data_root can serve all five datasets.
        link_or_copy(src, INTEGRATION / ds)
    link_or_copy(PROJECT / "dataset" / "FBDB15K", INTEGRATION / "FBDB15K")
    fbyg = INTEGRATION / "FBYG15K"
    prep = prepare_fbyg(fbyg) if not (fbyg / "yg_rel_triples").exists() else {"reused": True}
    reports = {"FBYG15K_prepare": prep}
    for ds in ("zh_en", "ja_en", "fr_en", "FBDB15K", "FBYG15K"):
        data_dir = INTEGRATION / ("DBP15K" / Path(ds) if ds in {"zh_en", "ja_en", "fr_en"} else Path(ds))
        candidate_name = f"candidate_DBP15K_{ds}.json" if ds in {"zh_en", "ja_en", "fr_en"} else f"candidate_{ds}.json"
        reports[ds] = bridge_candidates(ds, data_dir, data_dir / candidate_name)
    (ROOT / "agent_bridge_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
