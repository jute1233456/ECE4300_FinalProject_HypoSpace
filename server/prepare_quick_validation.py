"""Freeze six median-size cases, excluding all 19 teacher families, including pending ones."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from experiments.generate_medium_training import rows, family


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def family_key(row, domain, meta):
    signature = family(row, domain, meta)
    # Conservatively exclude matching observations even across Boolean operator sets.
    if domain == "boolean":
        return (domain, len(meta["variables"]), meta["max_depth"], signature[-1])
    return signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen validation manifest")
    teacher = read(root / "datasets/sampled.json")
    jobs, exclusions, hashes = [], [], {}
    for domain in ("causal", "3d", "boolean"):
        forbidden = set()
        for dataset in (("boolean", "boolean_extended") if domain == "boolean" else (domain,)):
            path = root / f"datasets/balanced/{dataset}.json"
            data = read(path)
            hashes[path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
            desired = {j["case_id"] for j in teacher[domain]
                       if domain != "boolean" or j["case_id"].startswith("extended_") == (dataset == "boolean_extended")}
            found = set()
            for row in rows(data, domain):
                cid = row.get("observation_set_id", row.get("observation_id"))
                if cid in desired:
                    found.add(cid)
                    forbidden.add(family_key(row, domain, data["metadata"]))
            assert found == desired, (dataset, desired - found)
        path = root / f"datasets/medium_sampled/{domain}.json"
        data = read(path)
        hashes[path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
        eligible = []
        for row in rows(data, domain):
            cid = row.get("observation_set_id", row.get("observation_id"))
            signature = family_key(row, domain, data["metadata"])
            refkey = {"causal": "ground_truth_graphs", "3d": "ground_truth_structures",
                      "boolean": "ground_truth_expressions"}[domain]
            count = len(row[refkey])
            if signature in forbidden:
                exclusions.append(dict(dataset=domain, case_id=cid, reason="teacher_equivalent_family"))
            else:
                eligible.append((cid, count, signature))
        counts = sorted(x[1] for x in eligible)
        median = counts[len(counts) // 2]
        eligible.sort(key=lambda x: (abs(x[1] - median), x[0]))
        seen = set()
        selected = []
        for cid, count, signature in eligible:
            if signature in seen:
                continue
            selected.append(dict(dataset=domain, case_id=cid, references=count,
                                 family_sha256=sha256(repr(signature).encode()).hexdigest()))
            seen.add(signature)
            if len(selected) == 2:
                break
        assert len(selected) == 2, domain
        jobs.extend(selected)
    assert len(jobs) == 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version="quick6-v1", source_pool="medium_sampled_27", jobs=jobs,
                    total_references=sum(j["references"] for j in jobs), exclusions=exclusions,
                    selection="nearest reference-count median, unique families, tie by case ID",
                    excluded_teacher="all sampled19, completed and pending", source_sha256=hashes)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
