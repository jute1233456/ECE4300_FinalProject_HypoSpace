"""Build QLoRA data from newly sampled hypotheses, excluding the scored banks."""
import json
import random
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "HypoSpace" / "_HypoSpace-main"
sys.path.insert(0, str(ROOT))

from agents.search import memory_prompt
from experiments.direct_evaluation import boolean_eval_suffix, load_domain

DATA = ROOT / "finetune" / "data"
SEED = 20261126
MECH = {
    "apply_commutativity": True,
    "apply_idempotence_and_or": True,
    "flatten_associativity": True,
}


def clear_upstream_modules():
    blocked = {
        "modules", "run_causal_benchmark", "run_3d_benchmark", "boolean_benchmark",
        "boolean_dataset", "generate_causal_dataset", "generate_3d_dataset_complete",
    }
    for key in list(sys.modules):
        if key in blocked or key.startswith("modules."):
            del sys.modules[key]


def load_fresh(domain, path):
    clear_upstream_modules()
    sys.path.insert(0, str(UPSTREAM / domain))
    return load_domain(domain, path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def eval_fingerprints():
    causal, boolean, voxel = set(), set(), set()
    for directory in (ROOT / "datasets" / "balanced", ROOT / "datasets" / "hard"):
        for path in directory.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            if "datasets_by_n_observations" in data:
                for group in data["datasets_by_n_observations"].values():
                    rows.extend(group)
            rows.extend(data.get("datasets") or [])
            rows.extend(data.get("observation_sets") or [])
            for row in rows:
                if "ground_truth_graphs" in row:
                    causal.add(tuple(sorted(item["string"] for item in row["observations"])))
                elif "ground_truth_expressions" in row:
                    boolean.add(tuple(sorted(item["string"] for item in row["observations"])))
                elif "ground_truth_structures" in row:
                    voxel.add(row["observation"])
    print(f"Held-out fingerprints: causal {len(causal)}, boolean {len(boolean)}, 3d {len(voxel)}", flush=True)
    return causal, boolean, voxel


def graph_answer(edges):
    if not edges:
        return "Graph: No edges"
    parts = [f"{src}->{dst}" for src, dst in sorted((src, dst) for src, dst in edges)]
    return "Graph: " + ", ".join(parts)


def expression_answer(formula):
    return "Expression: " + " ".join(formula.split())


def structure_answer(layers, grid):
    lines = ["Structure:"]
    for index, layer in enumerate(layers, 1):
        lines.append(f"Layer {index}:")
        for row in range(grid):
            lines.append(" ".join(layer[row * grid:(row + 1) * grid]))
    return "\n".join(lines)


def effects_of(edges, nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    descendants = {node: nx.descendants(graph, node) for node in nodes}
    observations = []
    for node in nodes:
        effects = {other: (1 if other in descendants[node] else 0) for other in nodes}
        effects[node] = 0
        text = " ".join(f"{name}:{effects[name]}" for name in nodes)
        observations.append({
            "perturbed_node": node,
            "effects": effects,
            "string": f"Perturb({node}) -> {text}",
        })
    return observations


def sample_dag(rng, nodes, max_edges):
    order = rng.sample(nodes, len(nodes))
    possible = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    count = rng.randint(0, min(max_edges, len(possible)))
    edges = rng.sample(possible, count) if count else []
    return edges


def group_graphs(graphs, nodes, limit_per_group):
    groups = defaultdict(list)
    for edges in graphs:
        observations = effects_of(edges, nodes)
        key = tuple(item["string"] for item in observations)
        bucket = groups[key]
        if len(bucket) < limit_per_group:
            bucket.append((observations, edges))
    return groups


def causal_cases(rng, blocked):
    sys.path.insert(0, str(UPSTREAM / "causal"))
    from generate_causal_dataset import CausalDatasetGenerator

    cases = []
    for width, max_edges, cap in ((4, 6, 80), (5, 8, 160)):
        nodes = [chr(65 + index) for index in range(width)]
        print(f"Enumerating {width}-node DAGs, max_edges={max_edges}", flush=True)
        dags = CausalDatasetGenerator.generate_all_dags(nodes, max_edges)
        groups = group_graphs((dag.edges for dag in dags), nodes, 24)
        chosen = [key for key in groups if key not in blocked and len(groups[key]) >= 2]
        rng.shuffle(chosen)
        for index, key in enumerate(chosen[:cap]):
            observations = groups[key][0][0]
            edges = [item[1] for item in groups[key]]
            cases.append({
                "nodes": nodes,
                "max_edges": max_edges,
                "case_id": f"c{width}_{index:04d}",
                "observations": observations,
                "graphs": edges,
            })
        print(f"  kept {min(cap, len(chosen))} of {len(chosen)} groups", flush=True)

    nodes = list("ABCDEF")
    print("Sampling 6-node DAGs", flush=True)
    seen = set()
    sampled = []
    while len(sampled) < 60000:
        edges = tuple(sorted(sample_dag(rng, nodes, 10)))
        if edges in seen:
            continue
        seen.add(edges)
        sampled.append(edges)
    groups = group_graphs(sampled, nodes, 24)
    chosen = [key for key in groups if key not in blocked and len(groups[key]) >= 2]
    rng.shuffle(chosen)
    for index, key in enumerate(chosen[:500]):
        observations = groups[key][0][0]
        cases.append({
            "nodes": nodes,
            "max_edges": 10,
            "case_id": f"c6_{index:04d}",
            "observations": observations,
            "graphs": [item[1] for item in groups[key]],
        })
    print(f"  kept {min(500, len(chosen))} of {len(chosen)} six-node groups", flush=True)
    return cases


def boolean_cases(rng, blocked):
    sys.path.insert(0, str(UPSTREAM / "boolean"))
    from boolean_dataset import BooleanDiscoveryGame, BooleanObservation

    variables = ["x", "y", "z"]
    operators = BooleanDiscoveryGame.OPERATOR_SETS["full"]
    print("Enumerating 3-variable Boolean expressions", flush=True)
    expressions = BooleanDiscoveryGame.generate_all_expressions(variables, operators, 2, MECH)
    print(f"  expressions {len(expressions)}", flush=True)
    inputs = list(product((0, 1), repeat=len(variables)))
    cases = []
    seen = set()
    attempts = 0
    while len(cases) < 500 and attempts < 4000:
        attempts += 1
        anchor = rng.choice(expressions)
        count = rng.choice((1, 2, 3, 4))
        chosen = rng.sample(inputs, count)
        observations = []
        for values in chosen:
            assignment = dict(zip(variables, values))
            observations.append(BooleanObservation(assignment, anchor.evaluate(assignment)))
        key = tuple(sorted(item.to_string() for item in observations))
        if key in seen or key in blocked:
            continue
        compatible = [item for item in expressions if all(
            item.evaluate(obs.inputs) == obs.output for obs in observations)]
        if len(compatible) < 2:
            continue
        seen.add(key)
        if len(compatible) > 24:
            compatible = rng.sample(compatible, 24)
        cases.append({
            "case_id": f"b3_{len(cases):04d}",
            "observations": [{
                "inputs": obs.inputs, "output": obs.output, "string": obs.to_string(),
            } for obs in observations],
            "expressions": [{
                "formula": item.formula,
                "mechanistic_key": str(item.mechanistic_key(**MECH)),
            } for item in compatible],
        })
        if len(cases) % 50 == 0:
            print(f"  boolean cases {len(cases)}", flush=True)
    print(f"  kept {len(cases)} boolean cases", flush=True)
    return variables, sorted(operators), cases


def voxel_cases():
    sys.path.insert(0, str(UPSTREAM / "3d"))
    from generate_3d_dataset_complete import Dataset3DGenerator

    cases = []
    for blocks in (1, 5, 6, 7):
        print(f"Enumerating 3D structures with {blocks} occupied columns", flush=True)
        generator = Dataset3DGenerator(grid_size=3, max_blocks=blocks, max_height=3)
        generator.generate_all_top_views(fixed_blocks=True)
        generator.generate_all_structures()
        generator.generate_observation_sets()
        cases.extend(generator.observation_sets)
        print(f"  observation sets {len(generator.observation_sets)}", flush=True)
    return cases


def verified_answers(domain, path, answer_of, per_case):
    benchmark, prepare, score = load_fresh(domain, path)
    kept = []
    dropped = 0
    for case in benchmark.all_observation_sets:
        observations, _ = prepare(case)
        case_id = case.get("observation_set_id") or case.get("observation_id")
        prompt = benchmark.create_prompt(observations, [])
        if domain == "boolean":
            prompt += boolean_eval_suffix(benchmark)
        answers = []
        for raw in answer_of(case):
            text = raw.strip()
            key, valid = score(text, observations)
            if valid and key is not None and text not in answers:
                answers.append(text)
            else:
                dropped += 1
            if len(answers) >= per_case:
                break
        if answers:
            kept.append((case_id, prompt, answers))
    print(f"{domain}: {len(kept)} cases, dropped {dropped} unverified targets", flush=True)
    return kept


def add_rows(rows_plain, rows_mix, domain, cases, rng):
    plain_added = mix_added = 0
    for case_id, prompt, answers in cases:
        take = answers if len(answers) <= 8 else rng.sample(answers, 8)
        for answer in take:
            row = row_of(domain, case_id, prompt, answer)
            rows_plain.append(row)
            plain_added += 1
            if rng.random() < 0.35:
                rows_mix.append(row)
                mix_added += 1
        if len(answers) < 2:
            continue
        for _ in range(min(8, len(answers))):
            target = rng.choice(answers)
            pool = [answer for answer in answers if answer != target]
            priors = rng.sample(pool, rng.randint(1, min(8, len(pool))))
            history = {str(index): text for index, text in enumerate(priors)}
            rows_mix.append(row_of(domain, case_id, memory_prompt(prompt, history), target))
            mix_added += 1
    print(f"{domain}: plain {plain_added}, mix {mix_added}", flush=True)


def row_of(domain, case_id, prompt, answer):
    return {
        "domain": domain,
        "case_id": f"{domain}:{case_id}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def split_holdout(rows, rng):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["case_id"].split(":", 1)[0]].append(row["case_id"])
    held = set()
    for ids in by_domain.values():
        unique = sorted(set(ids))
        rng.shuffle(unique)
        holdout = max(1, round(len(unique) * 0.05))
        held.update(unique[:holdout])
    train = [row for row in rows if row["case_id"] not in held]
    valid = [row for row in rows if row["case_id"] in held]
    return train, valid


def dump(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}", flush=True)


def main():
    rng = random.Random(SEED)
    DATA.mkdir(parents=True, exist_ok=True)
    blocked_causal, blocked_boolean, blocked_voxel = eval_fingerprints()

    causal = causal_cases(rng, blocked_causal)
    causal_ready = []
    for width in (4, 5, 6):
        subset = [case for case in causal if len(case["nodes"]) == width]
        if not subset:
            continue
        path = DATA / f"causal_{width}.json"
        write_json(path, {
            "metadata": {"nodes": subset[0]["nodes"], "max_edges": subset[0]["max_edges"]},
            "datasets": [{
                "observation_set_id": case["case_id"],
                "n_observations": len(case["observations"]),
                "observations": case["observations"],
                "ground_truth_graphs": [{"nodes": case["nodes"], "edges": edges} for edges in case["graphs"]],
            } for case in subset],
        })
        causal_ready.extend(verified_answers(
            "causal", path, lambda case: [graph_answer(graph["edges"]) for graph in case["ground_truth_graphs"]], 16))

    variables, operators, boolean = boolean_cases(rng, blocked_boolean)
    boolean_path = DATA / "boolean_pool.json"
    write_json(boolean_path, {
        "metadata": {
            "variables": variables,
            "operators": operators,
            "max_depth": 2,
            "mechanistic_opts": MECH,
        },
        "datasets_by_n_observations": {"mixed": [{
            "observation_set_id": case["case_id"],
            "observations": case["observations"],
            "ground_truth_expressions": case["expressions"],
        } for case in boolean]},
    })
    boolean_ready = verified_answers(
        "boolean", boolean_path,
        lambda case: [expression_answer(item["formula"]) for item in case["ground_truth_expressions"]], 16)

    voxel = [case for case in voxel_cases() if case["observation"] not in blocked_voxel]
    voxel_path = DATA / "voxel_pool.json"
    write_json(voxel_path, {
        "metadata": {"grid_size": 3, "max_blocks": 7, "max_height": 3},
        "observation_sets": voxel,
    })
    voxel_ready = verified_answers(
        "3d", voxel_path,
        lambda case: [structure_answer(structure["layers"], 3) for structure in case["ground_truth_structures"]], 16)

    plain, mix = [], []
    add_rows(plain, mix, "causal", causal_ready, rng)
    add_rows(plain, mix, "boolean", boolean_ready, rng)
    add_rows(plain, mix, "3d", voxel_ready, rng)
    rng.shuffle(plain)
    rng.shuffle(mix)
    plain_train, plain_valid = split_holdout(plain, rng)
    mix_train, mix_valid = split_holdout(mix, rng)
    dump(DATA / "plain.jsonl", plain_train)
    dump(DATA / "plain_valid.jsonl", plain_valid)
    dump(DATA / "mix.jsonl", mix_train)
    dump(DATA / "mix_valid.jsonl", mix_valid)


if __name__ == "__main__":
    main()
