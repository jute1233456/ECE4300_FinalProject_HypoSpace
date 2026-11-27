"""Shared HypoSpace evaluation for the quick test and full direct baseline."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
import sys
from itertools import count
from time import sleep

from .boolean_scoring import expand_nor
from .rate_limit import record_request

from agents import DirectAgent
from agents.reasoning_tree import ReasoningTreeAgent
from agents.search import MCTSAgent, MemoryAgent, StructuredMemoryAgent
from agents.coverage_search import CoverageMCTSAgent
from llm import ArkCodingPlanClient, LLMError, OpenAICompatibleClient
from llm.ark_coding_plan import ANSWER_MODEL

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "HypoSpace" / "_HypoSpace-main"


def load_domain(domain, path):
    for name in ("causal", "3d", "boolean"):
        directory = str(UPSTREAM / name)
        if directory not in sys.path:
            sys.path.append(directory)

    if domain == "causal":
        from run_causal_benchmark import CausalBenchmarkEnhanced, CausalGraph

        benchmark = CausalBenchmarkEnhanced(str(path))

        def prepare(case):
            observations = case["observations"]
            keys = {CausalGraph.from_dict(graph).get_hash()
                    for graph in case["ground_truth_graphs"]}
            return observations, keys

        def score(text, observations):
            graph = benchmark.parse_llm_response(text)
            if graph is None:
                return None, False
            return graph.get_hash(), benchmark.validate_hypothesis(graph, observations)

    elif domain == "3d":
        from run_3d_benchmark import Benchmark3D, Structure3D

        benchmark = Benchmark3D(str(path))

        def prepare(case):
            keys = set()
            size = benchmark.metadata["grid_size"]
            for structure in case["ground_truth_structures"]:
                layers = [[list(map(int, layer[start:start + size]))
                           for start in range(0, len(layer), size)]
                          for layer in structure["layers"]]
                keys.add(Structure3D(layers).get_hash())
            return case["observation"], keys

        def score(text, observations):
            structure = benchmark.parse_llm_response(text)
            if structure is None:
                return None, False
            return (structure.get_hash(),
                    len(structure.layers) <= benchmark.metadata["max_height"]
                    and benchmark.validate_structure_matches_observations(structure, observations))

    else:
        from boolean_benchmark import BooleanBenchmarkRefined, BooleanObservation
        from boolean_dataset import BooleanDiscoveryGame

        benchmark = BooleanBenchmarkRefined(str(path))
        # Use the generator's hypothesis space, including its primitive NOR depth.
        expressions = {
            expression.mechanistic_key(**benchmark.mechanistic_opts): expression
            for expression in BooleanDiscoveryGame.generate_all_expressions(
                benchmark.variables, benchmark.operators, benchmark.max_depth,
                benchmark.mechanistic_opts,
            )
        }

        def prepare(case):
            observations = [BooleanObservation(obs["inputs"], obs["output"])
                            for obs in case["observations"]]
            keys = {ast.literal_eval(expr["mechanistic_key"])
                    for expr in case["ground_truth_expressions"]}
            return observations, keys

        tokens = "|".join(re.escape(token)
                          for token in sorted(set(benchmark.variables) | benchmark.operators))
        allowed_expression = re.compile(rf"(?:\s+|[(),]|\b(?:{tokens})\b)+")

        def score(text, observations):
            expression = benchmark.parse_llm_response(text)
            if expression is None or not allowed_expression.fullmatch(expression):
                return None, False
            expression = expand_nor(expression)
            if expression is None:
                return None, False
            key = benchmark.get_expression_mechanistic_key(expression)
            if key is None:
                return None, False
            representative = expressions.get(key)
            valid = representative is not None and all(
                representative.evaluate(observation.inputs) == observation.output
                for observation in observations
            )
            return key, valid

    return benchmark, prepare, score


def summarize(cases):
    summary = {}
    for group in ("all", "causal", "3d", "boolean"):
        rows = [case for case in cases if group == "all" or case["domain"] == group]
        if rows:
            # Cases skipped on repeated request errors keep partial results; they are
            # excluded from validity/uniqueness/recovery and rerun separately later.
            normal = [row for row in rows if row.get("stop_reason") != "request_errors"]
            summary[group] = {
                "cases": len(rows),
                "normal_cases": len(normal),
                "requests": sum(len(row["records"]) for row in rows),
                "submissions": sum(row.get("submitted", len(row["records"])) for row in rows),
                "total_tokens": sum(
                    (record.get("total_tokens") or 0)
                    + (record.get("reflection_response", {}).get("total_tokens") or 0)
                    for row in rows for record in row["records"]
                ),
                "request_errors": sum("error" in record for row in rows for record in row["records"]),
                **{metric: mean(row["metrics"][metric] for row in (normal or rows))
                   for metric in ("validity", "uniqueness", "recovery")},
            }
    return summary


def wait_if_paused(pause_file):
    if not pause_file.exists():
        return
    print(f"Paused. Remove {pause_file} to continue.", flush=True)
    while pause_file.exists():
        sleep(1)
    print("Resumed.", flush=True)


def main(*, quick_cases=None, medium=False, hard=False, branch_examples=False, balanced=False, easy=False):
    quick = quick_cases is not None
    early_stop = medium or hard or easy
    dataset_root = ROOT / "datasets"
    if hard:
        dataset_root /= "balanced" if balanced else "hard"
    medium_cases = json.loads((ROOT / "datasets/medium_cases.json").read_text()) if medium else None
    parser = argparse.ArgumentParser(description="Quick test" if quick else "Full HypoSpace baseline")
    parser.add_argument("--dataset", choices=("causal", "3d", "boolean", "boolean_extended"))
    parser.add_argument("--dataset-root", type=Path, help="Override the dataset directory (default: datasets or datasets/balanced)")
    parser.add_argument("--case-id", help="Evaluate one observation set")
    parser.add_argument("--output", type=Path, help="Independent result path")
    parser.add_argument("--seed-case", type=Path, help="Import this case's history from an earlier run")
    parser.add_argument("--queries", type=int, default=None,
                        help="Fixed calls per case. Default: one call per distinct reference hypothesis.")
    parser.add_argument("--method", choices=("direct", "memory", "memory-structured", "mcts", "mcts-breadth", "mcts-recent", "mcts-llm", "mcts-tree"), default="direct")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, help="Resume an interrupted run with the same model")
    if balanced:
        continuation.add_argument("--continue-from", type=Path,
                                  help="Fork an old-model checkpoint and continue with the fixed answer model")
    parser.add_argument("--search-cap", type=int, help="Total API attempts for a single case")
    parser.add_argument("--unlimited-search", action="store_true",
                        help="Continue tree search without a fixed attempt or submission budget")
    token_limit = parser.add_mutually_exclusive_group()
    token_limit.add_argument("--max-tokens", type=int, default=2048)
    token_limit.add_argument("--provider-token-limit", dest="max_tokens", action="store_const", const=None)
    parser.add_argument("--tree-policy", choices=("uct", "breadth", "recent", "llm"), default="uct")
    parser.add_argument("--full-tree", action="store_true", help="Expose and save the complete LLM search tree")
    parser.add_argument('--tree-tools', action='store_true', help='Read the tree through Qwen-Agent tools before searching')
    parser.add_argument('--mcts-version', choices=('1.0','2.0','2.1','2.2'), help='Public method version, independent of model version')
    reflection = parser.add_mutually_exclusive_group()
    reflection.add_argument("--self-reflection", action="store_true")
    reflection.add_argument("--two-step-reflection", action="store_true")
    parser.add_argument("--partition-reflection", action="store_true")
    parser.add_argument("--stage22-revision", action="store_true")
    parser.add_argument("--provider", choices=("ark", "deepseek"), default="ark")
    parser.add_argument("--exploration-list", choices=("on", "off"),
                        help="Compare LLM node selection with or without round-based deferral")
    tree_limit = parser.add_mutually_exclusive_group()
    tree_limit.add_argument("--child-limit", type=int, choices=(5, 7), default=5)
    tree_limit.add_argument("--unbounded-tree", action="store_true",
                            help="Allow reasoning paths and sibling counts to grow within the search budget")
    parser.add_argument("--duplicate-reward", type=int, choices=(0, -1), default=0)
    parser.add_argument("--model", default=ANSWER_MODEL)
    parser.add_argument('--reflection-model', help='Separate local model for reflection only; search keeps --model')
    parser.add_argument("--base-url", help="Optional local OpenAI-compatible API; HTTP restricted to loopback")
    parser.add_argument("--enable-thinking", action="store_true", help="Local Qwen thinking mode; score final answer only")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pause-file", type=Path, default=ROOT / "results" / "PAUSE",
                        help="Pause between requests while this file exists")
    parser.add_argument("--plan", action="store_true", help="Show workload without model calls")
    if balanced:
        parser.add_argument("--import-progress", type=Path, help="Prepare a checkpoint from the old hard run")
    args = parser.parse_args()
    if args.provider == "deepseek":
        from llm.deepseek_official import DeepSeekOfficialClient
        args.model = DeepSeekOfficialClient.MODEL
    if args.dataset_root:
        dataset_root = args.dataset_root
    if args.provider == "ark" and not args.base_url and args.model != ANSWER_MODEL:
        parser.error("Alternative models require --base-url; Ark default is unchanged")
    if args.enable_thinking and not args.base_url:
        parser.error("--enable-thinking requires --base-url")
    import_progress = args.import_progress if balanced else None
    continue_from = args.continue_from if balanced else None
    if import_progress and (args.resume or continue_from):
        parser.error("Choose either --import-progress, --resume or --continue-from")
    if args.queries is not None and args.queries < 1:
        parser.error("--queries must be positive")

    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.exploration_list and (args.method != "mcts-tree" or args.tree_policy != "llm"):
        parser.error("--exploration-list requires --method mcts-tree --tree-policy llm")
    if args.unbounded_tree and (args.method != "mcts-tree" or args.tree_policy != "llm"):
        parser.error("--unbounded-tree requires --method mcts-tree --tree-policy llm")
    if (args.self_reflection or args.two_step_reflection) and not (branch_examples and args.method == "mcts-tree" and args.tree_policy == "llm"):
        parser.error("Reflection requires the Boolean branch-example test with LLM node selection")
    tree = args.method.startswith("mcts")
    if args.mcts_version and (args.method != 'mcts-tree' or args.tree_policy != 'llm'):
        parser.error('--mcts-version requires --method mcts-tree --tree-policy llm')
    if args.mcts_version == '1.0' and args.tree_tools:
        parser.error('MCTS 1.0 does not use tree tools')
    if args.mcts_version in ('2.0','2.1','2.2'):
        args.tree_tools = True
    if args.mcts_version == '2.2' and not args.two_step_reflection:
        parser.error('Stage-MCTS 2.2 requires two-step reflection')
    if args.method == 'mcts-tree' and args.tree_policy == 'llm' and not args.mcts_version:
        args.mcts_version = '2.0' if args.tree_tools else '1.0'
    if args.full_tree and (args.method != 'mcts-tree' or args.tree_policy != 'llm'):
        parser.error('--full-tree requires --method mcts-tree --tree-policy llm')
    if args.tree_tools and (args.method != 'mcts-tree' or args.tree_policy != 'llm' or args.full_tree):
        parser.error('--tree-tools requires --method mcts-tree --tree-policy llm without --full-tree')
    jobs = []
    datasets = {}
    names = ("causal", "3d", "boolean", "boolean_extended") if balanced else ("causal", "3d", "boolean")
    if args.dataset:
        names = (args.dataset,)
    for name in quick_cases if quick else names:
        domain = "boolean" if name == "boolean_extended" else name
        path = dataset_root / f"{name}.json"
        benchmark, prepare, score = load_domain(domain, path)
        cases = benchmark.all_observation_sets
        if not cases:
            raise ValueError(f"Empty dataset: {domain}")
        datasets[name] = {"sha256": sha256(path.read_bytes()).hexdigest(), "cases": len(cases)}
        if quick:
            selected_id = quick_cases[domain]
            cases = [case for case in cases
                     if (case.get("observation_set_id") or case["observation_id"]) == selected_id]
            if len(cases) != 1:
                raise ValueError(f"Expected exactly one case for {domain}/{selected_id}")
        if medium:
            by_id = {(case.get("observation_set_id") or case["observation_id"]): case for case in cases}
            cases = [by_id[case_id] for case_id in medium_cases[domain]]
        for case in cases:
            selected_id = case.get("observation_set_id") or case.get("observation_id")
            if args.case_id and selected_id != args.case_id:
                continue
            observations, keys = prepare(case)
            if not keys:
                raise ValueError(f"Case without references: {domain}")
            case_id = case.get("observation_set_id") or case["observation_id"]
            prompt = benchmark.create_prompt(observations, [])
            if domain == "boolean" and (balanced or not args.resume):
                prompt += (
                    "\n\nCandidate formatting: Put the complete expression on a single line "
                    "beginning with Expression:. Use the listed variable names, uppercase operators, "
                    "and balanced parentheses. Keep each operation and its grouping explicit. "
                    "Keep nested operations even when a shorter formula has the same truth table. "
                    "Format-only template: Expression: (<left operand>) AND (<right operand>). "
                    "Replace placeholders with actual operands."
                )
                if "NOR" in benchmark.operators:
                    prompt += (
                        " Use NOT (...) around an OR expression to write each NOR gate. "
                        "For scoring, NOT ((A) OR (B)) denotes the same NOR gate as NOR(A, B). "
                        "Apply this notation recursively: NOR(A, NOR(B, C)) is written as "
                        "NOT ((A) OR NOT ((B) OR (C))). A, B and C are placeholders for operands, "
                        "not extra variables. Keep the original grouping and operator structure; "
                        "do not apply De Morgan's laws, absorption, or double-negation elimination."
                    )
            jobs.append((domain, case_id, prompt, observations, keys, score,
                         args.queries or (len(keys) + 5 if early_stop else len(keys))))

    if args.search_cap is not None and (args.search_cap < 1 or len(jobs) != 1):
        parser.error("--search-cap requires one case and a positive total")
    budget = sum(job[-1] for job in jobs)
    if not jobs:
        parser.error("No matching cases")
    call_cap = args.search_cap or budget * (2 if early_stop and tree else 1)
    submissions = "no separate submission cap" if args.unbounded_tree else f"up to {budget} submissions"
    print(f"{len(jobs)} cases, {submissions}, up to {call_cap} search attempts", flush=True)
    if args.two_step_reflection:
        print(f"Two-step reflection allows up to {2 * call_cap} API requests.", flush=True)
    if args.plan:
        return

    def make_agent(client):
        if args.method == "mcts-tree":
            return ReasoningTreeAgent(client, args.tree_policy, args.child_limit, args.duplicate_reward,
                                      rotate=args.exploration_list == "on", branch_examples=branch_examples,
                                      reflect=args.self_reflection, unbounded_tree=args.unbounded_tree,
                                      full_tree=args.full_tree, mcts_version=args.mcts_version)
        if args.method.startswith("mcts-"):
            return CoverageMCTSAgent(client, args.method.split("-", 1)[1])
        if args.method == "memory-structured":
            return StructuredMemoryAgent(client)
        return {"direct": DirectAgent, "memory": MemoryAgent, "mcts": MCTSAgent}[args.method](client)
    # Independent stochastic samples are essential for coverage; a fixed per-request
    # seed would make identical direct prompts return identical candidates.
    local_options = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                         chat_template_kwargs={"enable_thinking": args.enable_thinking})
    if args.tree_tools and not args.enable_thinking:
        from llm.deepseek_tree_tools import DeepseekTreeToolClient
        if args.mcts_version == '2.2':
            from llm.ark_stage_tree import ArkStageTreeClient
            client = ArkStageTreeClient(model=args.model, provider=args.provider)
            client.revision = args.stage22_revision
        else:
            client = DeepseekTreeToolClient(model=args.model, provider=args.provider)
    elif args.enable_thinking:
        from llm.local_thinking import LocalThinkingClient
        client_type = LocalThinkingClient
        if args.tree_tools:
            from llm.tree_tools import TreeToolClient
            client_type = TreeToolClient
            if args.mcts_version == '2.2':
                from llm.split_tree_tools import SplitTreeToolClient
                client_type = SplitTreeToolClient
        client = client_type(tokenizer_path=ROOT/'models/Qwen3-8B', api_key='local',
            model=args.model, base_url=args.base_url, timeout=3600, generation_options=local_options)
    else:
        if args.provider == "deepseek":
            from llm.deepseek_official import DeepSeekOfficialClient
            client = DeepSeekOfficialClient(model=args.model)
        else:
            client = (OpenAICompatibleClient(api_key="local", model=args.model, base_url=args.base_url,
                                            timeout=3600, generation_options=local_options)
                      if args.base_url else ArkCodingPlanClient(model=args.model))
    if args.reflection_model:
        if not (args.enable_thinking and args.tree_tools and args.two_step_reflection):
            parser.error('--reflection-model requires local thinking tree search with reflection')
        from llm.stage_routing import route_reflection
        route_reflection(client, args.reflection_model)
    now = datetime.now(timezone.utc)
    mode = "easy" if easy else "medium" if medium else "quick" if quick else "baseline"
    if hard:
        mode = "hard_quick" if quick else "hard_full"
    if balanced:
        mode = "hard_balanced"
    label = f"{args.method}-list-{args.exploration_list}" if args.exploration_list else args.method
    if branch_examples:
        label += "-branch-examples"
    if args.self_reflection:
        label += "-reflection"
    if args.two_step_reflection:
        label += "-two-step"
    if args.unbounded_tree:
        label += "-unbounded"
    output = ROOT / "results" / f"{label}_{mode}_{now:%Y%m%dT%H%M%S%fZ}.json"
    if args.output:
        output = args.output
        if output.exists() and not args.resume:
            parser.error("Output exists; use --resume")
    result = {"mode": mode, "agent": type(make_agent(client)).__name__, "history": args.method != "direct",
              "started_utc": now.isoformat(), "requested_model": client.model,
              "max_tokens": args.max_tokens, "queries_per_case": args.queries,
              "datasets": datasets, "expected_cases": len(jobs), "expected_requests": budget,
              "method": args.method, "call_cap": call_cap, "early_stop": early_stop,
              "completed": False, "cases": [], "summary": {}}
    if args.base_url:
        result["local_api"] = {"base_url": args.base_url, "generation_options": local_options}
        if args.enable_thinking:
            result['thinking_config']={'enabled':True,'final_answer_only':True,
                'max_output_tokens':args.max_tokens,'context_window':32768,
                'context_policy':'retain full input; reduce output cap to remaining context',
                'reasoning_tokens':'provider count when available; otherwise local tokenizer estimate'}
        manifest = ROOT / "logs/model_manifest.json"
        if manifest.exists():
            result["model_manifest"] = json.loads(manifest.read_text())

    if args.method == "mcts-tree":
        result["tree_config"] = {"policy": args.tree_policy, "duplicate_reward": args.duplicate_reward}
        if args.mcts_version:
            from agents.mcts_versions import COMPLETION_POLICY
            result['mcts_version'] = args.mcts_version
            result['mcts_variant'] = 'two-step' if args.two_step_reflection else 'standard'
            if args.mcts_version in ('2.1','2.2'):
                result['tree_config']['completion_policy'] = '2.1'
                result['completion_policy'] = COMPLETION_POLICY
                result['completion_policy_sha256'] = sha256(COMPLETION_POLICY.encode()).hexdigest()
        if args.tree_tools:
            framework = 'Qwen-Agent' if args.enable_thinking else 'native-function-calling'
            result['tree_config']['tree_tools'] = dict(version=2, framework=framework,
                max_requests_per_search=5, max_tools_per_search=4, reflection_tools=False)
        if args.mcts_version == '2.2':
            result['tree_config']['tree_tools'].update(max_requests_per_search=4, max_tools_per_search=1)
            result['tree_config']['action_decomposition'] = dict(name='Stage-MCTS',
                stages=['reflection','read_tree','selection','expansion','candidate'],
                context='full tree for selection; selected path and children for expansion/candidate; full answer history retained',
                exhaustion='model theoretical argument required; not formally verified',
                max_requests_per_search=4, reflection_requests=1)
        if args.full_tree:
            result['tree_config']['full_tree'] = True
            result['tree_config']['full_tree_version'] = 1
        if args.unbounded_tree:
            result["tree_config"]["unbounded_tree"] = True
        else:
            result["tree_config"]["child_limit"] = args.child_limit

    if args.exploration_list:
        result["tree_config"].update(exploration_list=args.exploration_list, branch_attribution=2)

    if branch_examples and args.method == "mcts-tree":
        result["tree_config"]["branch_examples"] = True

    if args.self_reflection:
        result["tree_config"]["self_reflection"] = True

    if args.two_step_reflection:
        result["tree_config"]["two_step_reflection"] = True
        if args.reflection_model:
            result['tree_config']['stage_models'] = client.stage_models
            result['tree_config']['model_strategy'] = 'hybrid-reflection-v3'
        if args.stage22_revision:
            result["tree_config"]["stage22_revision"] = True
        if args.partition_reflection:
            result["tree_config"]["partition_reflection"] = True
        result["api_call_cap"] = 2 * call_cap

    if import_progress:
        old = json.loads(import_progress.read_text(encoding="utf-8"))
        fields = ("mode", "method", "agent", "requested_model", "max_tokens", "queries_per_case",
                  "tree_config")
        if old["mode"] != "hard_full" or any(old.get(key) != result.get(key)
                                              for key in fields if key != "mode"):
            parser.error("Old run does not match the current method and settings")
        completed = {(case["domain"], case["case_id"]): case for case in old["cases"]}
        partial = old.get("interrupted_case")
        for domain, case_id, prompt, _, keys, _, queries in jobs:
            case = completed.get((domain, case_id))
            if case is None:
                if partial and (partial["domain"], partial["case_id"]) == (domain, case_id):
                    result["interrupted_case"] = partial
                break
            if (case["prompt"], case["reference_count"], case["submission_cap"]) != (prompt, len(keys), queries):
                parser.error(f"Cannot reuse changed case {case_id}")
            result["cases"].append(case)
        result["summary"] = summarize(result["cases"])
        result["parent_run"] = import_progress.name
        output = ROOT / "results" / f"{args.method}_hard_balanced_checkpoint.json"
        if output.exists():
            parser.error(f"Checkpoint already exists: {output}")

    if args.seed_case:
        if len(jobs) != 1 or args.resume or continue_from:
            parser.error("--seed-case requires one case and a fresh output")
        source = json.loads(args.seed_case.read_text(encoding="utf-8"))
        if source.get("method") != result["method"] or source.get("tree_config") != result.get("tree_config"):
            parser.error("Seed method and tree configuration must match")
        domain, case_id = jobs[0][:2]
        candidates = list(source["cases"])
        if source.get("interrupted_case"):
            candidates.append(source["interrupted_case"])
        saved = next((case for case in candidates
                      if (case["domain"], case["case_id"]) == (domain, case_id)), None)
        if saved:
            result["interrupted_case"] = {key: saved[key] for key in ("domain", "case_id", "prompt", "records")}
            result["parent_run"] = args.seed_case.name
            result["seed_requested_model"] = source["requested_model"]

    if args.resume or continue_from:
        if args.method not in ("direct", "memory", "mcts", "mcts-tree"):
            parser.error("--resume supports direct, memory, UCT and reasoning-tree runs")
        previous = json.loads((args.resume or continue_from).read_text(encoding="utf-8"))
        if args.mcts_version:
            from agents.mcts_versions import infer_version
            if infer_version(previous) != args.mcts_version:
                parser.error('Cannot resume a different MCTS method version; use a new output file')
            if args.mcts_version in ('2.1','2.2') and previous.get('completion_policy_sha256') != result.get('completion_policy_sha256'):
                parser.error('MCTS 2.1 completion policy changed; publish a new version instead')
            previous.setdefault('mcts_version', args.mcts_version)
            previous.setdefault('mcts_variant', result['mcts_variant'])
        fields = ("mode", "method", "agent", "max_tokens", "queries_per_case",
                  "datasets", "expected_cases", "expected_requests")
        if args.base_url or previous.get("local_api"):
            if previous.get("local_api", {}).get("generation_options") != result.get("local_api", {}).get("generation_options"):
                parser.error("Resume generation settings differ")
            fields += ("model_manifest",)
        if args.method == "mcts-tree":
            fields += ("tree_config",)
        if previous["completed"] or any(previous.get(key) != result.get(key) for key in fields):
            parser.error("Resume file must be incomplete and match the current run settings")
        if continue_from:
            if previous["requested_model"] == result["requested_model"]:
                parser.error("Use --resume to continue with the same model")
            if previous["call_cap"] != result["call_cap"]:
                parser.error("The source checkpoint must use the same search budget")
            result.update(cases=previous["cases"], summary=previous["summary"], model_transition={
                "source_checkpoint": continue_from.name,
                "previous_requested_model": previous["requested_model"],
                "carried_cases": len(previous["cases"]),
                "carried_search_attempts": len(previous.get("interrupted_case", {}).get("records", [])),
            })
            if "interrupted_case" in previous:
                result["interrupted_case"] = previous["interrupted_case"]
        else:
            if previous["requested_model"] != result["requested_model"]:
                parser.error("Checkpoint model differs from the fixed answer model; see AGENTS.md")
            output = args.resume
            result = previous
            if args.base_url:
                routes = result.setdefault("api_routes", [result["local_api"]["base_url"]])
                if args.base_url not in routes:
                    routes.append(args.base_url)

    active_agent = None
    active_case = None
    tree_file = output.with_suffix('.tree.json')
    tree_snapshots = (json.loads(tree_file.read_text(encoding='utf-8'))
                      if (args.full_tree or args.tree_tools) and args.resume and tree_file.exists() else {})

    def save():
        output.parent.mkdir(exist_ok=True)
        if (args.full_tree or args.tree_tools) and active_agent is not None:
            tree_snapshots[active_case] = active_agent.tree_snapshot()
            tree_temporary = tree_file.with_suffix('.tmp')
            tree_temporary.write_text(json.dumps(tree_snapshots, ensure_ascii=False, indent=2), encoding='utf-8')
            tree_temporary.replace(tree_file)
            result['tree_snapshot_file'] = tree_file.name
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    print(f"Saving results to {output.name}", flush=True)
    save()
    if import_progress:
        return
    consecutive_errors = 0
    for index, (domain, case_id, prompt, observations, keys, score, queries) in enumerate(jobs, 1):
        agent = make_agent(client)
        agent.max_tokens = args.max_tokens
        records, unique, recovered = [], set(), set()
        submitted = 0
        search_cap = args.search_cap or queries * (2 if early_stop and tree else 1)
        if index <= len(result["cases"]):
            continue
        active_agent = agent
        active_case = f'{domain}/{case_id}'
        partial = result.get("interrupted_case")
        if partial:
            normalize = lambda value: re.sub(r"(Use ONLY these operators: )([^\r\n]+)",
                                             lambda m: m[1] + ", ".join(sorted(m[2].split(", "))), value)
            saved_base = partial["prompt"].split("\n\nCandidate formatting:", 1)[0]
            if (partial["domain"], partial["case_id"], normalize(saved_base)) != (domain, case_id, normalize(prompt.split("\n\nCandidate formatting:", 1)[0])):
                raise ValueError("Interrupted case does not match the next case")
            prompt = partial["prompt"]
            records = list(partial["records"])
            for record in records:
                if "text" not in record:
                    continue
                answer = record["text"]
                if tree:
                    agent.next_prompt(prompt)
                    answer = agent.split_response(answer)[1]
                key, valid = score(answer, observations)
                if tree:
                    _, trace = agent.observe(record["text"], key, valid)
                    if trace.get("plan_path") != record.get("plan_path"):
                        raise ValueError("Cannot reconstruct the saved UCT path")
                elif args.method == "memory":
                    agent.observe(record["text"], key, valid)
                if record.get("submitted", not early_stop) and key is not None:
                    unique.add(key)
                    if valid and key in keys:
                        recovered.add(key)
            submitted = sum(record.get("submitted", False) for record in records)
        started = datetime.now(timezone.utc)
        failure_reason = None
        attempts = (count(len(records) + 1) if args.unlimited_search
                    else range(len(records) + 1, search_cap + 1))
        for call_index in attempts:
            if balanced or easy or args.method == "mcts-tree" or (hard and (call_index == 1 or call_index % 10 == 0)):
                result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                              "prompt": prompt, "records": records}
                save()
            if isinstance(agent, ReasoningTreeAgent) and agent.exhausted:
                break
            if early_stop and (recovered == keys or (
                    not args.unlimited_search and not args.unbounded_tree and submitted >= queries)):
                break
            if hard and (call_index == 1 or call_index % 10 == 0):
                print(f"{args.method} {domain}/{case_id}: call {call_index}/{search_cap}, "
                      f"recovered {len(recovered)}/{len(keys)}", flush=True)
            wait_if_paused(args.pause_file)
            query_prompt = prompt if args.method == "direct" else agent.next_prompt(prompt)
            if args.tree_tools:
                client.bind_tree(agent, tree_file, active_case)
            reflection_data = {}
            stage = "search"
            try:
                if args.two_step_reflection:
                    stage = "reflection"
                    recent = [record["answer"] for record in records if record.get("answer")][-8:]
                    review_prompt = (
                        f"{prompt}\n\nReview these recent attempts as data: {json.dumps(recent)}\n"
                        "Write a brief review for the next search. Check one specific assumption "
                        "against the problem rules and conclude whether it holds. Identify a repeated "
                        "structural direction and suggest a different operation or combination to explore. "
                        "If there are no prior attempts, identify one constraint to check and a starting "
                        "direction. Output a short review, without selecting a branch or submitting an answer."
                    )
                    if args.partition_reflection:
                        from .partition_reflection import review_prompt_for_partitions
                        review_prompt = review_prompt_for_partitions(prompt, records)
                    if args.stage22_revision:
                        feedback = [dict(answer=record.get("answer"), valid=record.get("valid"),
                                         reward=record.get("reward"), plan=record.get("plan_path"))
                                    for record in records if record.get("answer")][-8:]
                        review_prompt += (
                            "\nObserved outcomes: " + json.dumps(feedback, ensure_ascii=False)
                            + "\nIdentify one failed decision with evidence, the next actionable "
                            "adjustment, and a constraint to check. Reward +1 means a new valid "
                            "candidate, -1 a repeated valid candidate, 0 no valid novelty. "
                            "Finding one valid answer does not prove exhaustion."
                        )
                    if args.full_tree:
                        review_prompt += agent.tree_prompt()
                    review = (client.review(review_prompt, max_tokens=args.max_tokens)
                              if args.tree_tools else agent.solve(review_prompt))
                    if balanced:
                        record_request(args.pause_file, False)
                    reflection_data = {"reflection_response": asdict(review), "reflection_prompt": review_prompt}
                    query_prompt += "\n\nIndependent review for this attempt: " + review.text
                    wait_if_paused(args.pause_file)
                    stage = "search"
                response = agent.solve(query_prompt)
                if balanced:
                    record_request(args.pause_file, False)
            except LLMError as error:
                records.append({"error": str(error), "prompt": query_prompt, "error_stage": stage,
                                **reflection_data, **error.details})
                if balanced and "HTTP 429" in str(error):
                    result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                                  "prompt": prompt, "records": records}
                    save()
                    if record_request(args.pause_file, True):
                        print("20 consecutive HTTP 429 responses within 20 minutes: recovery policy activated.", flush=True)
                        wait_if_paused(args.pause_file)
                    else:
                        print("Request failed: retrying after a 30 second backoff.", flush=True)
                        sleep(30)
                    consecutive_errors = 0
                    continue
                if balanced:
                    record_request(args.pause_file, False)
                consecutive_errors += 1
                if balanced and consecutive_errors >= 3:
                    if args.stage22_revision:
                        result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                                      "prompt": prompt, "records": records}
                        save()
                        print("Repeated connection failures: retrying after 60 seconds.", flush=True)
                        sleep(60)
                        consecutive_errors = 0
                        continue
                    failure_reason = str(error)
                    print(f"Skipping {domain}/{case_id} after repeated request errors: {failure_reason}",
                          flush=True)
                    consecutive_errors = 0
                    break
                if consecutive_errors >= 3 and (not early_stop or "no usable text" not in str(error)):
                    result["interrupted_case"] = {"domain": domain, "case_id": case_id,
                                                  "prompt": prompt, "records": records}
                    save()
                    raise SystemExit(f"Three consecutive request failures. Partial result: {output.name}")
                continue
            consecutive_errors = 0
            answer = response.text
            if isinstance(agent, CoverageMCTSAgent):
                answer = agent.answer_body(answer)
            if tree:
                answer = agent.split_response(answer)[1]
            key, valid = score(answer, observations) if answer.strip() else (None, False)
            strategy = {}
            if args.method != "direct":
                _, strategy = agent.observe(response.text, key, valid)
                strategy.update(prompt=query_prompt, answer=answer)
            is_submission = not early_stop or not tree or valid
            submitted += int(is_submission)
            records.append({**asdict(response), **strategy, **reflection_data, "valid": valid,
                            "submitted": is_submission, "hypothesis_key": key,
                            "reference_hit": valid and key in keys})
            if is_submission and key is not None:
                unique.add(key)
                if valid and key in keys:
                    recovered.add(key)
        denominator = submitted if early_stop else queries
        result["cases"].append({
            "submission_cap": None if args.unbounded_tree else queries,
            "submitted": submitted, "model_call_cap": search_cap,
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            **({"failure_reason": failure_reason} if failure_reason else {}),
            "stop_reason": ("request_errors" if failure_reason else
                            "all_recovered" if recovered == keys and early_stop else
                            "submission_cap" if not args.unbounded_tree and submitted >= queries else
                            "tree_exhausted" if isinstance(agent, ReasoningTreeAgent) and agent.exhausted
                            else "model_call_cap"),
            "domain": domain, "case_id": case_id, "prompt": prompt,
            "reference_count": len(keys), "records": records,
            "metrics": {"validity": sum(record.get("valid", False) for record in records) / max(denominator, 1),
                        "uniqueness": len(unique) / max(denominator, 1), "recovery": len(recovered) / len(keys)},
        })
        result.pop("interrupted_case", None)
        result["summary"] = summarize(result["cases"])
        save()
        print(f"{index}/{len(jobs)} {domain}/{case_id}", flush=True)

    if args.two_step_reflection:
        result["api_calls"] = sum(
            (len(record.get('api_requests', [])) or 1) + int("reflection_response" in record)
            for case in result["cases"] for record in case["records"]
        )
    elif args.tree_tools:
        result['api_calls'] = sum(len(r.get('api_requests', [])) or 1
                                  for c in result['cases'] for r in c['records'])
    result["completed"] = True
    save()
    print(json.dumps(result["summary"], indent=2))
