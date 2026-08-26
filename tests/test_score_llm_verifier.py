import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import score_llm_verifier as scorer
from experiments.common import candidate_seal


def _write_criteria(tmp_path):
    criteria = tmp_path / "terminal_bench.md"
    criteria.write_text(
        "# Terminal Bench\n\n## specification\nCorrectness.\n",
        encoding="utf-8",
    )
    return criteria


def _write_official_trace(root, task, suffix, reward):
    trial = f"{task}__{suffix}"
    path = (
        root
        / "mini-swe-agent_deepseek-v4-flash"
        / task
        / f"{trial}_trajectory.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": f"id-{suffix}",
        "trial_name": trial,
        "task_name": task,
        "reward": reward,
        "trajectory": {
            "schema_version": "atif-converted-1.0",
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": f"Solve {task}",
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "I will inspect it.",
                    "tool_calls": [
                        {
                            "tool_name": "bash",
                            "arguments": {"keystrokes": "pwd"},
                        }
                    ],
                    "observation": {
                        "results": [{"content": '{"output":"/app"}'}]
                    },
                },
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_official_release(root, tasks=("task-a",), candidates=5):
    for task in tasks:
        for index in range(candidates):
            _write_official_trace(root, task, f"trial-{index}", index % 2)


def _make_frozen_candidates(root, tasks=("task-a",), candidates=5):
    specs = []
    seals = []
    for task in tasks:
        for index in range(candidates):
            trial = f"{task}__trial-{index}"
            trajectory_path = root / "traces" / task / f"{trial}.json"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": 1,
                                "source": "user",
                                "message": f"Solve {task}",
                            },
                            {
                                "step_id": 2,
                                "source": "agent",
                                "message": f"candidate {index}",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            spec = {
                "schema": "harbor-rec-codex-candidate/v1",
                "dataset": "terminal-bench/terminal-bench-2-1",
                "dataset_digest": "a" * 64,
                "task": f"terminal-bench/{task}",
                "task_digest": "c" * 64,
                "trial": trial,
                "model": "gpt-5.6-terra",
                "effort": "medium",
                "trajectory_path": str(trajectory_path),
                "world_digest": "b" * 64,
                "reward": index % 2,
            }
            seal = candidate_seal.seal_from_spec(spec)
            specs.append(spec)
            seals.append(seal)
    paths = {
        "specs": root / "specs.jsonl",
        "seals": root / "seals.jsonl",
        "views": root / "verifier-inputs.jsonl",
    }
    candidate_seal.write_jsonl_atomic(paths["specs"], specs)
    candidate_seal.write_jsonl_atomic(paths["seals"], seals)
    candidate_seal.write_jsonl_atomic(
        paths["views"], (candidate_seal.verifier_view(seal) for seal in seals)
    )
    return paths, specs, seals


def test_frozen_candidate_loader_checks_reward_free_join_and_stable_dataset(
    tmp_path,
):
    paths, _specs, seals = _make_frozen_candidates(
        tmp_path, tasks=("task-a", "task-b"), candidates=2
    )

    tasks, dataset = scorer.load_frozen_candidate_tasks(
        paths["specs"], paths["seals"], paths["views"], 0, {"task-a"}
    )

    assert list(tasks) == ["task-a"]
    assert len(tasks["task-a"]) == 2
    assert tasks["task-a"][0]["problem"] == "Solve task-a"
    assert tasks["task-a"][0]["source"].startswith("candidate:")
    assert tasks["task-a"][0]["source_sha256"] == \
        seals[0]["identity"]["trajectory"]["sha256"]
    assert dataset == {
        "kind": "frozen-harbor-candidates",
        "name": "terminal-bench/terminal-bench-2-1",
        "dataset_sha256": "a" * 64,
    }
    assert "reward" not in paths["views"].read_text(encoding="utf-8")

    views = candidate_seal.load_jsonl(paths["views"])
    views[0]["evaluation"] = {"reward": 1}
    candidate_seal.write_jsonl_atomic(paths["views"], views)
    with pytest.raises(
        candidate_seal.SealError,
        match="reward-free seal view",
    ):
        scorer.load_frozen_candidate_tasks(
            paths["specs"], paths["seals"], paths["views"], 0, set()
        )


def test_frozen_candidate_loader_rejects_mutated_bound_trajectory(tmp_path):
    paths, specs, _seals = _make_frozen_candidates(tmp_path, candidates=2)
    trajectory = Path(specs[0]["trajectory_path"])
    trajectory.write_text('{"steps": []}', encoding="utf-8")

    with pytest.raises(
        candidate_seal.SealError,
        match="no longer matches its seal",
    ):
        scorer.load_frozen_candidate_tasks(
            paths["specs"], paths["seals"], paths["views"], 0, set()
        )


def test_frozen_candidate_path_map_preserves_identity_and_checks_relocated_bytes(
    tmp_path,
):
    source = tmp_path / "mac-bundle"
    paths, specs, seals = _make_frozen_candidates(source, candidates=2)
    original_tasks, original_dataset = scorer.load_frozen_candidate_tasks(
        paths["specs"], paths["seals"], paths["views"], 0, set()
    )
    destination = tmp_path / "hephaestus-bundle"
    shutil.copytree(source, destination)
    shutil.rmtree(source)
    relocated_paths = {
        "specs": destination / "specs.jsonl",
        "seals": destination / "seals.jsonl",
        "views": destination / "verifier-inputs.jsonl",
    }
    path_maps = scorer.parse_candidate_path_maps(
        [f"{source}={destination}"]
    )

    relocated_tasks, relocated_dataset = scorer.load_frozen_candidate_tasks(
        relocated_paths["specs"],
        relocated_paths["seals"],
        relocated_paths["views"],
        0,
        set(),
        path_maps,
    )

    assert relocated_dataset == original_dataset
    assert scorer.task_trace_identity(
        "task-a", relocated_tasks["task-a"]
    ) == scorer.task_trace_identity("task-a", original_tasks["task-a"])
    assert {
        row["source"].removeprefix("candidate:")
        for row in relocated_tasks["task-a"]
    } == {seal["candidate_id"] for seal in seals}

    relative_trajectory = Path(specs[0]["trajectory_path"]).relative_to(source)
    (destination / relative_trajectory).write_text(
        '{"steps": []}', encoding="utf-8"
    )
    with pytest.raises(
        candidate_seal.SealError,
        match="no longer matches its seal",
    ):
        scorer.load_frozen_candidate_tasks(
            relocated_paths["specs"],
            relocated_paths["seals"],
            relocated_paths["views"],
            0,
            set(),
            path_maps,
        )


def test_candidate_path_map_cli_rejects_relative_or_ambiguous_roots(tmp_path):
    common = [
        "--candidate-specs", str(tmp_path / "specs.jsonl"),
        "--candidate-seals", str(tmp_path / "seals.jsonl"),
        "--candidate-verifier-inputs", str(tmp_path / "views.jsonl"),
        "--backend", "codex-text",
    ]
    with pytest.raises(SystemExit):
        scorer.parse_args([*common, "--candidate-path-map", "relative=/new"])
    with pytest.raises(SystemExit):
        scorer.parse_args([
            *common,
            "--candidate-path-map", "/old=/new",
            "--candidate-path-map", "/old/nested=/new/nested",
        ])
    args = scorer.parse_args([
        *common,
        "--candidate-path-map", f"{tmp_path / 'old'}={tmp_path / 'new'}",
    ])
    assert args.candidate_path_map == (
        (tmp_path / "old", tmp_path / "new"),
    )


def test_frozen_candidate_main_passes_no_reward_to_llm_select(
    tmp_path, monkeypatch, capsys
):
    paths, _specs, _seals = _make_frozen_candidates(tmp_path, candidates=2)
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            index=1,
            scores=[0.25, 0.75],
            ranking=[1, 0],
            n_comparisons=2,
        )

    monkeypatch.setattr(scorer.llm_verifier, "select", fake_select)
    monkeypatch.setattr(
        scorer,
        "configure_backend",
        lambda *_args: (
            SimpleNamespace(calls=0, failures=0),
            lambda *_args, **_kwargs: (0.5, 0.5),
        ),
    )
    scorer.main(
        [
            "--candidate-specs", str(paths["specs"]),
            "--candidate-seals", str(paths["seals"]),
            "--candidate-verifier-inputs", str(paths["views"]),
            "--backend", "codex-text",
            "--model", "gpt-5.6-terra",
            "--codex-effort", "medium",
            "--expected-candidates", "2",
            "--cache-dir", str(tmp_path / "cache"),
            "--results-jsonl", str(tmp_path / "results.jsonl"),
        ]
    )

    assert len(calls) == 1
    assert set(calls[0]) == {
        "problem", "candidates", "criteria", "n_evaluations", "pivots",
        "seed", "max_workers", "model", "cache", "progress", "on_error",
        "client",
    }
    assert not any("reward" in key for key in calls[0])
    assert calls[0]["problem"] == "Solve task-a"
    assert len(calls[0]["candidates"]) == 2
    assert "loaded 2 traces in 1 tasks" in capsys.readouterr().err


def test_official_loader_reads_recursive_five_candidate_layout(tmp_path):
    root = tmp_path / "terminal_bench_2.1_trajs"
    _make_official_release(root, tasks=("task-a", "task-b"), candidates=5)

    tasks = scorer.load_official_tasks(root, trace_chars=0, selected_tasks=set())

    assert sorted(tasks) == ["task-a", "task-b"]
    assert [len(tasks[task]) for task in sorted(tasks)] == [5, 5]
    row = tasks["task-a"][0]
    assert row["problem"] == "Solve task-a"
    assert "[Command] pwd" in row["trace"]
    assert "[Output]" in row["trace"]
    assert row["source"].endswith("_trajectory.json")
    assert len(row["source_sha256"]) == 64

    original_trace_identity = scorer.task_trace_identity(
        "task-a", tasks["task-a"]
    )
    original_dataset_identity = scorer.dataset_identity(
        "official-terminal-bench-trajectories", tasks
    )
    source_path = root / row["source"]
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["reward"] = 1 - payload["reward"]
    source_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reward_edited = scorer.load_official_tasks(
        root, trace_chars=0, selected_tasks=set()
    )
    assert reward_edited["task-a"][0]["source_sha256"] == row["source_sha256"]
    assert (
        scorer.task_trace_identity("task-a", reward_edited["task-a"])
        == original_trace_identity
    )
    assert (
        scorer.dataset_identity(
            "official-terminal-bench-trajectories", reward_edited
        )
        == original_dataset_identity
    )

    relocated_root = tmp_path / "relocated" / "terminal_bench_2.1_trajs"
    shutil.copytree(root, relocated_root)
    relocated = scorer.load_official_tasks(
        relocated_root, trace_chars=0, selected_tasks=set()
    )
    assert scorer.dataset_identity(
        "official-terminal-bench-trajectories", relocated
    ) == scorer.dataset_identity(
        "official-terminal-bench-trajectories", reward_edited
    )

    selected = scorer.load_official_tasks(
        root, trace_chars=0, selected_tasks={"task-b"}
    )
    assert list(selected) == ["task-b"]


def test_official_dry_run_loads_without_configuring_backend(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "terminal_bench_2.1_trajs"
    _make_official_release(root, tasks=("task-a", "task-b"), candidates=5)
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8371/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_if_configured(*_args, **_kwargs):
        raise AssertionError("dry-run must not construct a provider client")

    monkeypatch.setattr(scorer, "configure_backend", fail_if_configured)
    scorer.main(
        [
            "--trajectory-root",
            str(root),
            "--backend",
            "vllm-logprob",
            "--served-model-identity",
            "deepseek-ai/DeepSeek-V4-Flash-0731@checkpoint-a",
            "--expected-candidates",
            "5",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "loaded 10 traces in 2 tasks" in captured.err
    assert "candidate-counts=5x2" in captured.err
    assert "reward-pools=all-pass:0,swing:2,all-fail:0" in captured.err
    assert "dry run: no verifier calls made" in captured.err
    assert "expected 3" not in captured.err
    assert not (tmp_path / "cache").exists()


def test_vllm_backend_uses_openai_environment_and_deepseek_32k_high_path(
    monkeypatch,
):
    fake_client = SimpleNamespace()
    created = {}

    def fake_create_openai_client(**kwargs):
        created.update(kwargs)
        return fake_client

    monkeypatch.setenv("OPENAI_BASE_URL", "http://gpu-node:8371/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "runtime-secret")
    monkeypatch.setattr(
        scorer.fine_reward, "create_openai_client", fake_create_openai_client
    )

    client, pair_scorer = scorer.configure_backend(
        SimpleNamespace(backend="vllm-logprob"),
        "served-deepseek",
        workers=8,
    )

    assert client is fake_client
    assert pair_scorer is scorer.score_pair_logprob
    assert created == {
        "base_url": "http://gpu-node:8371/v1",
        "api_key": "runtime-secret",
    }
    assert client._gv_vllm_deepseek is True
    assert client._llm_verifier_deepseek is True
    assert client._llm_verifier_model == "served-deepseek"
    assert scorer.BACKEND_DEFAULTS["vllm-logprob"]["workers"] == 8

    request = {}

    def position(token, alternatives=None, actual_logprob=None):
        alternatives = alternatives or [(token, 0.0)]
        return SimpleNamespace(
            token=token,
            logprob=(
                alternatives[0][1]
                if actual_logprob is None
                else actual_logprob
            ),
            top_logprobs=[
                SimpleNamespace(token=alternative, logprob=logprob)
                for alternative, logprob in alternatives
            ],
        )

    def create_completion(**kwargs):
        request.update(kwargs)
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    "analysis\n<score_A>A</score_A>\n"
                    "<score_B>T</score_B>"
                )
            ),
            logprobs=SimpleNamespace(
                content=[
                    position("analysis"),
                    position("\n<score_A>"),
                    # Simulate a provider that omits the sampled A from its
                    # top-logprobs array while still reporting A's logprob on
                    # the position itself.
                    position(
                        "A",
                        [("T", math.log(0.1)), ("X", math.log(0.01))],
                        actual_logprob=math.log(0.9),
                    ),
                    position("</score_A>\n<score_B>"),
                    position("T", [("A", math.log(0.2)), ("T", math.log(0.8))]),
                    position("</score_B>"),
                ]
            ),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])

    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=create_completion)
    )
    monkeypatch.setattr(
        scorer.fine_reward, "USAGE", SimpleNamespace(record=lambda _response: None)
    )
    monkeypatch.setattr(
        scorer.fine_reward,
        "call_verifier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not use generic 4096-token OpenAI path")
        ),
    )
    score_a, score_b = scorer.score_pair_logprob(
        client,
        problem="problem",
        trace_a="trace a",
        trace_b="trace b",
        criterion={"name": "correctness", "description": "be correct"},
        ground_truth_note="",
        model="served-deepseek",
    )

    assert score_a == pytest.approx(0.9)
    assert score_b == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="score-tag token boundary"):
        scorer.expected_logprob_score(
            ["A"], [[("A", 0.0)]], "<score_A>"
        )
    with pytest.raises(RuntimeError, match="available tokens=.*X"):
        scorer.expected_logprob_score(
            ["<score_A>", "X"], [[], [("X", 0.0)]], "<score_A>"
        )
    assert request["max_tokens"] == 32_768
    assert request["temperature"] == 1.0
    assert request["top_p"] == 1.0
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 20
    assert request["extra_body"]["thinking"] == {"type": "enabled"}
    assert request["extra_body"]["reasoning_effort"] == "high"
    assert request["extra_body"]["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
        "reasoning_effort": "high",
    }


def test_logprob_scorer_resamples_missing_score_boundary(monkeypatch, capsys):
    malformed = (
        "analysis without a score block",
        ["analysis without a score block"],
        [[("analysis", 0.0)]],
    )
    valid = (
        "<score_A>A</score_A>\n<score_B>T</score_B>",
        ["<score_A>", "A", "</score_A>\n<score_B>", "T"],
        [[], [("A", 0.0)], [], [("T", 0.0)]],
    )
    samples = iter([malformed, valid])
    calls = []

    def call_verifier(*args, **kwargs):
        calls.append((args, kwargs))
        return next(samples)

    monkeypatch.setattr(scorer.fine_reward, "call_verifier", call_verifier)

    score_a, score_b = scorer.score_pair_logprob(
        SimpleNamespace(),
        problem="problem",
        trace_a="trace a",
        trace_b="trace b",
        criterion={"name": "correctness", "description": "be correct"},
        ground_truth_note="",
    )

    assert (score_a, score_b) == pytest.approx((1.0, 0.0))
    assert len(calls) == 2
    assert "score-tag token boundary not found at <score_A>" in \
        capsys.readouterr().err


def test_logprob_scorer_exhausts_bad_samples_without_fallback(monkeypatch):
    calls = []

    def call_verifier(*_args, **_kwargs):
        calls.append(None)
        return (
            "<score_A>X</score_A>\n<score_B>X</score_B>",
            ["<score_A>", "X", "</score_A>\n<score_B>", "X"],
            [[], [("X", 0.0)], [], [("X", 0.0)]],
        )

    monkeypatch.setattr(scorer.fine_reward, "call_verifier", call_verifier)

    with pytest.raises(
        scorer.RetryableVerifierSampleError,
        match="no A-T token-logprob mass",
    ):
        scorer.score_pair_logprob(
            SimpleNamespace(),
            problem="problem",
            trace_a="trace a",
            trace_b="trace b",
            criterion={"name": "correctness", "description": "be correct"},
            ground_truth_note="",
        )

    assert len(calls) == scorer.LOGPROB_SAMPLE_ATTEMPTS


def test_logprob_scorer_does_not_retry_provider_failures(monkeypatch):
    calls = []

    def call_verifier(*_args, **_kwargs):
        calls.append(None)
        raise RuntimeError("provider disconnected")

    monkeypatch.setattr(scorer.fine_reward, "call_verifier", call_verifier)

    with pytest.raises(RuntimeError, match="provider disconnected"):
        scorer.score_pair_logprob(
            SimpleNamespace(),
            problem="problem",
            trace_a="trace a",
            trace_b="trace b",
            criterion={"name": "correctness", "description": "be correct"},
            ground_truth_note="",
        )

    assert len(calls) == 1


def test_endpoint_and_run_identity_include_inputs_but_not_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    endpoint = scorer.endpoint_identity(
        "vllm-logprob",
        "http://user:url-secret@gpu-node:8371/v1?api_key=query-secret",
    )
    row = {
        "trial": "task__trial",
        "reward": 1.0,
        "problem": "problem",
        "trace": "trace",
        "source": "task/task__trial_trajectory.json",
        "source_sha256": "a" * 64,
    }
    tasks = {"task": [row]}
    dataset = scorer.dataset_identity("official", tasks)
    payload = scorer.run_identity_payload(
        dataset=dataset,
        tasks=tasks,
        backend="vllm-logprob",
        model="served-deepseek",
        served_model_identity="checkpoint-a",
        endpoint=endpoint,
        scoring_mode="token-logprob-expectation",
        scorer_source_sha256="b" * 64,
        request_config=scorer.vllm_deepseek_request_config(),
        n_evaluations=2,
        pivots=1,
        seed=0,
        trace_chars=0,
    )
    rendered = json.dumps(payload, sort_keys=True)

    assert payload["backend"] == "vllm-logprob"
    assert payload["model"] == "served-deepseek"
    assert payload["served_model_identity"] == "checkpoint-a"
    assert payload["scorer_source_sha256"] == "b" * 64
    assert payload["request_config"]["max_tokens"] == 32_768
    assert payload["request_config"]["temperature"] == 1.0
    assert payload["request_config"]["top_p"] == 1.0
    assert payload["request_config"]["reasoning_effort"] == "high"
    assert payload["dataset"]["name"] == "official"
    assert "root" not in payload["dataset"]
    assert payload["task_traces"]["task"]
    assert payload["endpoint"] == endpoint
    assert "url-secret" not in rendered
    assert "query-secret" not in rendered
    assert "runtime-secret" not in rendered
    assert payload["protocol"] == scorer.SCORING_PROTOCOL

    reward_changed = {"task": [{**row, "reward": 0.0}]}
    reward_changed_payload = scorer.run_identity_payload(
        dataset=scorer.dataset_identity("official", reward_changed),
        tasks=reward_changed,
        backend="vllm-logprob",
        model="served-deepseek",
        served_model_identity="checkpoint-a",
        endpoint=endpoint,
        scoring_mode="token-logprob-expectation",
        scorer_source_sha256="b" * 64,
        request_config=scorer.vllm_deepseek_request_config(),
        n_evaluations=2,
        pivots=1,
        seed=0,
        trace_chars=0,
    )
    assert scorer.stable_digest(reward_changed_payload) == scorer.stable_digest(
        payload
    )

    checkpoint_changed = {**payload, "served_model_identity": "checkpoint-b"}
    source_changed = {**payload, "scorer_source_sha256": "c" * 64}
    request_changed = {
        **payload,
        "request_config": {**payload["request_config"], "top_p": 0.9},
    }
    assert len({
        scorer.stable_digest(payload),
        scorer.stable_digest(checkpoint_changed),
        scorer.stable_digest(source_changed),
        scorer.stable_digest(request_changed),
    }) == 4


def test_raw_cache_identity_binds_checkpoint_source_and_request_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    row = {
        "trial": "task__trial",
        "reward": 1.0,
        "problem": "problem",
        "trace": "trace",
        "source": "task/task__trial_trajectory.json",
        "source_sha256": "a" * 64,
    }
    rows = [row]
    dataset = scorer.dataset_identity("official", {"task": rows})
    endpoint = scorer.endpoint_identity(
        "vllm-logprob", "http://gpu-node:8371/v1"
    )
    request_config = scorer.vllm_deepseek_request_config()

    def cache(checkpoint="checkpoint-a", source="b" * 64, config=None):
        return scorer.cache_file_for(
            tmp_path / "cache",
            dataset,
            "task",
            rows,
            "vllm-logprob",
            "served-deepseek",
            checkpoint,
            endpoint,
            source,
            request_config if config is None else config,
        )

    assert len(scorer.scorer_source_hash()) == 64
    assert len({
        cache(),
        cache(checkpoint="checkpoint-b"),
        cache(source="c" * 64),
        cache(config={**request_config, "temperature": 0.7}),
    }) == 4


def test_vllm_cli_requires_nonempty_served_model_identity(tmp_path):
    base = [
        "--trajectory-root", str(tmp_path),
        "--backend", "vllm-logprob",
    ]
    with pytest.raises(SystemExit):
        scorer.parse_args(base)
    with pytest.raises(SystemExit):
        scorer.parse_args([*base, "--served-model-identity", "   "])

    args = scorer.parse_args([
        *base,
        "--served-model-identity", " checkpoint-a ",
    ])
    assert args.served_model_identity == "checkpoint-a"


def test_jsonl_task_result_is_resumed_and_select_keeps_raise_policy(
    tmp_path, monkeypatch
):
    root = tmp_path / "terminal_bench_2.1_trajs"
    _make_official_release(root, candidates=2)
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8371/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "runtime-secret")
    calls = []
    configurations = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            index=1,
            scores=[0.25, 0.75],
            ranking=[1, 0],
            n_comparisons=2,
        )

    def fake_configure(*args):
        configurations.append(args)
        return SimpleNamespace(), lambda *_args, **_kwargs: (0.5, 0.5)

    monkeypatch.setattr(scorer.llm_verifier, "select", fake_select)
    monkeypatch.setattr(scorer, "configure_backend", fake_configure)
    results = tmp_path / "results.jsonl"
    argv = [
        "--trajectory-root",
        str(root),
        "--backend",
        "vllm-logprob",
        "--served-model-identity",
        "deepseek-ai/DeepSeek-V4-Flash-0731@checkpoint-a",
        "--expected-candidates",
        "2",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--results-jsonl",
        str(results),
    ]

    scorer.main(argv)
    scorer.main(argv)

    assert len(calls) == 1
    assert len(configurations) == 1
    assert calls[0]["on_error"] == "raise"
    assert len(calls[0]["candidates"]) == 2
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["task"] == "task-a"
    assert records[0]["scores"] == [0.25, 0.75]
    assert records[0]["ranking"] == [1, 0]
    assert records[0]["run_identity"]["protocol"] == scorer.SCORING_PROTOCOL
    assert records[0]["served_model_identity"].endswith("@checkpoint-a")
    assert len(records[0]["scorer_source_sha256"]) == 64
    assert records[0]["request_config"] == \
        scorer.vllm_deepseek_request_config()
    assert "runtime-secret" not in results.read_text()
    assert str(root) not in results.read_text()


def test_jsonl_reader_tolerates_a_crash_truncated_line(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    path.write_text('{"run_id": "unfinished"', encoding="utf-8")
    record = {
        "schema_version": scorer.RESULT_SCHEMA_VERSION,
        "run_id": "run-1",
        "task": "task-a",
        "scores": [1.0],
        "ranking": [0],
    }

    scorer.append_task_record(path, record)

    assert scorer.load_completed_records(path, "run-1") == {"task-a": record}
    assert "ignoring malformed JSONL line" in capsys.readouterr().err


def test_vllm_backend_requires_both_openai_environment_variables(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        scorer.configure_backend(
            SimpleNamespace(backend="vllm-logprob"), "model", workers=1
        )

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8371/v1")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        scorer.configure_backend(
            SimpleNamespace(backend="vllm-logprob"), "model", workers=1
        )


def test_provider_usage_is_durable_and_aggregated(tmp_path, monkeypatch):
    usage_path = tmp_path / "calls.usage.jsonl"
    client = SimpleNamespace(
        _gv_usage_path=usage_path,
        _gv_usage_context={
            "run_id": "run-1",
            "task": "task-a",
            "backend": "openrouter-logprob",
            "model": "deepseek/deepseek-v4-flash",
            "served_model_identity": "provider-pinned",
        },
    )
    monkeypatch.setattr(
        scorer.fine_reward,
        "USAGE",
        SimpleNamespace(record=lambda _response: (100, 20, 30, 25)),
    )
    response = SimpleNamespace(
        id="call-1",
        model="deepseek/deepseek-v4-flash",
        model_extra={"provider": "DigitalOcean"},
        usage=SimpleNamespace(model_extra={"cost": 0.125}),
    )

    scorer.record_call_usage(client, response, "secret prompt", "stop")

    record = json.loads(usage_path.read_text())
    assert "secret prompt" not in usage_path.read_text()
    assert record["response_provider"] == "DigitalOcean"
    assert record["input_tokens"] == 100
    assert record["cached_input_tokens"] == 20
    assert record["output_tokens"] == 30
    assert record["reasoning_tokens"] == 25
    assert record["cost_usd"] == pytest.approx(0.125)
    assert scorer.aggregate_call_usage(usage_path, "run-1", "task-a") == {
        "calls": 1,
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "uncached_input_tokens": 80,
        "output_tokens": 30,
        "reasoning_tokens": 25,
        "cost_usd": pytest.approx(0.125),
    }


def test_codex_text_records_reported_cli_usage_durably(tmp_path, monkeypatch):
    from experiments.common import codex_endpoint

    answer = "<score_A>A</score_A>\n<score_B>T</score_B>"
    events = [
        {"type": "thread.started", "thread_id": "thread-usage-1"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": answer},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1_200,
                "cached_input_tokens": 800,
                "cache_write_input_tokens": 0,
                "output_tokens": 90,
                "reasoning_output_tokens": 70,
            },
        },
    ]

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        assert "--json" in command
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr="",
        )

    added = []
    monkeypatch.setattr(codex_endpoint.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scorer.fine_reward,
        "USAGE",
        SimpleNamespace(add=lambda **values: added.append(values)),
    )
    usage_path = tmp_path / "codex.usage.jsonl"
    client = codex_endpoint.CodexRunner(
        model="gpt-5.6-terra",
        effort="high",
        timeout=30,
        concurrency=1,
        service_tier="fast",
    )
    client._gv_usage_path = usage_path
    client._gv_usage_context = {
        "run_id": "run-terra",
        "task": "task-a",
        "backend": "codex-text",
        "model": "gpt-5.6-terra",
        "served_model_identity": "gpt-5.6-terra",
    }

    scores = scorer.score_pair_text(
        client,
        problem="private task",
        trace_a="private trace a",
        trace_b="private trace b",
        criterion={"name": "correctness", "description": "be correct"},
        ground_truth_note="",
        model="gpt-5.6-terra",
    )

    assert scores == pytest.approx((1.0, 0.0))
    record = json.loads(usage_path.read_text(encoding="utf-8"))
    assert "private task" not in usage_path.read_text(encoding="utf-8")
    assert record["call_id"] == "thread-usage-1"
    assert record["response_model"] == "gpt-5.6-terra"
    assert record["response_provider"] == "codex-cli"
    assert record["input_tokens"] == 1_200
    assert record["cached_input_tokens"] == 800
    assert record["output_tokens"] == 90
    assert record["reasoning_tokens"] == 70
    assert 'service_tier="fast"' in commands[0]
    assert commands[0][commands[0].index("--enable") + 1] == "fast_mode"
    assert added == [{
        "input_tokens": 1_200,
        "cached_input_tokens": 800,
        "output_tokens": 90,
        "reasoning_tokens": 70,
    }]
    assert scorer.aggregate_call_usage(
        usage_path, "run-terra", "task-a"
    )["uncached_input_tokens"] == 400


def test_codex_effort_and_tier_change_run_and_raw_cache_identity(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(scorer, "OFFICIAL_CRITERIA_PATH", _write_criteria(tmp_path))
    row = {
        "trial": "task__trial",
        "reward": 1.0,
        "problem": "problem",
        "trace": "trace",
        "source": "task/task__trial_trajectory.json",
        "source_sha256": "a" * 64,
    }
    tasks = {"task": [row]}
    dataset = scorer.dataset_identity("official", tasks)
    common = {
        "dataset": dataset,
        "tasks": tasks,
        "backend": "codex-text",
        "model": "gpt-5.6-terra",
        "served_model_identity": "gpt-5.6-terra",
        "endpoint": scorer.endpoint_identity("codex-text"),
        "scoring_mode": "text-argmax-20bin",
        "scorer_source_sha256": "b" * 64,
        "n_evaluations": 2,
        "pivots": 1,
        "seed": 0,
        "trace_chars": 0,
    }
    low = scorer.codex_request_config("low")
    high = scorer.codex_request_config("high")
    fast = scorer.codex_request_config("low", "fast")
    low_run = scorer.run_identity_payload(request_config=low, **common)
    high_run = scorer.run_identity_payload(request_config=high, **common)

    assert low_run["request_config"] == {
        "reasoning_effort": "low", "service_tier": "standard"}
    assert high_run["request_config"] == {
        "reasoning_effort": "high", "service_tier": "standard"}
    assert scorer.stable_digest(low_run) != scorer.stable_digest(high_run)
    fast_run = scorer.run_identity_payload(request_config=fast, **common)
    assert fast_run["request_config"]["service_tier"] == "fast"
    assert scorer.stable_digest(low_run) != scorer.stable_digest(fast_run)

    def cache(config):
        return scorer.cache_file_for(
            tmp_path / "cache",
            dataset,
            "task",
            [row],
            "codex-text",
            "gpt-5.6-terra",
            "gpt-5.6-terra",
            scorer.endpoint_identity("codex-text"),
            "b" * 64,
            config,
        )

    assert cache(low) != cache(high)
    assert cache(low) != cache(fast)


def test_codex_fast_tier_is_rejected_for_non_codex_backend(tmp_path):
    with pytest.raises(SystemExit):
        scorer.parse_args([
            str(tmp_path),
            "--backend", "deepseek-logprob",
            "--codex-service-tier", "fast",
        ])
