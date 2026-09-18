from research.experiment_tracking import (
    ExperimentCondition,
    ExperimentResult,
    RunMetrics,
    TurnResult,
)
from research.experiment_tracking.conditions import alpha_sweep_conditions
from research.experiment_tracking.eprime import turn_result_from_eprime_texts
from research.experiment_tracking.eprime import extract_current_assistant_response
from research.experiment_tracking.gpu_preflight import (
    has_required_vram,
    required_vram_bytes,
)
from research.experiment_tracking.wandb_adapter import WandbTracker


def sample_condition():
    return ExperimentCondition(
        condition_id="A2",
        comparison_group="MLR-20-A-alpha-sweep",
        spotlight=True,
        alpha=0.10,
        implementation="spotlight",
        model="Qwen/Qwen2-1.5B-Instruct",
        constraint_formulation="E-Prime",
        constraint_complexity="state-of-being-verbs-and-listed-contractions",
        intervention_timing="prefill",
        history="rolling-long-conversation",
        turns=2,
        replicate=1,
        temperature=0.0,
        max_tokens=128,
        history_window_messages=16,
        seed=0,
        notebook="notebooks/demo_spotlight_e_prime_colab.ipynb",
    )


def sample_result():
    turns = [
        TurnResult(
            condition="A2",
            turn=1,
            prompt="Answer without state-of-being verbs.",
            response="Use active phrasing.",
            compliant=True,
            violation_count=0,
            state_of_being_count=0,
            contraction_count=0,
        ),
        TurnResult(
            condition="A2",
            turn=2,
            prompt="Continue.",
            response="It is concise.",
            compliant=False,
            violation_count=1,
            state_of_being_count=1,
            contraction_count=0,
        ),
    ]
    return ExperimentResult(
        condition=sample_condition(),
        turns=turns,
        provenance={
            "git_sha": "abc123",
            "torch_version": "unknown",
            "vllm_version": "unknown",
            "spotlight_version": "abc123",
        },
        full_result={"transcript": turns[0].response + "\n" + turns[1].response},
    )


def test_run_metrics_from_turns():
    metrics = RunMetrics.from_turns(sample_result().turns)

    assert metrics.compliance_rate == 0.5
    assert metrics.mean_violations == 0.5
    assert metrics.total_violations == 1
    assert metrics.mean_state_of_being_count == 0.5
    assert metrics.total_contractions == 0
    assert metrics.first_violation_turn == 2


def test_disabled_wandb_tracker_returns_complete_payload():
    payload = WandbTracker(enabled=False).log_result(sample_result())

    assert payload["config"]["condition_id"] == "A2"
    assert payload["config"]["git_sha"] == "abc123"
    assert payload["metrics"]["compliance_rate"] == 0.5
    assert payload["turns"][0]["prompt"] == "Answer without state-of-being verbs."
    assert payload["turns"][0]["model_response"] == "Use active phrasing."
    assert payload["full_result"]["transcript"]


def test_turn_result_preserves_full_prompt_and_current_user_message():
    turn = TurnResult(
        condition="spotlight",
        turn=1,
        prompt="SYSTEM\nASSISTANT: Prior text is context.\nUSER: Current turn\nASSISTANT:",
        user_message="Current turn",
        response="Use active phrasing.",
        compliant=True,
        violation_count=0,
        state_of_being_count=0,
        contraction_count=0,
    )

    row = turn.to_row()

    assert row["prompt"].startswith("SYSTEM")
    assert row["user_message"] == "Current turn"
    assert row["model_response"] == "Use active phrasing."


def test_enabled_wandb_tracker_emits_run_table_metrics_and_artifact():
    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class FakeRun:
        def __init__(self):
            self.artifacts = []
            self.finished = False

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True

    class FakeWandb:
        Table = FakeTable
        Artifact = FakeArtifact

        def __init__(self):
            self.run = FakeRun()
            self.init_kwargs = None
            self.logged = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

        def log(self, payload):
            self.logged = payload

    fake_wandb = FakeWandb()
    payload = WandbTracker(
        group="MLR-20-A-alpha-sweep",
        run_name="A2-r1-alpha-0.10",
        tags=["MLR-20"],
        enabled=True,
        wandb_module=fake_wandb,
    ).log_result(sample_result())

    assert fake_wandb.init_kwargs["project"] == "vllm-hook-eprime"
    assert fake_wandb.init_kwargs["group"] == "MLR-20-A-alpha-sweep"
    assert fake_wandb.init_kwargs["name"] == "A2-r1-alpha-0.10"
    assert fake_wandb.init_kwargs["config"]["git_sha"] == "abc123"
    assert fake_wandb.logged["compliance_rate"] == 0.5
    assert fake_wandb.logged["turn_results"].columns[:9] == [
        "condition",
        "turn",
        "prompt",
        "user_message",
        "model_response",
        "compliant",
        "violation_count",
        "state_of_being_count",
        "contraction_count",
    ]
    assert len(fake_wandb.logged["turn_results"].rows) == 2
    assert fake_wandb.run.artifacts[0].type == "experiment-result"
    assert fake_wandb.run.artifacts[0].files[0][1] == "experiment_result.json"
    assert fake_wandb.run.finished is True
    assert payload["metrics"]["first_violation_turn"] == 2


def test_alpha_sweep_conditions_avoid_invalid_factorial_combinations():
    conditions = alpha_sweep_conditions(
        model="Qwen/Qwen2-1.5B-Instruct",
        turns=10,
        temperature=0.0,
        max_tokens=256,
        history_window_messages=16,
    )

    by_id = {condition.condition_id: condition for condition in conditions}
    assert by_id["A0"].spotlight is False
    assert by_id["A0"].alpha is None
    assert by_id["A1"].spotlight is True
    assert by_id["A1"].alpha == 0.05
    assert by_id["A4"].alpha == 0.20


def test_gpu_preflight_reservation_calculation():
    total = 80 * 1024**3
    required = required_vram_bytes(total, 0.30)

    assert required == int(total * 0.30)
    assert has_required_vram(
        free_bytes=required,
        total_bytes=total,
        gpu_memory_utilization=0.30,
    )
    assert not has_required_vram(
        free_bytes=required - 1,
        total_bytes=total,
        gpu_memory_utilization=0.30,
    )


def test_eprime_tracking_ignores_user_message_violations():
    turn = turn_result_from_eprime_texts(
        condition="spotlight",
        turn=1,
        user_message="Is there a way to be concise about this?",
        assistant_response="Use concise active phrasing.",
        score_fn=_simple_eprime_score,
    )

    assert turn.prompt == "Is there a way to be concise about this?"
    assert turn.response == "Use concise active phrasing."
    assert turn.compliant is True
    assert turn.violation_count == 0
    assert turn.state_of_being_count == 0
    assert turn.contraction_count == 0


def test_eprime_tracking_ignores_prompt_history_violations():
    prompt = "SYSTEM\nUSER: Earlier text is noisy.\nASSISTANT: It was verbose.\nUSER: Rewrite this."
    assistant_response = extract_current_assistant_response(
        f"{prompt}\nASSISTANT: Use concise active phrasing.",
        prompt,
    )
    turn = turn_result_from_eprime_texts(
        condition="spotlight",
        turn=1,
        user_message="Rewrite this.",
        assistant_response=assistant_response,
        score_fn=_simple_eprime_score,
        extra={"prompt": prompt},
    )

    assert turn.response == "Use concise active phrasing."
    assert turn.compliant is True
    assert turn.violation_count == 0
    assert turn.state_of_being_count == 0
    assert turn.contraction_count == 0


def test_eprime_tracking_counts_assistant_response_violations():
    turn = turn_result_from_eprime_texts(
        condition="spotlight",
        turn=1,
        user_message="Give a concise active phrasing.",
        assistant_response="This is concise.",
        score_fn=_simple_eprime_score,
    )

    assert turn.prompt == "Give a concise active phrasing."
    assert turn.response == "This is concise."
    assert turn.compliant is False
    assert turn.violation_count == 1
    assert turn.state_of_being_count == 1
    assert turn.contraction_count == 0


def test_aggregate_eprime_metrics_use_extracted_current_assistant_response_only():
    prompt_with_violations = "SYSTEM\nUSER: There is context.\nASSISTANT: It was context.\nUSER: Current"
    clean_response = extract_current_assistant_response(
        f"{prompt_with_violations}\nASSISTANT: Use active phrasing.",
        prompt_with_violations,
    )
    violating_response = extract_current_assistant_response(
        f"{prompt_with_violations}\nASSISTANT: This is concise.",
        prompt_with_violations,
    )
    turns = [
        turn_result_from_eprime_texts(
            condition="spotlight",
            turn=1,
            user_message="Current",
            assistant_response=clean_response,
            score_fn=_simple_eprime_score,
        ),
        turn_result_from_eprime_texts(
            condition="spotlight",
            turn=2,
            user_message="Current",
            assistant_response=violating_response,
            score_fn=_simple_eprime_score,
        ),
    ]

    metrics = RunMetrics.from_turns(turns)

    assert metrics.compliance_rate == 0.5
    assert metrics.mean_violations == 0.5
    assert metrics.total_violations == 1
    assert metrics.mean_state_of_being_count == 0.5
    assert metrics.total_contractions == 0


def test_extract_current_assistant_response_removes_prompt_echo_and_transcript_markers():
    prompt = "SYSTEM\nUSER: Prior question is here.\nASSISTANT: Prior reply was here.\nUSER: Current turn\nASSISTANT:"
    raw_generation = f"{prompt} Use active phrasing.\nUSER: Follow-up leaked."

    response = extract_current_assistant_response(raw_generation, prompt)

    assert response == "Use active phrasing."
    assert "USER:" not in response
    assert "Prior reply" not in response


def _simple_eprime_score(text):
    state_of_being_count = sum(
        1 for token in text.lower().replace("?", "").replace(".", "").split()
        if token in {"am", "is", "are", "was", "were", "be", "being", "been"}
    )
    contraction_count = sum(
        text.lower().count(contraction)
        for contraction in ["i'm", "you're", "we're", "they're", "it's"]
    )
    violation_count = state_of_being_count + contraction_count
    return {
        "state_of_being_count": state_of_being_count,
        "contraction_count": contraction_count,
        "e_prime_violation_count": violation_count,
        "e_prime_retained": violation_count == 0,
    }
