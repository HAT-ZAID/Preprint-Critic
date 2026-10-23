"""Contract tests for the shipped LoRA adapter in ``rc_project_adapter_V3``.

These tests protect the reproducibility metadata of the finished run: the PEFT
configuration, the shape of the weight file, the training budget recorded in
``trainer_state.json`` and the hyperparameters frozen in ``training_args.bin``.
Everything is read from disk; no base model and no GPU are required.
"""

import io
import json
import math
import pickle
import struct
import zipfile

import pytest

from conftest import REPO_ROOT, load_json

ADAPTER_DIR = REPO_ROOT / "rc_project_adapter_V3"
CONFIG_PATH = ADAPTER_DIR / "adapter_config.json"
WEIGHTS_PATH = ADAPTER_DIR / "adapter_model.safetensors"
TRAINER_STATE_PATH = ADAPTER_DIR / "trainer_state.json"
TRAINING_ARGS_PATH = ADAPTER_DIR / "training_args.bin"

#: The seven Qwen3 projection matrices a LoRA adapter must touch.
EXPECTED_TARGET_MODULES = [
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]

#: 7 projections x 36 layers x (rank 8 x in_features + out_features x rank 8).
EXPECTED_TRAINABLE_PARAMETERS = 16_515_072
EXPECTED_TENSOR_COUNT = 504

CONFIG = load_json("rc_project_adapter_V3/adapter_config.json")
TRAINER_STATE = load_json("rc_project_adapter_V3/trainer_state.json")


def read_safetensors_header(path):
    """Return ``(metadata, tensors)`` from a safetensors header without loading data.

    The format is: 8-byte little-endian uint64 header length, then a UTF-8 JSON
    header whose ``data_offsets`` point into the trailing byte buffer.
    """
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        assert len(raw_length) == 8, (
            f"expected an 8-byte safetensors header length in {path}, "
            f"found {len(raw_length)} bytes"
        )
        (header_length,) = struct.unpack("<Q", raw_length)
        raw_header = handle.read(header_length)
        assert len(raw_header) == header_length, (
            f"expected {header_length} header bytes in {path}, "
            f"found {len(raw_header)} (file truncated?)"
        )

    header = json.loads(raw_header)
    metadata = header.pop("__metadata__", {})
    return metadata, header


class _Stub:
    """Permissive stand-in for any class referenced by the pickled TrainingArguments.

    Pickle may call the class for reduced values (for example the string-backed
    ``SchedulerType`` enum), so the constructor accepts and ignores arguments.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:  # pragma: no cover - torch never uses the tuple form here
            self.__dict__["_state"] = state


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that maps every referenced class to :class:`_Stub`."""

    def find_class(self, module, name):
        return type(f"Stub_{name}", (_Stub,), {})


@pytest.fixture(scope="module")
def training_args():
    """Unpickle ``training_args.bin`` through the stub unpickler, once per session.

    A torch ``save`` of a dataclass is a zip archive containing ``data.pkl``;
    reading that member avoids importing torch while still yielding the real
    attribute values.
    """
    with zipfile.ZipFile(TRAINING_ARGS_PATH) as archive:
        pkl_members = [name for name in archive.namelist() if name.endswith(".pkl")]
        assert len(pkl_members) == 1, (
            "expected exactly one .pkl member in training_args.bin, "
            f"found {pkl_members!r}"
        )
        payload = archive.read(pkl_members[0])

    loaded = _StubUnpickler(io.BytesIO(payload)).load()
    attributes = loaded.__dict__
    assert attributes, "stub unpickler produced no attributes for training_args.bin"
    return attributes


@pytest.mark.parametrize(
    "key,expected",
    [
        ("r", 8),
        ("lora_alpha", 16),
        ("lora_dropout", 0.05),
        ("base_model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507"),
        ("inference_mode", True),
    ],
)
def test_adapter_config_values(key, expected):
    """The adapter must be the advertised rank-8, alpha-16 LoRA of Qwen3-4B.

    These numbers define the name and the scale of the released checkpoint;
    changing any of them silently invalidates the reported training run.
    """
    assert key in CONFIG, f"adapter_config.json is missing key {key!r}"
    assert CONFIG[key] == expected, (
        f"adapter_config.json[{key!r}]: expected {expected!r}, found {CONFIG[key]!r}"
    )


def test_adapter_config_target_modules():
    """LoRA must be attached to exactly the seven attention/MLP projections.

    Order is not part of the contract, but the set must be complete: a missing
    projection means part of the model was never adapted.
    """
    modules = CONFIG.get("target_modules")
    assert isinstance(modules, list), (
        f"expected target_modules to be a list, found {type(modules).__name__}"
    )
    assert len(modules) == 7, (
        f"expected 7 target modules, found {len(modules)}: {modules!r}"
    )
    assert sorted(modules) == EXPECTED_TARGET_MODULES, (
        f"expected target modules {EXPECTED_TARGET_MODULES!r}, found {sorted(modules)!r}"
    )


def test_adapter_weights_header_shape():
    """The weight file must contain 504 float32 tensors and 16,515,072 parameters.

    Reading only the safetensors header keeps the test fast and avoids
    materialising ~66 MB of weights. The parameter total is what makes the
    adapter's size claim checkable without a GPU.
    """
    metadata, tensors = read_safetensors_header(WEIGHTS_PATH)

    assert metadata.get("format") == "pt", (
        f"expected safetensors metadata format 'pt', found {metadata.get('format')!r}"
    )
    assert len(tensors) == EXPECTED_TENSOR_COUNT, (
        f"expected {EXPECTED_TENSOR_COUNT} tensors in adapter_model.safetensors, "
        f"found {len(tensors)}"
    )

    wrong_dtypes = {name: spec["dtype"] for name, spec in tensors.items() if spec["dtype"] != "F32"}
    assert not wrong_dtypes, (
        "expected every tensor dtype to be 'F32', found non-F32 tensors: "
        f"{dict(list(wrong_dtypes.items())[:5])}"
    )

    total = sum(math.prod(spec["shape"]) for spec in tensors.values())
    assert total == EXPECTED_TRAINABLE_PARAMETERS, (
        f"expected {EXPECTED_TRAINABLE_PARAMETERS} trainable parameters across the "
        f"adapter, found {total}"
    )


@pytest.mark.parametrize(
    "key,expected",
    [
        ("global_step", 350),
        ("best_global_step", 350),
        ("num_train_epochs", 1),
        ("max_steps", 350),
    ],
)
def test_trainer_state_budget(key, expected):
    """The recorded run must be the single-epoch, 350-step budget that was reported.

    ``best_global_step`` equal to the final step is what justifies shipping the
    last checkpoint as the best one.
    """
    assert key in TRAINER_STATE, f"trainer_state.json is missing key {key!r}"
    assert TRAINER_STATE[key] == expected, (
        f"trainer_state.json[{key!r}]: expected {expected!r}, found {TRAINER_STATE[key]!r}"
    )


def test_trainer_state_eval_history():
    """Exactly five evaluations at steps 80/160/240/320/350 with the reported losses.

    The eval curve is the only evidence in the repository that the fine-tune
    actually improved; a missing or reordered entry breaks that claim.
    """
    eval_entries = [entry for entry in TRAINER_STATE["log_history"] if "eval_loss" in entry]

    assert len(eval_entries) == 5, (
        f"expected 5 log entries containing eval_loss, found {len(eval_entries)}: "
        f"{[entry.get('step') for entry in eval_entries]}"
    )

    steps = [entry["step"] for entry in eval_entries]
    assert steps == [80, 160, 240, 320, 350], (
        f"expected eval steps [80, 160, 240, 320, 350], found {steps}"
    )

    losses = [entry["eval_loss"] for entry in eval_entries]
    assert losses == pytest.approx([2.2040, 2.1824, 2.1725, 2.1661, 2.1653], abs=1e-3), (
        "expected eval losses ~[2.2040, 2.1824, 2.1725, 2.1661, 2.1653] "
        f"(abs tol 1e-3), found {losses}"
    )


def test_trainer_state_best_metric_matches_final_eval():
    """``best_metric`` must equal the last eval loss, i.e. the shipped checkpoint.

    If these diverge, the adapter being distributed is not the one the recorded
    best score belongs to.
    """
    eval_entries = [entry for entry in TRAINER_STATE["log_history"] if "eval_loss" in entry]
    last_eval_loss = eval_entries[-1]["eval_loss"]

    assert TRAINER_STATE["best_metric"] == last_eval_loss, (
        f"expected best_metric to equal the final eval loss {last_eval_loss!r}, "
        f"found {TRAINER_STATE['best_metric']!r}"
    )


@pytest.mark.parametrize(
    "key,expected",
    [
        ("load_best_model_at_end", True),
        ("eval_steps", 80),
        ("save_steps", 80),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", 8),
        ("seed", 42),
    ],
)
def test_training_args_values(training_args, key, expected):
    """The frozen TrainingArguments must match the documented optimisation setup.

    Effective batch size is 1 x 8 accumulation steps, evaluation happens every
    80 optimizer steps, and the best checkpoint is restored at the end.
    """
    assert key in training_args, f"training_args.bin is missing attribute {key!r}"
    assert training_args[key] == expected, (
        f"training_args.bin[{key!r}]: expected {expected!r}, found {training_args[key]!r}"
    )


def test_training_args_learning_rate(training_args):
    """The LoRA learning rate must be 2e-4, the value quoted in the model notes."""
    assert "learning_rate" in training_args, (
        "training_args.bin is missing attribute 'learning_rate'"
    )
    learning_rate = training_args["learning_rate"]
    assert learning_rate == pytest.approx(2e-4, rel=1e-6), (
        f"expected learning_rate 2e-4, found {learning_rate!r}"
    )
