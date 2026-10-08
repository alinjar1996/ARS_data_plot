"""Exercise the notebook's actual helpers without executing plots or other cells."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def helpers():
    path = Path(__file__).resolve().parents[1] / "benchmark_stat_batch_sm.ipynb"
    notebook = json.loads(path.read_text())
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def phase_masked_log_weights(" in "".join(cell["source"])
    )
    nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"PHASE_WEIGHT_RULES", "PAPER_COST_LABELS"}
            for target in node.targets
        ))
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("task", ["Ball Lift", "Box Lift", "Tray Push"])
def test_every_component_uses_per_step_phase_and_paper_mapping(helpers, task):
    names = list(helpers["PAPER_COST_LABELS"][task])
    # Repeated switches catch implementations that mask only after first contact.
    phase = np.array([0, 1, 1, 0, 1])
    values = np.full((len(phase), len(names)), np.e)
    original = values.copy()
    result = helpers["phase_masked_log_weights"](values, phase, names, task)
    _, approach, transport, unused = helpers["PHASE_WEIGHT_RULES"][task]
    for column, name in enumerate(names):
        active = np.ones(len(phase), dtype=bool)
        if name in approach:
            active = phase == 0
        elif name in transport:
            active = phase == 1
        elif name in unused:
            active[:] = False
        expected = np.ones(len(phase))
        if task != "Tray Push" and name == "distance":
            expected += np.where(phase == 1, np.log(10), 0)
        expected[~active] = 0
        np.testing.assert_allclose(result[:, column], expected)
        assert np.all(result[~active, column] == 0)
    np.testing.assert_array_equal(values, original)


def test_inactive_is_zero_even_for_nonpositive_or_nonfinite_raw_outputs(helpers):
    values = np.array([[np.nan], [0.0], [np.inf], [-2.0]])
    mask = helpers["phase_masked_log_weights"]
    inactive = mask(values, [1] * 4, ["eef_to_obj"], "Ball Lift")
    active = mask(values, [0] * 4, ["eef_to_obj"], "Ball Lift")
    assert np.all(inactive == 0)
    assert np.all(np.isnan(active))


def test_log_is_taken_before_phase_mask_not_after(helpers):
    result = helpers["phase_masked_log_weights"](
        [[0.1], [0.1], [1.0]], [0, 1, 0], ["eef_to_obj"], "Ball Lift"
    )
    np.testing.assert_allclose(result[:, 0], [np.log(0.1), 0, 0])


@pytest.mark.parametrize("phase", [[0], [0, 2], [0, np.nan], [[0], [1]]])
def test_invalid_or_misaligned_phase_is_rejected(helpers, phase):
    with pytest.raises(ValueError, match="planning phase"):
        helpers["phase_masked_log_weights"]([[2], [2]], phase, ["theta"], "Ball Lift")


def test_unknown_component_is_not_silently_assumed_active(helpers):
    with pytest.raises(ValueError, match="unmapped cost components"):
        helpers["phase_masked_log_weights"]([[2]], [0], ["unknown"], "Ball Lift")


def test_misaligned_weight_columns_are_rejected(helpers):
    with pytest.raises(ValueError, match="names/values"):
        helpers["phase_masked_log_weights"]([[2, 3]], [0], ["theta"], "Ball Lift")


def test_labels_use_paper_symbols_and_task_specific_phase_names(helpers):
    title = helpers["cost_weight_panel_title"]
    assert "$w_{yz}$" in title("Box Lift", "z-axis")
    assert "lift only" in title("Box Lift", "object_orientation")
    assert "symbol unspecified" in title("Box Lift", "object_orientation")
    assert "unused: always 0" in title("Box Lift", "smoothness")
    assert "both phases" in title("Ball Lift", "smoothness")
    assert r"$w_p^{\mathrm{push}}$" in title("Tray Push", "position_move")
    assert "push only" in title("Tray Push", "position_move")
    assert "approach only" in title("Tray Push", "orientation_pick")


def test_loader_uses_pre_step_phase_linear_tray_values_and_nan_padding(helpers, tmp_path):
    values = np.empty(2, dtype=object)
    values[:] = [np.array([[np.e], [np.e**2]]), np.array([[np.e**3]])]
    phase = np.empty(2, dtype=object)
    phase[:] = [np.array([0, 1]), np.array([0])]
    post_phase = np.empty(2, dtype=object)
    post_phase[:] = [np.array([1, 1]), np.array([1])]
    path = tmp_path / "tray.npz"
    np.savez(
        path, cost_weights_keys=["position_pick"], cost_weights_lin=values,
        cost_weights=np.empty(0), task_flag_pre=phase, task_flag=post_phase,
    )
    # The module namespace is also the functions' global namespace.
    helpers["BASE"] = tmp_path
    names, result = helpers["load_cost_weight_trajectories"]([path.name], "Tray Push")
    assert names == ["position_pick"]
    np.testing.assert_allclose(result[:, :, 0], [[1, 0], [3, np.nan]], equal_nan=True)

