"""No physics reruns: check phase gates, time alignment and plotting inputs."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from make_ars_video_snapshots import DEFAULT_NOTEBOOK, Episode, TASKS
from make_ars_weights_snapshots import (
    DEFAULT_SELECTIONS, FOCUS_WEIGHTS, box_settled_start, effective_weights,
    goal_distance, masked_weights, phase_changes, plotted_policy_weights,
    read_weight_config, render_figure, snapshot_times, write_csv,
)


@pytest.fixture(scope="module")
def config():
    return read_weight_config(DEFAULT_NOTEBOOK)


@pytest.fixture(scope="module")
def notebook_mask():
    """Compare with the notebook's actual existing helper, not a copied formula."""
    notebook = json.loads(DEFAULT_NOTEBOOK.read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "def phase_masked_log_weights(" in "".join(c["source"]))
    nodes = [n for n in ast.parse(cell).body if
             (isinstance(n, ast.FunctionDef) and n.name == "phase_masked_log_weights") or
             (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {"PAPER_COST_LABELS", "PHASE_WEIGHT_RULES"})]
    env = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-mask", "exec"), env)
    return env["phase_masked_log_weights"]


@pytest.mark.parametrize("task", TASKS.values())
def test_masks_match_notebook_including_repeated_switches(config, notebook_mask, task):
    labels, rules = config
    names = list(labels[task])
    phase = np.array([0, 1, 1, 0, 1])
    raw = np.exp(np.random.default_rng(10).uniform(-4, 4, (5, len(names))))
    original = raw.copy()
    logs, active, multiplier = masked_weights(raw, phase, names, task, labels, rules)
    np.testing.assert_allclose(logs, notebook_mask(raw, phase, names, task))
    np.testing.assert_array_equal(raw, original)
    assert np.all(logs[~active] == 0)
    effective = effective_weights(raw, active, multiplier)
    np.testing.assert_array_equal(effective, np.where(active, raw * multiplier, 0))
    np.testing.assert_allclose(np.log(effective[active]), logs[active])
    assert np.all(effective[~active] == 0)
    if task != "Tray Push":
        np.testing.assert_array_equal(multiplier[:, names.index("distance")], [1, 10, 10, 1, 10])


def test_box_unused_smoothness_is_zero_even_if_raw_is_large(config):
    labels, rules = config
    logs, active, _ = masked_weights([[1e4], [2e4]], [0, 1], ["smoothness"], "Box Lift", labels, rules)
    assert not np.any(active)
    assert np.all(logs == 0)


def test_inactive_invalid_values_are_not_logged(config):
    labels, rules = config
    values = [[np.nan], [0], [np.inf], [-2]]
    logs, _, _ = masked_weights(values, [1]*4, ["eef_to_obj"], "Ball Lift", labels, rules)
    assert np.all(logs == 0)
    with pytest.raises(ValueError, match="invalid active"):
        masked_weights(values, [0]*4, ["eef_to_obj"], "Ball Lift", labels, rules)


def test_effective_weights_ignore_invalid_inactive_outputs():
    raw = np.array([[np.nan, 3.], [np.inf, 4.]])
    np.testing.assert_array_equal(effective_weights(raw, [[False, True]]*2, [[1, 10]]*2), [[0, 30], [0, 40]])
    with pytest.raises(ValueError, match="same shape"):
        effective_weights(raw, [True], [1])
    with pytest.raises(ValueError, match="Invalid effective"):
        effective_weights(raw, np.ones_like(raw, bool), np.ones_like(raw))


@pytest.mark.parametrize("task", TASKS.values())
def test_policy_plots_do_not_apply_fixed_planner_gains(config, task):
    labels, rules = config
    names = list(labels[task])
    raw = np.full((2, len(names)), 3.)
    _, active, gains = masked_weights(raw, [0, 1], names, task, labels, rules)
    plotted = plotted_policy_weights(raw, active)
    np.testing.assert_array_equal(plotted, np.where(active, raw, 0))
    assert plotted[1, names.index('distance')] == 3.
    if task != 'Tray Push':
        assert effective_weights(raw, active, gains)[1, names.index('distance')] == 30.
    if task == 'Box Lift':
        assert not plotted[:, names.index('smoothness')].any()


@pytest.mark.parametrize("phase", [[0], [0, 2], [0, np.nan], [[0], [1]]])
def test_invalid_phase_rejected(config, phase):
    with pytest.raises(ValueError, match="phase"):
        masked_weights([[2], [3]], phase, ["theta"], "Ball Lift", *config)


def test_unmapped_cost_rejected(config):
    with pytest.raises(ValueError, match="Unknown"):
        masked_weights([[2]], [0], ["not_a_cost"], "Ball Lift", *config)


def test_pre_weight_and_post_frame_have_correct_one_step_offset():
    pre = np.array([0]*6 + [1]*4)
    post = np.array([0]*5 + [1]*5)
    indices, edges, times = snapshot_times(pre, post, .1)
    assert indices == [0, 2, 5, 9]
    np.testing.assert_allclose(times, [.1, .3, .6, 1])
    assert edges[6] == times[2]  # First lift solve starts at the transition frame.
    with pytest.raises(ValueError, match="phase-aligned"):
        snapshot_times(pre, pre, .1)


def test_final_step_transition_cannot_supply_four_stages():
    with pytest.raises(ValueError, match="separate goal-reaching"):
        snapshot_times([0]*10, [0]*9 + [1], .1)


def test_start_and_approach_must_be_distinct():
    with pytest.raises(ValueError, match="four distinct"):
        snapshot_times([0, 0, 1, 1], [0, 1, 1, 1], .1)


def test_settled_start_repositions_approach_without_rebasing_time():
    indices, edges, times = snapshot_times([0]*6 + [1]*4, [0]*5 + [1]*5, .1, start=1)
    assert indices == [1, 3, 5, 9]
    np.testing.assert_allclose(times, [.2, .4, .6, 1.])
    assert edges[0] == 0
    with pytest.raises(ValueError, match="Start must"):
        snapshot_times([0]*6 + [1]*4, [0]*5 + [1]*5, .1, start=5)


@pytest.fixture
def settling_states():
    qpos, qvel = np.zeros((20, 19)), np.zeros((20, 18))
    qpos[:, 14] = [.3, .2, .05, .08, .09, .095, .097, .099, .1] + [.1]*11
    qvel[:, 14] = [-1., -1., -1., .3, .1, .04, .02, .008, .004, .003, .002, .001] + [0.]*8
    return qpos, qvel, np.zeros(20)


def test_box_start_is_after_rebound_and_sustained_settling(settling_states):
    frame, info = box_settled_start(*settling_states, .1)
    assert frame == 9
    assert info['start_time_s'] == 1.
    assert info['initial_rebound_frame'] == 3
    assert info['observed_max_speed_m_s'] <= .005
    assert info['observed_height_range_m'] <= .001


def test_box_start_rejects_unknown_landing_or_unsettled_approach(settling_states):
    qpos, qvel, post = settling_states
    with pytest.raises(ValueError, match="landing/rebound"):
        box_settled_start(qpos, np.zeros_like(qvel), post, .1)
    post[8:] = 1
    with pytest.raises(ValueError, match="No settled"):
        box_settled_start(qpos, qvel, post, .1)
    with pytest.raises(ValueError, match="Invalid Box"):
        box_settled_start(qpos, qvel[:, :12], post, .1)


@pytest.mark.parametrize("dt", [0, -.1, np.nan])
def test_snapshot_time_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match="Invalid"):
        snapshot_times([0]*6 + [1]*4, [0]*5 + [1]*5, dt)


def test_goal_error_uses_post_state_not_stale_box_diagnostic():
    qpos = np.zeros((3, 19)); qpos[:, 12] = [3, 2, 1]
    data = {"qpos_post": [qpos], "target": [np.zeros((3, 7))], "cost_obj_to_targ": [[99]*3]}
    values, source = goal_distance(data, "box", 0)
    np.testing.assert_allclose(values, [3, 2, 1])
    assert "qpos_post" in source
    data = {"cost_g_tray": [[.3, .2, .1]]}
    np.testing.assert_allclose(goal_distance(data, "tray", 0)[0], [.3, .2, .1])


def test_notebook_literal_reader_does_not_execute_code(tmp_path):
    path = tmp_path / "fake.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [
        "PAPER_COST_LABELS = {}\nPHASE_WEIGHT_RULES = __import__('os').system('false')\n"
    ]}]}))
    with pytest.raises(ValueError):
        read_weight_config(path)


def test_pinned_choices_and_focus_components_are_explicit(config):
    assert DEFAULT_SELECTIONS == {"ball": (50, 4, 18), "box": (50, 5, 5), "tray": (250, 27, 4)}
    labels, _ = config
    for task, names in FOCUS_WEIGHTS.items():
        assert len(names) == 4 and set(names) <= labels[TASKS[task]].keys()


@pytest.mark.parametrize("task", TASKS)
def test_all_linear_panels_and_four_unannotated_photographs(config, tmp_path, task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    labels, rules = config
    names = list(labels[TASKS[task]])
    phase = np.array([0]*6 + [1]*4); post = np.array([0]*5 + [1]*5)
    indices, edges, times = snapshot_times(phase, post, .1, start=1 if task == "box" else 0)
    raw = np.full((10, len(names)), np.e)
    logs, active, multiplier = masked_weights(raw, phase, names, TASKS[task], labels, rules)
    effective = effective_weights(raw, active, multiplier)
    plotted = plotted_policy_weights(raw, active)
    trace = dict(names=names, linear=raw, masked_log=logs, effective=effective,
                 plotted_linear=plotted, active=active, multiplier=multiplier,
                 edges=edges, indices=indices, times=times, goal_distance=np.linspace(.3, .03, 10),
                 pre_phase=phase, post_phase=post, plot_start_s=times[0] if task == "box" else 0.)
    ep = Episode(task, 50, Path("a.npz"), 0, 0, 10, .1, True)
    frames = [Image.new("RGB", (64, 48), color) for color in ["red", "green", "blue", "yellow"]]
    fig = render_figure(ep, trace, frames, labels)
    try:
        assert len(fig.axes) == 4 + len(names)
        for ax, frame in zip(fig.axes[:4], frames):
            np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), np.asarray(frame))
            assert not ax.texts  # Titles are separate from the image; no overlays.
            assert ax.get_position().y0 > fig.axes[4].get_position().y1
        np.testing.assert_allclose([ax.get_position().y0 for ax in fig.axes[:4]], fig.axes[0].get_position().y0)
        for column, weight_ax in enumerate(fig.axes[4:]):
            stairs = [p for p in weight_ax.patches if hasattr(p, 'get_data')]
            assert len(stairs) == 1
            assert stairs[0].get_data().baseline is None
            np.testing.assert_array_equal(stairs[0].get_data().values, plotted[:, column])
            assert '×10' not in weight_ax.get_title()
            assert weight_ax.get_yscale() == 'linear'
            assert weight_ax.get_ylim()[0] <= 0
            assert weight_ax.get_xlim()[0] == trace['plot_start_s']
            for tag, t in zip("ABCD", times):
                marker = next(text for text in weight_ax.texts if text.get_text() == tag)
                assert marker.get_position()[0] == t
                assert marker.get_position()[1] < 1  # Separate from external phase banner.
            for name in ['Approach', 'Push' if task == 'tray' else 'Lift']:
                banner = next(text for text in weight_ax.texts if text.get_text() == name)
                assert banner.get_position()[1] > 1  # No vertical line through phase text.
        fig.savefig(tmp_path / "smoke.png")
        write_csv(tmp_path / "trace.csv", trace, .1)
        assert f"masked_ln_{names[-1]}" in (tmp_path / "trace.csv").read_text()
        assert f"effective_linear_{names[-1]}" in (tmp_path / "trace.csv").read_text()
        assert f"plotted_policy_linear_{names[-1]}" in (tmp_path / "trace.csv").read_text()
        assert len((tmp_path / "trace.csv").read_text().splitlines()) == 11  # No loss of initial data.
        changes = phase_changes(trace)
        assert len(changes) == len(names)
        assert changes[names[0]]['transport']['within_phase_actor_last_over_first'] == 1
        if task != 'tray':
            jump = changes['distance']['at_phase_boundary']
            assert jump['multiplier_before'] == 1
            assert jump['multiplier_after'] == 10
            assert jump['raw_before'] == jump['raw_after']  # Programmed jump is not learned.
            assert jump['plotted_before'] == jump['plotted_after']
            assert changes['distance']['transport']['plotted_first'] == np.e
            assert changes['distance']['transport']['effective_first'] == np.e * 10
        if task == 'box':
            assert changes['smoothness']['transport']['effective_max'] == 0
            assert changes[names[0]]['approach']['first_solve_time_s'] == trace['plot_start_s']
    finally:
        plt.close(fig)
