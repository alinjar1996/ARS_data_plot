import ast
from pathlib import Path

import pytest
import numpy as np

from make_ars_video_snapshots import (
    Episode, config_value, phase_frame_indices, make_montage, read_sources,
    resolve_video, select_episode,
)


def episode(index, steps, success=True, seed=0):
    return Episode("ball", 50, Path("run/npz/ars_batch50/a.npz"),
                   index, seed, steps, 0.1, success)


def test_notebook_sources_match_current_small_batches():
    notebook = Path(__file__).resolve().parents[1] / "benchmark_stat_batch_sm.ipynb"
    sources, batches, count = read_sources(notebook)
    assert batches == [50, 150, 250]
    assert count == 5
    assert sources["Ball Lift"]["ARS"][50].endswith("/ars_batch50/*.npz")
    assert sources["Box Lift"]["ARS"][150].endswith("/ars4218_batch150/*.npz")
    assert sources["Tray Push"]["ARS"][250].endswith("/ars4124_batch250/*.npz")


def test_configuration_reader_rejects_calls():
    node = ast.parse("__import__('os').system('false')", mode="eval").body
    with pytest.raises(ValueError, match="Unsupported"):
        config_value(node, {})


def test_select_median_success_includes_episode_zero_and_excludes_failure():
    episodes = [episode(0, 100), episode(1, 300), episode(2, 500), episode(3, 10, False)]
    assert select_episode(episodes).index == 1
    assert select_episode(episodes, override=(0, 1)).index == 0
    with pytest.raises(ValueError, match="Expected one"):
        select_episode(episodes, override=(0, 4))
    with pytest.raises(ValueError, match="No recorded-success"):
        select_episode([episode(0, 100, False)])


def test_phase_frames_use_post_step_transition_and_terminal_success():
    indices = phase_frame_indices([0] * 64 + [1] * 22)
    assert indices == [32, 64, 85]
    assert (indices[1] + 1) * 0.1 == 6.5
    assert (indices[-1] + 1) * 0.1 == 8.6


@pytest.mark.parametrize("flags", [[0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, np.nan, 1]])
def test_phase_frames_reject_missing_or_invalid_transition(flags):
    with pytest.raises(ValueError):
        phase_frame_indices(flags)


@pytest.mark.parametrize("columns", [1, 3])
def test_montage_has_only_original_pixels_and_white_gutters(columns):
    from PIL import Image

    frames = [Image.new("RGB", (20, 10), color) for color in ("red", "green", "blue")]
    montage = make_montage(frames, columns)
    expected = Image.new("RGB", (84, 10) if columns == 3 else (20, 54), "white")
    for i, frame in enumerate(frames):
        expected.paste(frame, (i * 32, 0) if columns == 3 else (0, i * 22))
    assert montage.size == expected.size
    assert montage.tobytes() == expected.tobytes()


def test_box_video_uses_zero_based_index_and_success_suffix(tmp_path):
    ep = Episode("box", 50, tmp_path / "npz/ars4218_batch50/a.npz", 0, 4, 100, .1, True)
    video = tmp_path / "videos/ars4218_batch50/run1/ars4218_batch50_seed4_ep0000_success.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert resolve_video(ep) == video
    duplicate = video.parent.parent / "run2" / video.name
    duplicate.parent.mkdir()
    duplicate.touch()
    with pytest.raises(ValueError, match="ONE matching video"):
        resolve_video(ep)


def test_ball_saved_absolute_path_is_rebased_to_selected_root(tmp_path):
    tail = "videos/ars/policy_batch50/run1/ep001_success.mp4"
    video = tmp_path / tail
    video.parent.mkdir(parents=True)
    video.touch()
    ep = Episode("ball", 50, tmp_path / "npz/ars_batch50/a.npz", 0, 0, 100, .1, True,
                 "/old/machine/eval_run/" + tail)
    assert resolve_video(ep) == video


def test_tray_video_matches_seed_and_zero_based_episode(tmp_path):
    video = tmp_path / "videos/ars4124_batch150/run1/ep_0003_seed27.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    ep = Episode("tray", 150, tmp_path / "npz/ars4124_batch150/a.npz", 3, 27, 100, .1, True)
    assert resolve_video(ep) == video
