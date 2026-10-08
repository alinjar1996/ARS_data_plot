"""Task-description rows contain real frames, validated phases and no labels."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from make_task_description_snapshots import crop_image, make_row, quadrotor_snapshot, transport_frame


@pytest.mark.parametrize('fraction, expected', [(0, 3), (.5, 4), (1, 5)])
def test_transport_selection_excludes_transition_step(fraction, expected):
    assert transport_frame([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], fraction) == expected


@pytest.mark.parametrize('fraction', [-.1, 1.1, np.nan, np.inf])
def test_transport_selection_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        transport_frame([0, 0, 1], [0, 1, 1], fraction)


@pytest.mark.parametrize('pre, post', [([0, 0, 1], [0, 0, 1]), ([0, 1], [1, np.nan]), ([], [])])
def test_transport_selection_rejects_invalid_timing(pre, post):
    with pytest.raises(ValueError, match='phases'):
        transport_frame(pre, post, 1.)


def test_no_transport_solve_is_not_a_lifting_snapshot():
    with pytest.raises(ValueError, match='No recorded'):
        transport_frame([0, 0, 0], [0, 0, 1], 1.)


def test_row_has_exactly_four_equal_slots_without_text_or_stretching():
    frames = [Image.new('RGB', (40, 30), c) for c in ('red', 'green', 'blue', 'yellow')]
    row, placements = make_row(frames, panel_width=40, gap=4)
    assert row.size == (172, 30)
    for i, frame in enumerate(frames):
        assert row.crop((i*44, 0, i*44+40, 30)).tobytes() == frame.tobytes()
        assert placements[i]['displayed_size_px'] == [40, 30]
        if i < 3:
            assert set(row.crop((i*44+40, 0, i*44+44, 30)).getdata()) == {(255, 255, 255)}
    frames[-1] = Image.new('RGB', (40, 20), 'yellow')
    row, placements = make_row(frames, panel_width=40, gap=4)
    assert placements[-1] == dict(panel=4, paste_xy_px=[132, 5], displayed_size_px=[40, 20])
    assert row.getpixel((150, 0)) == (255, 255, 255)
    assert row.getpixel((150, 10)) == (255, 255, 0)


def test_crop_only_removes_explicit_pixels():
    frame = Image.fromarray(np.arange(12*16*3, dtype=np.uint8).reshape(12, 16, 3))
    assert crop_image(frame, None) is frame
    cropped = crop_image(frame, (3, 2, 8, 6))
    np.testing.assert_array_equal(cropped, np.asarray(frame)[2:8, 3:11])
    for crop in [(-1, 0, 4, 4), (0, 0, 0, 1), (15, 0, 4, 4)]:
        with pytest.raises(ValueError, match='outside'):
            crop_image(frame, crop)


@pytest.fixture
def quad_source(tmp_path, monkeypatch):
    import cv2

    root = tmp_path/'data'
    run = 'ars_batch150_seed42'
    stem = 'quadrotor_150_20_30_50_00'
    video = root/'video'/'session'/run/(stem+'_SUCCESS.mp4')
    video.parent.mkdir(parents=True)
    video.write_bytes(b'fake video only decoded by the mock')
    summary = root/'trials'/'session'/run/'quad_trials_seed42.json'
    summary.parent.mkdir(parents=True)
    metadata = dict(weight_source='ARS policy (test.json)', checkpoint='test.json',
                    results=[dict(trial=0, success=True, crashed=False, steps=89)])
    summary.write_text(json.dumps(metadata))
    trajectory = root/'planner/trajectory'/'session'/run/(stem+'.npz')
    trajectory.parent.mkdir(parents=True)
    np.savez(trajectory, success=True, num_batch=150, times=np.arange(1, 89)*.05)

    class Capture:
        def __init__(self, _):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: 129, cv2.CAP_PROP_FPS: 20,
                    cv2.CAP_PROP_POS_FRAMES: self.index}[key]
        def set(self, key, index):
            assert key == cv2.CAP_PROP_POS_FRAMES
            self.index = index
        def read(self):
            self.index += 1
            return True, np.zeros((12, 16, 3), dtype=np.uint8)
        def release(self):
            pass

    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    return SimpleNamespace(video=video, summary=summary, metadata=metadata)


def test_quad_uses_summary_steps_not_video_end_card_or_shorter_diagnostic_log(quad_source):
    frame, source = quadrotor_snapshot(quad_source.video, 27, (2, 2, 8, 6))
    assert frame.size == (8, 6)
    assert source['recorded_success'] and source['method'] == 'ARS'
    assert source['frame_index_zero_based'] == 27
    assert source['simulation_time_s'] == pytest.approx(1.4)
    assert source['video_time_s'] == pytest.approx(1.35)
    assert source['trajectory_samples'] == 88
    assert source['usable_video_frames'] == 89
    assert source['excluded_terminal_frames'] == 40
    for index in [-1, 89, 128]:
        with pytest.raises(ValueError, match='status cards'):
            quadrotor_snapshot(quad_source.video, index, None)


@pytest.mark.parametrize('field, value', [('success', False), ('crashed', True)])
def test_quad_requires_recorded_success_not_just_success_in_filename(quad_source, field, value):
    quad_source.metadata['results'][0][field] = value
    quad_source.summary.write_text(json.dumps(quad_source.metadata))
    with pytest.raises(ValueError, match='successful ARS'):
        quadrotor_snapshot(quad_source.video, 27, None)


def test_quad_requires_ars_summary_not_just_ars_folder(quad_source):
    quad_source.metadata['weight_source'] = 'PPO policy (test.json)'
    quad_source.summary.write_text(json.dumps(quad_source.metadata))
    with pytest.raises(ValueError, match='successful ARS'):
        quadrotor_snapshot(quad_source.video, 27, None)
