import pickle

import numpy as np
import pytest

from pi0_ur5e.dataset_reader import DatasetReader


def test_dataset_reader_reads_fake_episode(tmp_path):
    ep = tmp_path / "episode_000"
    ep.mkdir()
    for i in range(3):
        frame = {
            "base_rgb": np.zeros((8, 8, 3), dtype=np.uint8) + i,
            "wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "ee_pos_quat": np.array([i, 0, 0, 0, 0, 0], dtype=np.float32),
            "joint_positions": np.array([i, 0, 0, 0, 0, 0, float(i > 1)], dtype=np.float32),
            "control": np.array([i + 10, 0, 0, 0, 0, 0, 0.5], dtype=np.float32),
            "gripper_position": np.array([float(i > 1)], dtype=np.float32),
            "language_instruction": "test prompt",
        }
        with open(ep / f"{i}.pkl", "wb") as f:
            pickle.dump(frame, f)
    episodes = DatasetReader(tmp_path).episodes()
    assert len(episodes) == 1
    out = episodes[0]
    assert out.base_rgb.shape == (3, 8, 8, 3)
    assert out.wrist_rgb.shape == (3, 8, 8, 3)
    assert out.robot_state.shape == (3, 7)
    assert np.allclose(out.robot_state[:, -1], [0.0, 0.0, 1.0])
    assert out.action.shape == (3, 7)
    assert np.allclose(out.action[:, 0], [10.0, 11.0, 12.0])
    assert out.language_instruction == "test prompt"


def test_dataset_reader_builds_tactile_image_embeddings(tmp_path):
    ep = tmp_path / "episode_000"
    ep.mkdir()
    for i in range(2):
        frame = {
            "base_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "ee_pos_quat": np.array([i, 0, 0, 0, 0, 0], dtype=np.float32),
            "joint_positions": np.array([i, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "gripper_position": np.array([0.0], dtype=np.float32),
            "tactile_left_rgb": np.zeros((12, 12, 3), dtype=np.uint8) + i,
            "tactile_right_rgb": np.ones((12, 12, 3), dtype=np.uint8) * 32,
        }
        with open(ep / f"{i}.pkl", "wb") as f:
            pickle.dump(frame, f)

    episodes = DatasetReader(
        tmp_path,
        config={
            "include_tactile": True,
            "tactile_feature_mode": "image_embedding",
            "tactile_embedding_dim": 64,
        },
    ).episodes()

    out = episodes[0]
    assert out.robot_state.shape == (2, 7)
    assert out.tactile.shape == (2, 64)
    assert out.tactile.dtype == np.float32
    assert np.isfinite(out.tactile).all()


def test_dataset_reader_reports_invalid_episode_path(tmp_path):
    ep = tmp_path / "episode_bad"
    ep.mkdir()
    (ep / "trajectory.h5").write_text("not an hdf5 file", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"episode_bad/trajectory\.h5: failed to read episode"):
        DatasetReader(tmp_path).episodes()
