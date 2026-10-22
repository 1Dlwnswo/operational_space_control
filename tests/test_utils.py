import numpy as np
import pytest

from src.utils import create_experiment_dir, wrap_to_pi


def test_wrap_to_pi_returns_angles_in_expected_interval():
    wrapped = wrap_to_pi(np.array([-3 * np.pi, -np.pi / 2, 3 * np.pi]))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi / 2, -np.pi])


def test_create_experiment_dir_creates_timestamped_directory(tmp_path, monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            class FixedNow:
                @staticmethod
                def strftime(_format):
                    return "2026-10-22_120000"

            return FixedNow()

    monkeypatch.setattr("src.utils.datetime", FixedDateTime)

    experiment_dir = create_experiment_dir("line", tmp_path)

    assert experiment_dir == tmp_path / "line" / "2026-10-22_120000"
    assert experiment_dir.is_dir()


def test_create_experiment_dir_raises_if_timestamped_directory_exists(
    tmp_path, monkeypatch
):
    class FixedDateTime:
        @staticmethod
        def now():
            class FixedNow:
                @staticmethod
                def strftime(_format):
                    return "2026-10-22_120000"

            return FixedNow()

    monkeypatch.setattr("src.utils.datetime", FixedDateTime)
    create_experiment_dir("line", tmp_path)

    with pytest.raises(FileExistsError):
        create_experiment_dir("line", tmp_path)
