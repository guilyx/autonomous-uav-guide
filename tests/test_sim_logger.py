"""Tests for simulation trace logging."""

import json

from flybots.logging import SimLogger


def test_saved_trace_describes_downsampling(tmp_path):
    """A consumer can recover how the stored samples relate to source steps."""
    logger = SimLogger("downsampled", out_dir=tmp_path, downsample=3)

    for step in range(8):
        logger.log_step(step=step)

    with logger.save().open() as fh:
        payload = json.load(fh)

    assert payload["timeseries"]["step"] == [0, 3, 6]
    assert payload["trace"] == {
        "source_steps": 8,
        "recorded_steps": 3,
        "downsample": 3,
    }


def test_empty_trace_has_zero_step_counts(tmp_path):
    """Metadata-only simulations still emit an unambiguous trace summary."""
    logger = SimLogger("metadata_only", out_dir=tmp_path)

    with logger.save().open() as fh:
        payload = json.load(fh)

    assert payload["trace"] == {
        "source_steps": 0,
        "recorded_steps": 0,
        "downsample": 1,
    }


def test_log_file_ends_with_a_newline(tmp_path):
    """CI's end-of-file hook rewrites the log otherwise.

    `json.dump` writes no trailing newline, so every regenerated log needed
    fixing by the pre-commit hook. That is invisible locally -- the hook
    edits the file and reports success -- and fails CI, where nothing is
    allowed to change.
    """
    logger = SimLogger("newline_probe", out_dir=tmp_path)
    logger.log_metadata("algorithm", "probe")
    logger.log_step(t=0.0, value=1.0)
    logger.log_summary("done", True)
    path = logger.save()

    assert path.read_bytes().endswith(b"\n")
