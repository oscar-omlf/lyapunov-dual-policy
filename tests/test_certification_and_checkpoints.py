import pytest

from util.certification import bisect_verified_level
from util.checkpoints import validate_checkpoint


def test_bisection_checks_both_endpoints_and_returns_verified_lower():
    checked = []

    def check(*, level):
        checked.append(level)
        return level <= 0.6, None if level <= 0.6 else {"x": level}

    result = bisect_verified_level(
        check, lower=0.1, upper=0.9, tolerance=1e-3, max_iterations=20
    )
    assert checked[:2] == [0.1, 0.9]
    assert result.lower_verified
    assert not result.upper_verified
    assert result.verified_level <= 0.6
    assert 0.6 - result.verified_level <= 1e-3


def test_bisection_rejects_unverified_lower_endpoint():
    result = bisect_verified_level(
        lambda *, level: (False, level), lower=0.1, upper=0.9
    )
    assert result.verified_level is None
    assert not result.lower_verified
    assert result.checks == 1


def test_checkpoint_blending_metadata_is_strict():
    checkpoint = {"agent": "ldp", "beta": 0.9, "c_star": 1.2}
    validate_checkpoint(checkpoint, expected_agent="ldp", beta=0.9, c_star=1.2)
    with pytest.raises(ValueError, match="beta"):
        validate_checkpoint(checkpoint, expected_agent="ldp", beta=0.8, c_star=1.2)
    with pytest.raises(ValueError, match="expected"):
        validate_checkpoint(checkpoint, expected_agent="las-td3")
