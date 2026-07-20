import json

import numpy as np

from significance_testing import _paired_tests


def test_identical_paired_samples_produce_valid_json():
    result = _paired_tests(
        {
            "first": np.zeros(3),
            "second": np.zeros(3),
        }
    )
    comparison = result["first vs second"]
    assert comparison["paired_t"] == 0.0
    assert comparison["p_raw"] == 1.0
    assert comparison["p_holm"] == 1.0
    json.dumps(result, allow_nan=False)
