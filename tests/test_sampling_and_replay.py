import numpy as np
import pytest
import torch

from agents.abstract_agent import ReplayBuffer
from util.sampling import sample_out_of_region, sample_out_of_region_torch


def _assert_on_box_boundary(samples, lower, upper):
    inside = (samples >= lower).all(axis=1) & (samples <= upper).all(axis=1)
    on_face = np.isclose(samples, lower).any(axis=1) | np.isclose(samples, upper).any(
        axis=1
    )
    assert inside.all()
    assert on_face.all()


def test_boundary_samplers_use_scaled_box_faces():
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    samples = sample_out_of_region(1000, lower, upper, scale=2.0)
    _assert_on_box_boundary(samples, 2 * lower, 2 * upper)

    torch_samples = sample_out_of_region_torch(
        1000,
        torch.tensor(lower),
        torch.tensor(upper),
        scale=2.0,
        device="cpu",
    )
    _assert_on_box_boundary(torch_samples.numpy(), 2 * lower, 2 * upper)


def test_replay_buffer_is_bounded_and_sampled_without_full_copy():
    buffer = ReplayBuffer(capacity=3)
    for value in range(5):
        buffer.push(value)

    assert len(buffer) == 3
    assert buffer.get_buffer_list() == [2, 3, 4]
    assert set(buffer.sample(3)) == {2, 3, 4}
    with pytest.raises(ValueError):
        buffer.sample(4)
