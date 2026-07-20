import random
from abc import ABC, abstractmethod

from util.device import fetch_device


class ReplayBuffer:
    """Fixed-capacity replay buffer with O(1) insertion and no batch copy."""

    def __init__(self, capacity: int = 1_000_000):
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive")
        self.capacity = int(capacity)
        self._storage = []
        self._size = 0
        self._next = 0

    def push(self, transition):
        if self._size < self.capacity:
            self._storage.append(transition)
            self._size += 1
        else:
            self._storage[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def clear(self):
        self._storage.clear()
        self._size = 0
        self._next = 0

    def get_buffer_list(self):
        if self._size < self.capacity:
            return self._storage[: self._size]
        return self._storage[self._next :] + self._storage[: self._next]

    def sample(self, batch_size: int):
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} items from {self._size}")
        indices = random.sample(range(self._size), batch_size)
        return [self._storage[index] for index in indices]

    def __len__(self):
        return self._size


class AbstractAgent(ABC):
    def __init__(self, config: dict):
        self.state_space = config.get("state_space")
        self.action_space = config.get("action_space")
        if self.state_space is None or self.action_space is None:
            raise ValueError("state_space and action_space are required")
        self.state_dim = self.state_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.device = fetch_device(config.get("device", "auto"))
        self._replay_buffer = ReplayBuffer(config.get("replay_capacity", 1_000_000))

    @abstractmethod
    def add_transition(self, transition):
        """
        Abstract method to add a transition to the agent's replay buffer.

        :param transition: transition to add to the replay buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """
        Abstract method where the update rule is applied.
        """
        raise NotImplementedError

    @abstractmethod
    def policy(self, state):
        """
        Abstract method to define the agent's policy.
        For actor critic algorithms the output of the actor would be a probability distribution over actions.
        For discrete actions this is simply a discrete probability distributions, describing a probability
        for each action.
        For continuous actions you can have some kind of continuous distribution you sample actions from.

        :param state: The current state of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: str = "./") -> None:
        """
        Abstract method to save the agent's model.

        :param file_path: The path to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path: str = "./") -> None:
        """
        Abstract method to load the agent's model.

        :param file_path: The path to load the model from.
        """
        raise NotImplementedError
