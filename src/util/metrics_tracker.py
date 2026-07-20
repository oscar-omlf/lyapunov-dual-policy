import threading
from collections import defaultdict
from pathlib import Path

import numpy as np


class MetricsTracker:
    """Collect and vectorize per-episode metrics from independent runs."""

    def __init__(self):
        self._lock = threading.RLock()
        self._returns = defaultdict(list)
        self._actor_losses = defaultdict(list)
        self._critic_losses = defaultdict(list)

    @staticmethod
    def _as_run(values):
        run = np.asarray(values, dtype=np.float64)
        if run.ndim != 1:
            raise ValueError("A metric run must be one-dimensional")
        return run.copy()

    def add_run_returns(self, agent_id, returns):
        with self._lock:
            self._returns[agent_id].append(self._as_run(returns))

    def add_run_actor_losses(self, agent_id, actor_losses):
        with self._lock:
            self._actor_losses[agent_id].append(self._as_run(actor_losses))

    def add_run_critic_losses(self, agent_id, critic_losses):
        with self._lock:
            self._critic_losses[agent_id].append(self._as_run(critic_losses))

    def add_run_losses(self, agent_id, actor_losses, critic_losses):
        self.add_run_actor_losses(agent_id, actor_losses)
        self.add_run_critic_losses(agent_id, critic_losses)

    @staticmethod
    def _stats(runs):
        if not runs:
            return np.array([], dtype=int), np.array([]), np.array([])
        width = max(map(len, runs))
        values = np.full((len(runs), width), np.nan, dtype=np.float64)
        for index, run in enumerate(runs):
            values[index, : len(run)] = run

        finite = np.isfinite(values)
        counts = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, values, 0.0).sum(axis=0),
            counts,
            out=np.full(width, np.nan),
            where=counts > 0,
        )
        centered = np.where(finite, values - means, 0.0)
        sample_variance = np.divide(
            np.square(centered).sum(axis=0),
            counts - 1,
            out=np.zeros(width),
            where=counts > 1,
        )
        standard_errors = np.divide(
            np.sqrt(sample_variance),
            np.sqrt(counts),
            out=np.zeros(width),
            where=counts > 0,
        )
        return np.arange(width), means, standard_errors

    def _get(self, store, agent_id):
        with self._lock:
            return self._stats(store.get(agent_id, []))

    def get_avg_returns(self, agent_id):
        return self._get(self._returns, agent_id)

    def get_avg_actor_losses(self, agent_id):
        return self._get(self._actor_losses, agent_id)

    def get_avg_critic_losses(self, agent_id):
        return self._get(self._critic_losses, agent_id)

    def _ranked_agents(self):
        scores = {}
        for agent_id, runs in self._returns.items():
            _, means, _ = self._stats(runs)
            finite = means[np.isfinite(means)]
            if finite.size:
                scores[agent_id] = finite[-1]
        if scores:
            return [
                agent_id
                for agent_id, _ in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ]
        return sorted(set(self._actor_losses) | set(self._critic_losses))[:10]

    @staticmethod
    def _plot_series(axis, episodes, means, errors, label):
        if episodes.size == 0:
            return
        axis.plot(episodes, means, label=label)
        axis.fill_between(episodes, means - errors, means + errors, alpha=0.2)

    def _plot_losses(self, axis, agent_ids):
        for agent_id in agent_ids:
            self._plot_series(
                axis,
                *self.get_avg_actor_losses(agent_id),
                f"{agent_id}: actor",
            )
            self._plot_series(
                axis,
                *self.get_avg_critic_losses(agent_id),
                f"{agent_id}: critic",
            )
        axis.set(xlabel="Episode", ylabel="Loss", title="Training losses")
        if axis.lines:
            axis.legend()

    def _plot_returns(self, axis, agent_ids):
        for agent_id in agent_ids:
            self._plot_series(
                axis,
                *self.get_avg_returns(agent_id),
                f"{agent_id}: return",
            )
        axis.set(xlabel="Episode", ylabel="Return", title="Training returns")
        if axis.lines:
            axis.legend()

    def plot(self):
        import matplotlib.pyplot as plt

        with self._lock:
            figure, axes = plt.subplots(2, 1, figsize=(10, 8))
            agents = sorted(
                set(self._returns) | set(self._actor_losses) | set(self._critic_losses)
            )
            self._plot_losses(axes[0], agents)
            self._plot_returns(axes[1], agents)
            figure.tight_layout()
        plt.show()

    def plot_split(self):
        import matplotlib.pyplot as plt

        agents = sorted(
            set(self._returns) | set(self._actor_losses) | set(self._critic_losses)
        )
        _, loss_axis = plt.subplots(figsize=(15, 6))
        self._plot_losses(loss_axis, agents)
        plt.tight_layout()
        _, return_axis = plt.subplots(figsize=(15, 6))
        self._plot_returns(return_axis, agents)
        plt.tight_layout()
        plt.show()

    def plot_top_10_agents(self):
        import matplotlib.pyplot as plt

        _, axis = plt.subplots(figsize=(15, 6))
        self._plot_returns(axis, self._ranked_agents())
        plt.tight_layout()
        plt.show()

    def save_top10_plots(self, folder="plots"):
        import matplotlib.pyplot as plt

        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        agents = self._ranked_agents()

        returns_figure, returns_axis = plt.subplots(figsize=(15, 6))
        self._plot_returns(returns_axis, agents)
        returns_figure.tight_layout()
        returns_figure.savefig(output_dir / "top10_returns.png")
        plt.close(returns_figure)

        loss_figure, loss_axis = plt.subplots(figsize=(15, 6))
        self._plot_losses(loss_axis, agents)
        loss_figure.tight_layout()
        loss_figure.savefig(output_dir / "top10_losses.png")
        plt.close(loss_figure)

    def save_top10_losses_plot(self, folder="plots"):
        import matplotlib.pyplot as plt

        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(15, 6))
        self._plot_losses(axis, self._ranked_agents())
        figure.tight_layout()
        figure.savefig(output_dir / "top10_losses.png")
        plt.close(figure)
