from functools import partial

from agents.agent_factory import AgentFactory
from config import get_config
from util.certification import bisect_verified_level
from util.dreal import (
    dreal_clip_vector,
    dreal_var,
    in_box,
    is_unsat,
    on_boundary,
    quadratic_form,
    require_dreal,
    squared_norm,
)


def check_lqr_level(
    agent,
    *,
    level,
    dynamics_fn_dreal,
    bounds,
    scale=2.0,
    delta=1e-4,
    margin=1e-4,
):
    """Verify the saturated LQR with a full quadratic decrease condition."""
    d = require_dreal()
    x = dreal_var(agent.state_dim)
    value = quadratic_form(x, agent.P_np)
    raw_action = [
        -sum(agent.K_np[i, j] * x[j] for j in range(agent.state_dim))
        for i in range(agent.action_dim)
    ]
    action = dreal_clip_vector(raw_action, -agent.max_action, agent.max_action)
    vector_field = dynamics_fn_dreal(x, action)
    derivative = sum(
        vector_field[i] * value.Differentiate(x[i]) for i in range(agent.state_dim)
    )
    norm_sq = squared_norm(x)
    lb, ub = bounds
    violation = d.And(
        in_box(x, lb, ub, scale),
        value <= level,
        derivative + margin * norm_sq > 0,
    )
    violation_result = d.CheckSatisfiability(violation, delta)
    if not is_unsat(violation_result):
        return False, violation_result

    boundary_contact = d.And(on_boundary(x, lb, ub, scale), value <= level)
    boundary_result = d.CheckSatisfiability(boundary_contact, delta)
    return (True, None) if is_unsat(boundary_result) else (False, boundary_result)


def certify_agent(
    agent_name,
    *,
    checkpoint=None,
    episode="latest",
    device="auto",
    beta=None,
    level_lower=0.01,
    level_upper=0.95,
    tolerance=0.001,
    max_iterations=40,
    eps=0.25,
    bridge_level=None,
    local_margin=1e-4,
    scale=2.0,
    delta=1e-4,
):
    config = get_config(agent_name)
    config["device"] = device
    if beta is not None:
        config["beta"] = float(beta)
    agent = AgentFactory.create_agent(config)
    if agent_name != "lqr":
        if checkpoint is None:
            raise ValueError("A checkpoint directory is required")
        agent.load(checkpoint, episode=episode)

    if agent_name == "ldp":
        verifier = partial(
            agent.trainer.check_lyapunov_with_ce,
            scale=scale,
            eps=eps,
            bridge_level=bridge_level,
            local_margin=local_margin,
            delta=delta,
        )
        scope = "piecewise-local-global"
    elif agent_name == "lac":
        verifier = partial(
            agent.trainer.check_lyapunov_with_ce,
            scale=scale,
            eps=eps,
            delta=delta,
        )
        scope = "annular"
    elif agent_name == "lqr":
        verifier = partial(
            check_lqr_level,
            agent,
            dynamics_fn_dreal=config["dynamics_fn_dreal"],
            bounds=config["r1_bounds"],
            scale=scale,
            delta=delta,
            margin=local_margin,
        )
        scope = "full-local-quadratic"
    else:
        raise ValueError("Certification supports ldp, lac, and lqr")

    verified_metadata = {}

    def check_fn(*, level):
        outcome = verifier(level=level)
        if outcome[0] and hasattr(agent, "trainer"):
            verified_metadata[float(level)] = dict(agent.trainer.last_verification)
        return outcome

    result = bisect_verified_level(
        check_fn,
        lower=level_lower,
        upper=level_upper,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return {
        "agent": agent_name,
        "scope": scope,
        "verified_level": result.verified_level,
        "lower_verified": result.lower_verified,
        "upper_verified": result.upper_verified,
        "checks": result.checks,
        "delta": delta,
        "scale": scale,
        "eps": eps if agent_name in {"ldp", "lac"} else None,
        "verification": verified_metadata.get(
            float(result.verified_level) if result.verified_level is not None else None
        ),
        "bridge_level": (
            verified_metadata.get(
                float(result.verified_level)
                if result.verified_level is not None
                else None,
                {},
            ).get("bridge_level", bridge_level)
            if agent_name == "ldp"
            else None
        ),
        "local_margin": (local_margin if agent_name in {"ldp", "lqr"} else None),
    }
