def rk4_step(f, x, u, dt):
    """Advance dynamics with a zero-order-held control input.

    This is the appropriate transition model for discrete-time RL algorithms:
    ``u`` is computed once and held fixed across all four RK4 stages.
    """
    f1 = f(x, u)
    f2 = f(x + 0.5 * dt * f1, u)
    f3 = f(x + 0.5 * dt * f2, u)
    f4 = f(x + dt * f3, u)
    return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


def closed_loop_rk4_step(f, policy, x, dt):
    """Advance continuous closed-loop dynamics ``f(x, policy(x))`` with RK4."""
    f1 = f(x, policy(x))
    x2 = x + 0.5 * dt * f1
    f2 = f(x2, policy(x2))
    x3 = x + 0.5 * dt * f2
    f3 = f(x3, policy(x3))
    x4 = x + dt * f3
    f4 = f(x4, policy(x4))
    return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
