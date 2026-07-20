# Methodology and Implementation Contract

## Plant

The state is `x = [theta, omega]` and the saturated torque satisfies
`|u| <= 1`. The inverted-pendulum dynamics are

```text
theta_dot = omega
omega_dot = (g/l) sin(theta) - b/(m l^2) omega + u/(m l^2)
```

with `g=9.81`, `m=0.15`, `l=0.5`, and `b=0.1`. The equilibrium is the upright
state `x=0`.

`rk4_step` holds an action fixed across all RK4 stages. It is used for TD3 and
LAS-TD3 transitions. `closed_loop_rk4_step` reevaluates the policy at every
stage. It is used for continuous-feedback LAC/LDP rollouts, certificate-aligned
evaluation, and the interpretation of the continuous-time dReal model.

## Local Controller

Continuous-time LQR solves the CARE with `Q=I` and `R=I`, then applies
`u_local=clip(-Kx)`. Its quadratic candidate is `V_local=x^T P x`.

LAS-TD3 follows the thesis configuration: it discretizes the linearized plant,
constructs the discounted DARE problem from Zoboli et al., and uses a local
quadratic action value. The matrix returned by `compute_H_matrix` is a positive
cost matrix. Because TD3 maximizes reward, the implemented local critic is
`Q_local=-z^T H z`, where `z=[x,u]`. Omitting this minus sign reverses the local
actor objective.

The thesis uses identity LQR weights while its reported DRL performance metric
uses `theta^2 + 0.1 omega^2 + 0.001 u^2`. This repository retains that reported
experimental design. Consequently, LAS-TD3's local quadratic prior is a
stability prior, not the exact action value of the performance cost.

## Dual-Policy Construction

Let

```text
v(x) = V_local(x) / c_star
s = arctanh(beta),  0 < beta < 1
h1(x) = tanh(s v(x))
h2(x) = tanh(s v(x)^(3/2))
```

The policy and LDP candidate are

```text
u(x) = clip(u_local(x) + h1(x)(u_global(x)-u_local(x)))

W_local(x) = tanh(alpha V_local(x))
W(x) = W_local(x) + h2(x)(W_global(x)-W_local(x)).
```

The local action is clipped before blending and the final action is clipped
again. Runtime rollout, deployment, and symbolic dReal translation use this
same order.

Since `V_local=O(||x||^2)`, `h1=O(||x||^2)` and the global correction does not
change the policy Jacobian at the equilibrium. Likewise,
`h2=O(||x||^3)`, so the composite candidate retains the local value, gradient,
and Hessian at the origin.

## Physics-Informed Training

For running cost `phi(x)=||x||_2`, the Zubov transform
`W=tanh(alpha V)` satisfies

```text
grad(W) . f(x,u(x)) + alpha (1-W^2) phi(x) = 0.
```

The critic loss combines:

- the value at the origin;
- finite-rollout targets;
- the squared Zubov PDE residual;
- an outer-box boundary target of one.

The actor minimizes `grad(W) . f`. Its value gradient is detached, while the
vector field remains differentiable through the actor. The PDE vector field is
detached so that the PDE term updates the critic rather than duplicating the
actor objective.

Rollout labels are cached for a configurable number of optimizer steps. A
trajectory receives a finite target only if it enters the convergence ball.
Numerical divergence, exceeding the cost limit, or reaching the finite horizon
produces `W_target=1`. No stagnation heuristic is treated as convergence.
Trajectories are allocated only for plotting or explicit diagnostics.

## Certification

All solver outcomes are interpreted conservatively: only `UNSAT` proves a
property. Bisection checks both endpoints before assuming a valid monotone
bracket.

For LDP, a proposed outer level `c` is accepted only if:

1. `V_local <= c_bridge` implies strict quadratic decrease under the actual
   blended and saturated policy;
2. the local bridge cannot contact the verification-box boundary;
3. where `V_local >= c_bridge` and `W <= c`, the composite candidate is
   positive and strictly decreasing;
4. `W <= c` cannot contact the verification-box boundary.

The level-independent local bridge query is cached during each bisection and
invalidated after every optimizer update.

For LAC, the learned candidate is checked only where `||x|| >= eps`; this is an
annular statement. For LQR, the saturated controller is checked directly with
its quadratic candidate and a configurable decrease margin.

dReal is delta-complete. The configured `delta` is part of every certificate
record and should be reported with any claimed level set.

## CEGAR

Training alternates optimizer blocks and formal bisection. Failed solver models
are converted to counterexample states and kept in a bounded FIFO collection.
Subsequent physics batches combine recent counterexamples with uniform random
samples. The best verified checkpoint and certificate history are stored
separately from the latest checkpoint.

## Evaluation

Evaluation uses identical initial states for all controllers and reports
positive cumulative cost. Runs are paired, so the companion statistical script
uses paired tests and a proper Holm step-down correction.

The default continuous-feedback mode matches the certified LAC/LDP dynamics.
Sample-and-hold mode matches TD3's transition model. State projection wraps the
angle and clips angular velocity; it is useful for the benchmark MDP but
introduces a discontinuity absent from the continuous-time certificate.
