# Literature Map

## Foundations

Zubov's method characterizes the domain of attraction through a bounded
Lyapunov function satisfying a first-order PDE. Vannelli and Vidyasagar connect
maximal Lyapunov functions and domains of attraction; Khalil and Vidyasagar
provide the nonlinear-stability background used by the project.

The formal backend is dReal, a delta-complete SMT solver for nonlinear real
arithmetic:

- Gao, Kong, and Clarke, *dReal: An SMT Solver for Nonlinear Theories over the
  Reals*, CADE 2013.
- [dReal project](https://dreal.github.io/)

## Neural Lyapunov Control

Chang, Roohi, and Gao's *Neural Lyapunov Control* (NeurIPS 2019) established
the learner/falsifier pattern used here: train neural controller/certificate
candidates, translate them to dReal, and search for violating states.

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/2647c1dba23bc0e0f9cdf75339e120d2-Paper.pdf)

Wang and Fazlyab's *Actor-Critic Physics-Informed Neural Lyapunov Control*
(L-CSS 2024) supplies the LAC baseline: finite-trajectory Zubov targets, a PDE
residual, boundary samples, and concurrent policy improvement.

- [Paper](https://doi.org/10.1109/LCSYS.2024.3416235)
- [Reference code](https://github.com/Verified-Intelligence/ZubovControl)

The repository follows its small pendulum networks and physics-informed loss,
while making finite-horizon labels conservative and separating continuous
feedback from held-action integration.

## Dual Policies and TD3

Zoboli et al., *Reinforcement Learning Policies with Local LQR Guarantees for
Nonlinear Discrete-Time Systems* (CDC 2021), motivate blending a trusted local
policy/value with learned global counterparts. The orders of the blending
functions preserve the local controller derivative and local value Hessian.

- [Paper](https://doi.org/10.1109/CDC45484.2021.9683721)

Fujimoto, van Hoof, and Meger's TD3 provides the model-free baseline and the
global component of LAS-TD3. Twin critics, clipped target-policy noise, delayed
actor updates, and Polyak target updates are retained.

- [Paper](https://proceedings.mlr.press/v80/fujimoto18a.html)
- [Reference code](https://github.com/sfujim/TD3)

LDP transfers the dual-policy idea from a discounted action-value critic to a
continuous-time Zubov candidate. The key additional obligation is to certify
the *blended* saturated dynamics, not the global policy or neural candidate in
isolation.

## Later Directions

Several post-thesis directions are relevant but are not implemented here:

- **Architecturally stable neural controllers** encode local linear behavior
  and Lyapunov structure directly, reducing reliance on sampled local losses.
- **Complete branch-and-bound verification (CT-BaB)** replaces or complements
  nonlinear SMT with neural-network-specific bounds and can scale more
  effectively for piecewise-linear networks.
- **Two-stage Zubov methods** first learn a value/certificate target and then
  optimize a controller against it, reducing the moving-target coupling in
  simultaneous actor-critic training.
- **Discrete-time neural Lyapunov control (`nlc_discrete`)** treats the sampled
  transition map directly. This is the appropriate formal model when actions
  are held between samples.
- **Polar/radial Lyapunov parameterizations (PolarNet)** exploit geometry to
  parameterize star-convex level sets and can improve domain-of-attraction
  learning.

These methods suggest three concrete extensions: certify a sampled-data LDP
transition map, compare dReal with branch-and-bound verification, and replace
the unconstrained global critic with a geometry-aware candidate.

## Claim Boundary

The current result is a model-conditional continuous-time certificate on a
bounded region. It is not a robustness certificate for parameter uncertainty,
delay, observation error, state projection, or sampled-data deployment.
Standalone LAC certification excludes an epsilon ball. LDP closes that local
gap with a separately verified quadratic bridge.
