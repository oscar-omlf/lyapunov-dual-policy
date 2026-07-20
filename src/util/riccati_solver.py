import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are


class RiccatiSolver:
    """Validated wrappers around SciPy's Schur/QZ Riccati solvers."""

    @staticmethod
    def _symmetric(matrix):
        matrix = np.real_if_close(matrix)
        return 0.5 * (matrix + matrix.T)

    def solve_continuous_are(self, A, B, Q, R):
        solution = self._symmetric(solve_continuous_are(A, B, Q, R))
        residual = (
            A.T @ solution
            + solution @ A
            - solution @ B @ np.linalg.solve(R, B.T @ solution)
            + Q
        )
        if not np.allclose(residual, 0.0, atol=1e-7, rtol=1e-7):
            raise RuntimeError("Continuous Riccati solution failed residual check")
        return solution

    def solve_discrete_are(self, A, B, Q, R):
        solution = self._symmetric(solve_discrete_are(A, B, Q, R))
        gain_term = np.linalg.solve(R + B.T @ solution @ B, B.T @ solution @ A)
        residual = A.T @ solution @ A - solution - A.T @ solution @ B @ gain_term + Q
        if not np.allclose(residual, 0.0, atol=1e-7, rtol=1e-7):
            raise RuntimeError("Discrete Riccati solution failed residual check")
        return solution

    def solve_discounted_dare(self, A, B, Q_gamma, R_gamma, gamma):
        if not 0 < gamma <= 1:
            raise ValueError("Discount factor gamma must be in (0, 1]")
        scale = np.sqrt(gamma)
        return self.solve_discrete_are(scale * A, scale * B, Q_gamma, R_gamma)

    def compute_H_matrix(
        self,
        A,
        B,
        Q_gamma,
        R_gamma,
        P_solution_discounted_dare,
        gamma,
    ):
        if not 0 < gamma <= 1:
            raise ValueError("Discount factor gamma must be in (0, 1]")
        P = P_solution_discounted_dare
        H = np.block(
            [
                [
                    Q_gamma + gamma * A.T @ P @ A,
                    gamma * A.T @ P @ B,
                ],
                [
                    gamma * B.T @ P @ A,
                    R_gamma + gamma * B.T @ P @ B,
                ],
            ]
        )
        return self._symmetric(H)
