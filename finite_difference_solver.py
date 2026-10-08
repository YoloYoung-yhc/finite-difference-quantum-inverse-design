"""Solve the 1D Schrodinger equation with second-order finite differences."""

import numpy as np


def solve_schrodinger(x, potential, mass=1.0, hbar=1.0):
    """Solve the stationary equation with zero Dirichlet endpoints.

    The dense eigensolver returns all finite-domain states, not just bound states.
    For infinite-domain problems, check grid and truncation convergence separately.
    Eigenvector signs are arbitrary; near-degenerate states may mix numerically.

    Args:
        x: Real, increasing uniform grid of shape (N,), including both endpoints.
        potential: Finite real potential values of shape (N,).
        mass: Positive finite particle mass.
        hbar: Positive finite reduced Planck constant.

    Returns:
        A tuple (energies, wavefunctions). Energies have shape (N-2,) in
        ascending order. Wavefunctions have shape (N, N-2), with zero
        endpoints and sum(abs(psi)**2) * dx = 1 for each column.

    Raises:
        ValueError: If the grid, potential, or physical parameters are invalid.
    """
    if np.iscomplexobj(x) or np.iscomplexobj(potential):
        raise ValueError("x and potential must be real.")
    x = np.asarray(x, dtype=float)
    V = np.asarray(potential, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a one-dimensional grid with at least 3 points.")
    if V.shape != x.shape:
        raise ValueError("potential must have the same shape as x.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(V)):
        raise ValueError("x and potential must contain only finite values.")
    if (np.ndim(mass) != 0 or np.ndim(hbar) != 0
            or np.iscomplexobj(mass) or np.iscomplexobj(hbar)):
        raise ValueError("mass and hbar must be real scalars.")
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("mass and hbar must be positive finite numbers.")
    steps = np.diff(x)
    dx = steps[0]
    if np.any(steps <= 0) or not np.allclose(steps, dx, rtol=1e-8, atol=0.0):
        raise ValueError("x must be strictly increasing and uniformly spaced.")

    # Construct the tridiagonal Hamiltonian with the central difference stencil.
    m = mass
    N = len(x)
    t = -hbar**2 / (2 * m * dx**2)
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = V[i] - 2 * t
        if i > 0:
            H[i, i-1] = t
        if i < N - 1:
            H[i, i+1] = t

    # Fix endpoint values to zero and solve for the N-2 interior values.
    H_inner = H[1:-1, 1:-1]
    eigenvalues, eigenvectors = np.linalg.eigh(H_inner)

    # Normalize each state in physical units and restore zero endpoints.
    wavefunctions = np.zeros((N, N-2))
    for n in range(N-2):
        psi = eigenvectors[:, n].copy()
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
        wavefunctions[1:-1, n] = psi
    return eigenvalues, wavefunctions
