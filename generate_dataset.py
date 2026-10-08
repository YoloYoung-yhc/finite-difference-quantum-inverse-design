"""Generate Gaussian potentials and load energy regression datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset

from finite_difference_solver import solve_schrodinger


class QuantumDataset(Dataset):
    """Expose potential-energy pairs through the PyTorch Dataset interface.

    Normalization statistics must be fitted on training data only. No statistics
    are estimated here; omitting normalization preserves physical units.

    Attributes:
        X: Float32 tensor of shape (num_samples, N).
        y: Float32 tensor of shape (num_samples, 1).
        normalization: Applied statistics, or None for unscaled data.
    """

    def __init__(self, file_path, indices=None, normalization=None):
        """Load and optionally select and normalize samples.

        Args:
            file_path: NPZ archive containing arrays X and y.
            indices: Optional one-dimensional integer array selecting samples.
            normalization: Optional scalar tuple (X_mean, X_std, y_mean, y_std).

        Raises:
            ValueError: If samples, indices, or normalization statistics are invalid.
            KeyError: If the archive is missing X or y.
        """
        super().__init__()
        # Close the archive after loading without modifying the source file.
        with np.load(file_path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have shape (num_samples, N) and be nonempty.")
        if y.shape != (X.shape[0],):
            raise ValueError("y must have shape (num_samples,).")
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("X and y must be real.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")

        if indices is not None:
            indices = np.asarray(indices)
            if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
                raise ValueError("indices must be a nonempty one-dimensional integer array.")
            if np.any(indices < 0) or np.any(indices >= len(X)):
                raise ValueError("Sample index out of range.")
            X = X[indices]
            y = y[indices]

        self.normalization = None
        if normalization is not None:
            stats = np.asarray(normalization, dtype=float)
            if stats.shape != (4,) or not np.all(np.isfinite(stats)):
                raise ValueError("normalization must contain four finite scalars.")
            X_mean, X_std, y_mean, y_std = stats
            if X_std <= 0 or y_std <= 0:
                raise ValueError("Standard deviations must be positive.")
            # Use one global scale, rather than normalizing each curve separately.
            X = (X - X_mean) / X_std
            y = (y - y_mean) / y_std
            self.normalization = tuple(stats)

        # Use float32 and column targets to match the single-output network.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        if not torch.isfinite(self.X).all().item() or not torch.isfinite(self.y).all().item():
            raise ValueError("Normalized data must remain finite in float32.")

    def __len__(self):
        """Return the number of selected samples."""
        return len(self.X)

    def __getitem__(self, index):
        """Return one potential and its energy target.

        Args:
            index: Integer sample index.

        Returns:
            A pair of float32 tensors with shapes (N,) and (1,).
        """
        return self.X[index], self.y[index]


def potential(x, A, c, sigma):
    """Evaluate a sum of Gaussian components on a spatial grid.

    Args:
        x: Finite real grid of shape (N,).
        A: Component amplitudes of shape (K,); negative values create wells.
        c: Component centers of shape (K,).
        sigma: Positive component widths of shape (K,).

    Returns:
        A potential array of shape (N,).

    Raises:
        ValueError: If parameters are nonfinite, complex, or have invalid shapes
            or nonpositive widths.
    """
    if any(np.iscomplexobj(values) for values in (x, A, c, sigma)):
        raise ValueError("The grid and Gaussian parameters must be real.")
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional array.")
    if A.ndim != 1 or A.size == 0 or c.shape != A.shape or sigma.shape != A.shape:
        raise ValueError("A, c and sigma must be nonempty one-dimensional arrays of equal length.")
    if not all(np.all(np.isfinite(values)) for values in (A, c, sigma)):
        raise ValueError("Gaussian parameters must be finite.")
    if np.any(sigma <= 0):
        raise ValueError("Every Gaussian width sigma must be positive.")

    # V(x) = sum_j A[j] * exp(-(x-c[j])**2 / (2*sigma[j]**2))
    V = np.zeros_like(x)
    for j in range(len(A)):
        V += A[j] * np.exp(-(x - c[j])**2 / (2 * sigma[j]**2))
    return V


def generate_dataset(x, num_samples=100, K=3, mass=1.0, hbar=1.0,
                     A_range=(-3.0, 2.0), c_range=(-4.0, 4.0),
                     sigma_range=(0.7, 1.5), seed=42):
    """Generate random potentials and finite-domain FDM ground-state labels.

    Labels are not filtered by sign and do not certify infinite-domain binding
    or a particular continuum accuracy. All samples share a grid and physics.

    Args:
        x: Uniform spatial grid including the zero-valued boundary endpoints.
        num_samples: Positive number of samples to generate.
        K: Positive number of Gaussian components per sample.
        mass: Positive finite particle mass.
        hbar: Positive finite reduced Planck constant.
        A_range: Increasing lower and upper amplitude bounds.
        c_range: Increasing lower and upper center bounds.
        sigma_range: Increasing positive width bounds.
        seed: Seed passed to NumPy's local random generator.

    Returns:
        A dictionary containing X with shape (num_samples, N), y with shape
        (num_samples,), and shared x, mass, hbar, and boundary_condition.

    Raises:
        ValueError: If sampling settings or solver inputs are invalid.
    """
    if (isinstance(num_samples, (bool, np.bool_))
            or not isinstance(num_samples, (int, np.integer))
            or num_samples <= 0):
        raise ValueError("num_samples must be a positive integer.")
    if (isinstance(K, (bool, np.bool_))
            or not isinstance(K, (int, np.integer)) or K <= 0):
        raise ValueError("K must be a positive integer.")
    for name, limits in [("A_range", A_range), ("c_range", c_range),
                         ("sigma_range", sigma_range)]:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            raise ValueError(f"{name} must contain two finite increasing limits.")
    if sigma_range[0] <= 0:
        raise ValueError("sigma_range must be positive.")
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a one-dimensional grid with at least 3 points.")

    # Use a local random generator for reproducible sampling.
    rng = np.random.default_rng(seed)
    A = rng.uniform(*A_range, size=(num_samples, K))
    c = rng.uniform(*c_range, size=(num_samples, K))
    sigma = rng.uniform(*sigma_range, size=(num_samples, K))
    X = np.zeros((num_samples, len(x)))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        V = potential(x, A[i], c[i], sigma[i])
        eigenvalues, _ = solve_schrodinger(x, V, mass=mass, hbar=hbar)
        X[i] = V
        y[i] = eigenvalues[0]

    # Return training data and shared physics; Gaussian parameters are intermediate.
    return {
        "X": X, "y": y, "x": x.copy(),
        "mass": mass, "hbar": hbar,
        "boundary_condition": "dirichlet_zero",
    }
