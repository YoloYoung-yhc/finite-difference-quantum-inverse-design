"""Generate and save disjoint training and test datasets."""

from pathlib import Path

import numpy as np

from generate_dataset import generate_dataset


# Generate and save only when executed directly.
if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    data_directory = directory / 'data'
    data_directory.mkdir(parents=True, exist_ok=True)

    hbar = 1.0  # Reduced Planck's constant
    m = 1.0     # Mass of the particle
    L = 20.0    # Length of the spatial domain
    N = 200  # Number of spatial points
    x = np.linspace(-L/2, L/2, N)  # Spatial grid
    dx = x[1] - x[0]  # Spatial step size

    num_samples = 100
    K = 3
    seed = 42
    train_ratio = 0.8
    num_train = int(num_samples * train_ratio)
    if not 0 < num_train < num_samples:
        raise ValueError("The training and test sets must both contain samples.")

    dataset = generate_dataset(x, num_samples=num_samples, K=K,
                               mass=m, hbar=hbar, seed=seed)

    # Split X and y with identical indices to preserve sample-label pairing.
    # Use a fixed seed and disjoint training and test indices.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # Save identical physical settings and unstandardized values in both files.
    # Fit subsequent normalization on training data only.
    for output_path, sample_indices in [
        (data_directory / "train_dataset.npz", train_indices),
        (data_directory / "test_dataset.npz", test_indices),
    ]:
        X = dataset["X"][sample_indices]
        y = dataset["y"][sample_indices]
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            x=dataset["x"],
            mass=dataset["mass"],
            hbar=dataset["hbar"],
            boundary_condition=dataset["boundary_condition"],
        )
        print("Dataset saved to:", output_path)
        print("X shape (potentials):", X.shape)
        print("y shape (ground-state energies):", y.shape)
        print("Energy range:", y.min(), y.max())
