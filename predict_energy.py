"""Predict ground-state energies and compare them with finite differences."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from energy_mlp import EnergyMLP
from finite_difference_solver import solve_schrodinger
from generate_dataset import potential


def predict_energy(model, V, normalization, device):
    """Predict one ground-state energy using saved training statistics.

    This inference-only function disables gradients. Inverse design uses a
    separate differentiable path instead of this function.

    Args:
        model: Loaded EnergyMLP on the requested device.
        V: Real potential array matching the model input dimension.
        normalization: Mapping containing X_mean, X_std, y_mean, and y_std.
        device: Device for the input tensor and model.

    Returns:
        Predicted energy as a float in original physical units.

    Raises:
        ValueError: If input values, shape, or normalization are invalid.
        RuntimeError: If the predicted energy is not finite.
    """
    if np.iscomplexobj(V):
        raise ValueError('V must be real.')
    V = np.asarray(V, dtype=float)
    if V.ndim != 1 or V.size != model.network[0].in_features:
        raise ValueError('V must be a one-dimensional array matching the model input size.')
    if not np.all(np.isfinite(V)):
        raise ValueError('V must contain only finite values.')
    X_mean = float(normalization['X_mean'])
    X_std = float(normalization['X_std'])
    y_mean = float(normalization['y_mean'])
    y_std = float(normalization['y_std'])
    if not np.all(np.isfinite([X_mean, X_std, y_mean, y_std])) or min(X_std, y_std) <= 0:
        raise ValueError('Normalization statistics must be finite, with positive standard deviations.')

    V_scaled = (V - X_mean) / X_std
    # Keep a batch dimension for a single sample: (1, N).
    inputs = torch.tensor(V_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    if not torch.isfinite(inputs).all().item():
        raise ValueError('Standardized potential cannot be represented as finite float32 values.')
    model.eval()
    with torch.no_grad():
        prediction = model(inputs).item()
    E0 = prediction * y_std + y_mean
    if not np.isfinite(E0):
        raise RuntimeError('The model produced a non-finite energy.')
    return E0


def _plot_results(x, V, E0_predicted, E0_fdm):
    """Save and display the potential, energy estimates, and absolute error.

    Args:
        x: Spatial grid.
        V: Potential values on the grid.
        E0_predicted: MLP ground-state energy in physical units.
        E0_fdm: FDM ground-state energy in physical units.
    """
    figure_directory = Path(__file__).resolve().parent / 'figure'
    figure_directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, V, color='#222222', linewidth=2, label='Potential V(x)')
    ax.axhline(E0_fdm, color='#2563eb', linestyle='--', linewidth=1.8,
               label='FDM ground-state energy')
    ax.axhline(E0_predicted, color='#ea8a23', linestyle=':', linewidth=2.2,
               label='MLP ground-state energy')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Ground-State Energy Prediction', fontsize=15, pad=16)
    ax.set_xlim(x[0], x[-1])
    # Reserve space for annotations above the curves.
    ax.margins(y=0.25)
    absolute_error = abs(E0_predicted - E0_fdm)
    annotation = (
        f'FDM energy    {E0_fdm:.6f}\n'
        f'MLP energy    {E0_predicted:.6f}\n'
        f'Absolute error    {absolute_error:.3e}'
    )
    ax.text(0.03, 0.97, annotation, transform=ax.transAxes,
            ha='left', va='top', fontsize=10, linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor='#e5e7eb', alpha=0.95))
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color='#d1d5db', alpha=0.4, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figure_directory / 'energy_prediction_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    model_path = directory / 'model' / 'energy_mlp.pth'
    if not model_path.is_file():
        raise FileNotFoundError(f'Copy the trained energy_mlp.pth into this directory: {model_path.parent}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load onto CPU first to support checkpoints trained on another device.
    # Expect the checkpoint dictionary saved by train_energy_mlp.py.
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = EnergyMLP(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    normalization = checkpoint['normalization']

    # Reuse the training grid and physics, not just its number of points.
    settings = checkpoint['physical_settings']
    if settings['boundary_condition'] != 'dirichlet_zero':
        raise ValueError('This FDM comparison requires zero Dirichlet boundaries.')
    x = settings['x'].detach().cpu().numpy().copy()
    m = float(settings['mass'])
    hbar = float(settings['hbar'])

    # Edit the new potential here; this example uses the default sampling ranges:
    # K=3，A in [-3, 2]，c in [-4, 4]，sigma in [0.7, 1.5]。
    # Adjust these parameters if the model used a different training distribution.
    A = np.array([-2.2, 0.8, -1.6])
    c = np.array([-2.7, 0.3, 2.1])
    sigma = np.array([1.1, 0.9, 1.3])
    V = potential(x, A, c, sigma)

    E0_predicted = predict_energy(model, V, normalization, device)
    eigenvalues, _ = solve_schrodinger(x, V, mass=m, hbar=hbar)
    E0_fdm = eigenvalues[0]
    absolute_error = abs(E0_predicted - E0_fdm)

    print('Model:', model_path)
    print('Device:', device)
    print(f'MLP ground-state energy: {E0_predicted:.10f}')
    print(f'FDM ground-state energy: {E0_fdm:.10f}')
    print(f'Absolute error: {absolute_error:.6e}')
    _plot_results(x, V, E0_predicted, E0_fdm)
