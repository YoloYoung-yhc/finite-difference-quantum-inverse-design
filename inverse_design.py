"""Optimize Gaussian potentials with a frozen surrogate and verify with FDM."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from energy_mlp import EnergyMLP
from finite_difference_solver import solve_schrodinger
from generate_dataset import potential


def potential_torch(x, A, c, sigma):
    """Evaluate Gaussian components while preserving parameter gradients.

    Args:
        x: Spatial tensor of shape (N,).
        A: Amplitude tensor of shape (K,).
        c: Center tensor of shape (K,).
        sigma: Positive width tensor of shape (K,).

    Returns:
        Differentiable potential tensor of shape (N,).
    """
    return (A[:, None] * torch.exp(
        -(x[None, :] - c[:, None])**2 / (2 * sigma[:, None]**2)
    )).sum(dim=0)


def _plot_results(x, V_initial, V_final, losses, best_step,
                  E_target, E0_predicted, E0_fdm, figure_directory):
    """Save and display optimization loss and the verified design.

    Args:
        x: Spatial grid.
        V_initial: Potential before optimization.
        V_final: Selected candidate potential.
        losses: Squared surrogate target errors, including the initial state.
        best_step: Index of the selected candidate in the optimization history.
        E_target: Requested energy in physical units.
        E0_predicted: Predicted energy of the selected candidate.
        E0_fdm: FDM energy of the selected candidate.
        figure_directory: Path to the output figure directory.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Clip zeros only for log plotting; preserve the original saved losses.
    ax.semilogy(np.arange(len(losses)), np.maximum(losses, 1e-16))
    ax.axvline(best_step, color='gray', linestyle='--', label='Best candidate')
    ax.set_xlabel('Optimization step (0 = initial)')
    ax.set_ylabel('Squared energy error (original units)')
    ax.set_title('Inverse Design: Surrogate Optimization')
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_directory / 'inverse_design_loss.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, V_initial, color='gray', linestyle='--', label='Initial potential')
    ax.plot(x, V_final, color='#222222', linewidth=2, label='Designed potential')
    ax.axhline(E_target, color='#16803c', linestyle='-.', label='Target energy')
    ax.axhline(E0_fdm, color='#2563eb', linestyle='--', label='FDM energy')
    ax.axhline(E0_predicted, color='#ea8a23', linestyle=':', linewidth=2,
               label='MLP energy')
    ax.set_xlabel('x')
    ax.set_ylabel('Energy')
    ax.set_title('Inverse Design: Potential and FDM Verification')
    ax.set_xlim(x[0], x[-1])
    ax.margins(y=0.4)
    ax.text(0.03, 0.97,
            f'Target: {E_target:.6f}\nMLP: {E0_predicted:.6f}\n'
            f'FDM: {E0_fdm:.6f}\nFDM target error: {abs(E0_fdm-E_target):.3e}',
            transform=ax.transAxes, va='top', fontsize=10, linespacing=1.5,
            bbox=dict(facecolor='white', edgecolor='#e5e7eb', alpha=0.95))
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_directory / 'inverse_design_potential.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    model_path = directory / 'model' / 'energy_mlp.pth'
    output_path = directory / 'data' / 'inverse_design_result.npz'
    figure_directory = directory / 'figure'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prefer targets within the training energy range; reachability is not guaranteed.
    E_target = -2.0
    learning_rate = 0.01
    max_steps = 2000
    prediction_tolerance = 1e-4  # Stopping tolerance for surrogate optimization only.
    fdm_tolerance = 0.01        # Allowed absolute FDM target error; adjust as needed.

    # The checkpoint omits sampling bounds; use the generator defaults explicitly.
    # Match these bounds and component count to the actual training distribution.
    A_range = (-3.0, 2.0)
    c_range = (-4.0, 4.0)
    sigma_range = (0.7, 1.5)
    A_initial = np.array([-2.2, 0.8, -1.6])
    c_initial = np.array([-2.7, 0.3, 2.1])
    sigma_initial = np.array([1.1, 0.9, 1.3])
    if not np.isfinite(E_target) or min(prediction_tolerance, fdm_tolerance) <= 0:
        raise ValueError('Target must be finite and tolerances must be positive.')
    if max_steps < 1 or learning_rate <= 0:
        raise ValueError('max_steps and learning_rate must be positive.')
    for values, limits in [(A_initial, A_range), (c_initial, c_range),
                            (sigma_initial, sigma_range)]:
        if not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('Parameter bounds must be finite and increasing.')
        if not np.isfinite(values).all() or np.any(values < limits[0]) or np.any(values > limits[1]):
            raise ValueError('Initial parameters must lie within their bounds.')
    if sigma_range[0] <= 0:
        raise ValueError('Gaussian widths must be positive.')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = EnergyMLP(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Freeze network weights while retaining gradients with respect to the inputs.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    settings = checkpoint['physical_settings']
    if settings['boundary_condition'] != 'dirichlet_zero':
        raise ValueError('This script requires zero Dirichlet boundaries.')
    x = settings['x'].detach().cpu().numpy().copy()
    m = float(settings['mass'])
    hbar = float(settings['hbar'])
    if len(x) != checkpoint['model_config']['N']:
        raise ValueError('Checkpoint grid and model input dimension do not match.')
    normalization = checkpoint['normalization']
    X_mean, X_std, y_mean, y_std = (
        float(normalization[key]) for key in ('X_mean', 'X_std', 'y_mean', 'y_std')
    )
    if not np.isfinite([X_mean, X_std, y_mean, y_std]).all() or min(X_std, y_std) <= 0:
        raise ValueError('Invalid normalization statistics in checkpoint.')

    V_initial = potential(x, A_initial, c_initial, sigma_initial)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    A = torch.nn.Parameter(torch.tensor(A_initial, dtype=torch.float32, device=device))
    c = torch.nn.Parameter(torch.tensor(c_initial, dtype=torch.float32, device=device))
    sigma = torch.nn.Parameter(torch.tensor(sigma_initial, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([A, c, sigma], lr=learning_rate)
    losses = []
    predicted_energies = []
    best_loss = float('inf')
    best_step = 0
    best_parameters = None

    # Evaluate the initial state and the candidate after the final update.
    for step in range(max_steps + 1):
        optimizer.zero_grad()
        V = potential_torch(x_tensor, A, c, sigma)
        prediction = model(((V - X_mean) / X_std).unsqueeze(0)).squeeze() * y_std + y_mean
        loss = (prediction - E_target)**2
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite inverse-design loss.')
        loss_value = loss.item()
        losses.append(loss_value)
        predicted_energies.append(prediction.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_parameters = [p.detach().cpu().numpy().copy() for p in (A, c, sigma)]
        if step % 100 == 0:
            print(f'Step {step:4d}: predicted energy={prediction.item():.8f}, loss={loss_value:.6e}')
        if abs(prediction.item() - E_target) <= prediction_tolerance or step == max_steps:
            break
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in (A, c, sigma)):
            raise RuntimeError('Missing or non-finite gradient for potential parameters.')
        optimizer.step()
        # Project parameters back into the allowed training ranges.
        with torch.no_grad():
            A.clamp_(*A_range)
            c.clamp_(*c_range)
            sigma.clamp_(*sigma_range)

    # Select the lowest surrogate loss; FDM must independently verify success.
    A_final, c_final, sigma_final = best_parameters
    V_final = potential(x, A_final, c_final, sigma_final)
    # Reevaluate the exact potential array passed to FDM for a consistent comparison.
    with torch.no_grad():
        inputs = torch.tensor((V_final-X_mean)/X_std, dtype=torch.float32, device=device)
        E0_predicted = model(inputs.unsqueeze(0)).item() * y_std + y_mean
    eigenvalues, _ = solve_schrodinger(x, V_final, mass=m, hbar=hbar)
    E0_fdm = float(eigenvalues[0])
    fdm_error = abs(E0_fdm - E_target)
    success = bool(fdm_error <= fdm_tolerance)
    print('Best step:', best_step)
    print('A:', A_final, '\nc:', c_final, '\nsigma:', sigma_final)
    print(f'Target energy: {E_target:.10f}')
    print(f'MLP energy:    {E0_predicted:.10f}')
    print(f'FDM energy:    {E0_fdm:.10f}')
    print(f'MLP target error: {abs(E0_predicted-E_target):.6e}')
    print(f'FDM target error: {fdm_error:.6e}')
    print(f'MLP/FDM discrepancy: {abs(E0_predicted-E0_fdm):.6e}')
    print(f'FDM verification passed (tolerance={fdm_tolerance}): {success}')
    if not success:
        print('Candidate did not meet the FDM target tolerance; do not treat it as a successful design.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, x=x, V_initial=V_initial, V_final=V_final,
        A_initial=A_initial, c_initial=c_initial, sigma_initial=sigma_initial,
        A=A_final, c=c_final, sigma=sigma_final,
        A_range=A_range, c_range=c_range, sigma_range=sigma_range,
        E_target=E_target, E0_predicted=E0_predicted, E0_fdm=E0_fdm,
        losses=np.asarray(losses), predicted_energies=np.asarray(predicted_energies),
        best_step=best_step, fdm_error=fdm_error, success=success,
        prediction_tolerance=prediction_tolerance, fdm_tolerance=fdm_tolerance,
        learning_rate=learning_rate, max_steps=max_steps,
        mass=m, hbar=hbar, boundary_condition=settings['boundary_condition'],
        model_file=model_path.relative_to(directory).as_posix(),
    )
    print('Result saved to:', output_path)
    _plot_results(x, V_initial, V_final, losses, best_step,
                  E_target, E0_predicted, E0_fdm, figure_directory)
