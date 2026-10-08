"""Train an energy surrogate and save the best validation checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from energy_mlp import EnergyMLP
from generate_dataset import QuantumDataset


def evaluate_loss(model, loader, criterion, device):
    """Compute sample-weighted MSE and MAE without updating model weights.

    Args:
        model: Regression network on the requested device.
        loader: Loader yielding potential and scalar-target batches.
        criterion: MSELoss configured with reduction='mean'.
        device: Device used for evaluation.

    Returns:
        A tuple (mse, mae) in the units of the loader's targets. These are
        standardized units in this training script.

    Raises:
        ValueError: If the loader yields no samples.
    """
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for V, E in loader:
            V, E = V.to(device), E.to(device)
            prediction = model(V)
            loss = criterion(prediction, E)
            total_loss += loss.item() * len(V)
            total_absolute_error += torch.abs(prediction - E).sum().item()
            sample_count += len(V)
    if sample_count == 0:
        raise ValueError('Cannot evaluate an empty data loader.')
    return total_loss / sample_count, total_absolute_error / sample_count


def _plot_results(train_losses, test_losses, train_maes, test_maes, actual, predicted):
    """Save and display three independent training and prediction figures.

    Args:
        train_losses: Per-epoch training MSE in standardized units.
        test_losses: Per-epoch test MSE in standardized units.
        train_maes: Per-epoch training MAE in standardized units.
        test_maes: Per-epoch test MAE in standardized units.
        actual: FDM test energies in physical units.
        predicted: Predicted test energies in physical units.
    """
    figure_directory = Path(__file__).resolve().parent / 'figure'
    figure_directory.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (standardized energy)')
    plt.title('Energy MLP: Regression Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_directory / 'energy_mlp_mse.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_maes, label='Train')
    plt.plot(epochs, test_maes, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (standardized energy)')
    plt.title('Energy MLP: Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_directory / 'energy_mlp_mae.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, label='Test samples')
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], 'k--', label='Perfect prediction')
    plt.xlabel('FDM ground-state energy')
    plt.ylabel('Predicted ground-state energy')
    plt.title('Energy MLP: Test Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_directory / 'energy_mlp_test_predictions.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # Resolve paths relative to this script, independent of the working directory.
    directory = Path(__file__).resolve().parent
    data_path = directory / 'data' / 'train_dataset.npz'
    test_data_path = directory / 'data' / 'test_dataset.npz'
    model_path = directory / 'model' / 'energy_mlp.pth'

    seed = 42
    batch_size = 16
    learning_rate = 0.001
    weight_decay = 1e-4
    max_epochs = 1000
    patience = 100
    train_ratio = 0.8  # Split the training file into 80% training and 20% validation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training device:', device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Validate array shapes and finite values before normalization.
    raw_dataset = QuantumDataset(data_path)
    num_samples = len(raw_dataset)
    N = raw_dataset.X.shape[1]
    num_train = int(train_ratio * num_samples)
    num_val = num_samples - num_train
    if min(num_train, num_val) < 1:
        raise ValueError('Not enough samples for nonempty training and validation sets.')

    indices = rng.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Fit four scalars on training data only, using population standard deviations.
    train_X = raw_dataset.X[train_indices]
    train_y = raw_dataset.y[train_indices]
    X_mean = train_X.mean().item()
    X_std = train_X.std(unbiased=False).item()
    y_mean = train_y.mean().item()
    y_std = train_y.std(unbiased=False).item()
    if not np.all(np.isfinite([X_mean, X_std, y_mean, y_std])) or min(X_std, y_std) <= 0:
        raise ValueError('Training normalization statistics must be finite with positive variance.')
    normalization = (X_mean, X_std, y_mean, y_std)

    train_dataset = QuantumDataset(data_path, train_indices, normalization)
    val_dataset = QuantumDataset(data_path, val_indices, normalization)
    # Record test curves for display only, never for fitting or early stopping.
    test_dataset = QuantumDataset(test_data_path, normalization=normalization)
    num_test = len(test_dataset)
    if test_dataset.X.shape[1] != N:
        raise ValueError('Training and test sets must have the same input dimension.')
    # Require matching grids, physical parameters, and boundary conditions.
    with np.load(data_path, allow_pickle=False) as train_data, np.load(
        test_data_path, allow_pickle=False
    ) as test_data:
        for key in ('x', 'mass', 'hbar', 'boundary_condition'):
            if not np.array_equal(train_data[key], test_data[key]):
                raise ValueError(f'Training and test physical settings differ: {key}.')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=generator)
    # Evaluate training and validation losses using the same end-of-epoch weights.
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnergyMLP(N=N).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    test_losses = []
    test_maes = []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    print(f'Samples: train={num_train}, validation={num_val}, test={num_test}')

    for epoch in range(1, max_epochs + 1):
        model.train()
        for V, E in train_loader:
            V, E = V.to(device), E.to(device)
            optimizer.zero_grad()
            prediction = model(V)
            loss = criterion(prediction, E)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite training loss; check data and learning rate.')
            loss.backward()
            optimizer.step()

        train_loss, train_mae = evaluate_loss(model, train_eval_loader, criterion, device)
        val_loss, val_mae = evaluate_loss(model, val_loader, criterion, device)
        test_loss, test_mae = evaluate_loss(model, test_loader, criterion, device)
        if not np.all(np.isfinite([train_loss, val_loss, test_loss,
                                   train_mae, val_mae, test_mae])):
            raise RuntimeError('Non-finite evaluation loss; training stopped.')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        test_losses.append(test_loss)
        test_maes.append(test_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Clone weights so later updates cannot mutate the best checkpoint.
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f'Epoch {epoch:4d} (standardized): '
                  f'train MSE={train_loss:.6e}, val MSE={val_loss:.6e}, '
                  f'train MAE={train_mae:.6e}, val MAE={val_mae:.6e}')
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}.')
            break

    # Report test results for the validation-selected model, not the final epoch.
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for V, E in test_loader:
            V, E = V.to(device), E.to(device)
            predictions.append((model(V) * y_std + y_mean).cpu())
            targets.append((E * y_std + y_mean).cpu())
    predicted = torch.cat(predictions).numpy().ravel()
    actual = torch.cat(targets).numpy().ravel()
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    print(f'Best epoch: {best_epoch}; validation MSE: {best_val_loss:.6e}')
    print(f'Test MAE (original energy units): {mae:.6e}')
    print(f'Test RMSE (original energy units): {rmse:.6e}')

    # Save weights and inference metadata rather than serializing the model object.
    with np.load(data_path, allow_pickle=False) as data:
        physical_settings = {
            'x': torch.tensor(data['x'], dtype=torch.float64),
            'mass': float(data['mass'].item()),
            'hbar': float(data['hbar'].item()),
            'boundary_condition': str(data['boundary_condition'].item()),
        }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'model_config': {'N': N},
        'normalization': {'X_mean': X_mean, 'X_std': X_std,
                          'y_mean': y_mean, 'y_std': y_std},
        'physical_settings': physical_settings,
        'seed': seed,
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        # Training/validation indices refer to the training file; use the full test file.
        'data_files': {'train_validation': data_path.relative_to(directory).as_posix(),
                       'test': test_data_path.relative_to(directory).as_posix()},
        'test_indices': list(range(num_test)),
        'training_config': {'batch_size': batch_size, 'learning_rate': learning_rate,
                            'weight_decay': weight_decay, 'max_epochs': max_epochs,
                            'patience': patience, 'train_ratio': train_ratio},
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'test_losses': test_losses,
        'test_maes': test_maes,
        'training_device': str(device),
        'test_mae': mae,
        'test_rmse': rmse,
    }, model_path)
    print('Best model saved to:', model_path)

    _plot_results(train_losses, test_losses, train_maes, test_maes, actual, predicted)
