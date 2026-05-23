import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import sys
import random
from tqdm import tqdm

# Ensure the script can be run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from IPython import display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants initialization
num_workers = 4
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device):
    """Calculate classification accuracy of the model on given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total if total > 0 else 0


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=100, best_model_path=None):
    """Train the model and record per-step losses, accuracy curves."""
    model.to(device)
    train_loss_curve = [np.nan] * epochs_n
    train_acc_curve = [np.nan] * epochs_n
    val_acc_curve = [np.nan] * epochs_n
    max_val_accuracy = 0

    batches_n = len(train_loader)
    losses_list = []  # per-step losses (list of lists)
    grads = []        # per-step gradient norms

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad_list = []
        epoch_loss = 0.0

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            # record gradient norm (L2 norm of all gradients)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            grad_list.append(total_grad_norm)

            loss_list.append(loss.item())
            epoch_loss += loss.item()
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad_list)
        train_loss_curve[epoch] = epoch_loss / batches_n

        # compute accuracies
        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        train_acc_curve[epoch] = train_acc
        val_acc_curve[epoch] = val_acc

        if HAS_IPYTHON:
            display.clear_output(wait=True)

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

    return {
        'train_loss': train_loss_curve,
        'train_acc': train_acc_curve,
        'val_acc': val_acc_curve,
        'losses_list': losses_list,
        'grads': grads,
        'max_val_accuracy': max_val_accuracy,
    }


def train_models_with_lrs(model_class, lr_list, epochs_n, train_loader, val_loader,
                          criterion=None, seed=2020):
    """Train models with different learning rates and collect loss curves.

    Returns:
        all_losses: list of train_loss_curve for each lr
        all_grads: list of grads for each lr
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    all_losses = []
    all_grads = []

    for lr in lr_list:
        print(f"Training {model_class.__name__} with lr={lr}")
        set_random_seeds(seed_value=seed, device=str(device))
        model = model_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        results = train(model, optimizer, criterion, train_loader, val_loader,
                        epochs_n=epochs_n)
        all_losses.append(results['train_loss'])
        all_grads.append(results['grads'])

    return all_losses, all_grads


def compute_loss_landscape_curves(all_losses):
    """Compute max and min loss curves across different learning rates.

    For each step, take the max/min across all models trained with different lrs.
    """
    n_curves = len(all_losses)
    n_epochs = len(all_losses[0])

    max_curve = []
    min_curve = []

    for step in range(n_epochs):
        step_losses = [all_losses[i][step] for i in range(n_curves)]
        max_curve.append(np.max(step_losses))
        min_curve.append(np.min(step_losses))

    return min_curve, max_curve


def compute_gradient_predictiveness(all_grads):
    """Compute gradient predictiveness: L2 norm of gradient difference
    between consecutive steps, averaged over the training trajectory.

    Lower values indicate smoother/more predictable gradient changes.
    """
    n_curves = len(all_grads)
    predictiveness = []

    for i in range(n_curves):
        # flatten per-step gradient norms
        flat_grads = []
        for epoch_grads in all_grads[i]:
            flat_grads.extend(epoch_grads)

        if len(flat_grads) > 1:
            diffs = []
            for t in range(len(flat_grads) - 1):
                diffs.append(abs(flat_grads[t + 1] - flat_grads[t]))
            predictiveness.append(np.mean(diffs))
        else:
            predictiveness.append(0.0)

    return predictiveness


def plot_training_curves(results_bn, results_no_bn, save_path=None):
    """Plot training and validation accuracy curves for BN vs no-BN models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = np.arange(1, len(results_no_bn['train_loss']) + 1)

    # Training loss
    axes[0].plot(epochs, results_no_bn['train_loss'], label='VGG-A (no BN)', linewidth=2)
    axes[0].plot(epochs, results_bn['train_loss'], label='VGG-A + BN', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training accuracy
    axes[1].plot(epochs, results_no_bn['train_acc'], label='VGG-A (no BN)', linewidth=2)
    axes[1].plot(epochs, results_bn['train_acc'], label='VGG-A + BN', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Accuracy')
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Validation accuracy
    axes[2].plot(epochs, results_no_bn['val_acc'], label='VGG-A (no BN)', linewidth=2)
    axes[2].plot(epochs, results_bn['val_acc'], label='VGG-A + BN', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Validation Accuracy')
    axes[2].set_title('Validation Accuracy Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_landscape(min_curve_bn, max_curve_bn, min_curve_no_bn, max_curve_no_bn,
                        label_bn='With BN', label_no_bn='Without BN', save_path=None):
    """Plot loss landscape: fill between min and max loss curves.

    Compares VGG-A with BN and without BN on the same figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = np.arange(1, len(min_curve_bn) + 1)

    # VGG-A without BN
    axes[0].plot(epochs, max_curve_no_bn, 'r-', linewidth=1.5, label='Max Loss')
    axes[0].plot(epochs, min_curve_no_bn, 'b-', linewidth=1.5, label='Min Loss')
    axes[0].fill_between(epochs, min_curve_no_bn, max_curve_no_bn,
                         alpha=0.3, color='purple')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Landscape: {label_no_bn}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # VGG-A with BN
    axes[1].plot(epochs, max_curve_bn, 'r-', linewidth=1.5, label='Max Loss')
    axes[1].plot(epochs, min_curve_bn, 'b-', linewidth=1.5, label='Min Loss')
    axes[1].fill_between(epochs, min_curve_bn, max_curve_bn,
                         alpha=0.3, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Loss Landscape: {label_bn}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Combined comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.fill_between(epochs, min_curve_no_bn, max_curve_no_bn,
                    alpha=0.2, color='red', label=label_no_bn)
    ax.fill_between(epochs, min_curve_bn, max_curve_bn,
                    alpha=0.2, color='blue', label=label_bn)
    ax.plot(epochs, [(a + b) / 2 for a, b in zip(min_curve_no_bn, max_curve_no_bn)],
            'r-', linewidth=2, label=f'{label_no_bn} (mean)')
    ax.plot(epochs, [(a + b) / 2 for a, b in zip(min_curve_bn, max_curve_bn)],
            'b-', linewidth=2, label=f'{label_bn} (mean)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Landscape Comparison: BN vs Without BN')
    ax.legend()
    ax.grid(True, alpha=0.3)

    combined_path = save_path.replace('.png', '_combined.png') if save_path else None
    if combined_path:
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gradient_predictiveness(pred_bn, pred_no_bn, lr_list, save_path=None):
    """Plot gradient predictiveness comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(lr_list))
    width = 0.35
    ax.bar(x - width / 2, pred_no_bn, width, label='Without BN', alpha=0.8)
    ax.bar(x + width / 2, pred_bn, width, label='With BN', alpha=0.8)
    ax.set_xlabel('Learning Rate Index')
    ax.set_ylabel('Mean Gradient Change')
    ax.set_title('Gradient Predictiveness: BN vs Without BN')
    ax.set_xticks(x)
    ax.set_xticklabels([str(lr) for lr in lr_list])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Execution
# ============================================================
if __name__ == '__main__':
    # Load data
    print("Loading CIFAR-10 data...")
    train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers)

    # Verify data loading
    for X, y in train_loader:
        print(f"Batch shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")
        break

    # Hyperparameters
    EPOCHS = 30
    LR_LIST = [1e-3, 2e-3, 1e-4, 5e-4]
    SEED = 2020

    criterion = nn.CrossEntropyLoss()

    # ========================================
    # Part A: Train VGG-A without BN
    # ========================================
    print("\n" + "=" * 60)
    print("Training VGG-A WITHOUT BatchNorm")
    print("=" * 60)

    no_bn_losses, no_bn_grads = train_models_with_lrs(
        VGG_A, LR_LIST, EPOCHS, train_loader, val_loader,
        criterion=criterion, seed=SEED
    )
    min_curve_no_bn, max_curve_no_bn = compute_loss_landscape_curves(no_bn_losses)
    pred_no_bn = compute_gradient_predictiveness(no_bn_grads)

    # Save
    np.savetxt(os.path.join(figures_path, 'loss_no_bn.txt'),
               np.array(no_bn_losses).T, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(figures_path, 'min_curve_no_bn.txt'),
               min_curve_no_bn, fmt='%.6f')
    np.savetxt(os.path.join(figures_path, 'max_curve_no_bn.txt'),
               max_curve_no_bn, fmt='%.6f')

    # Train a single best model for accuracy comparison
    print("\nTraining final VGG-A model (without BN)...")
    set_random_seeds(seed_value=SEED, device=str(device))
    model_no_bn = VGG_A()
    optimizer_no_bn = torch.optim.SGD(model_no_bn.parameters(), lr=0.01, momentum=0.9)
    results_no_bn = train(model_no_bn, optimizer_no_bn, criterion, train_loader,
                          val_loader, epochs_n=EPOCHS,
                          best_model_path=os.path.join(models_path, 'vgg_a_best.pth'))

    # ========================================
    # Part B: Train VGG-A with BN
    # ========================================
    print("\n" + "=" * 60)
    print("Training VGG-A WITH BatchNorm")
    print("=" * 60)

    bn_losses, bn_grads = train_models_with_lrs(
        VGG_A_BatchNorm, LR_LIST, EPOCHS, train_loader, val_loader,
        criterion=criterion, seed=SEED
    )
    min_curve_bn, max_curve_bn = compute_loss_landscape_curves(bn_losses)
    pred_bn = compute_gradient_predictiveness(bn_grads)

    # Save
    np.savetxt(os.path.join(figures_path, 'loss_bn.txt'),
               np.array(bn_losses).T, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(figures_path, 'min_curve_bn.txt'),
               min_curve_bn, fmt='%.6f')
    np.savetxt(os.path.join(figures_path, 'max_curve_bn.txt'),
               max_curve_bn, fmt='%.6f')

    # Train a single best model for accuracy comparison
    print("\nTraining final VGG-A model (with BN)...")
    set_random_seeds(seed_value=SEED, device=str(device))
    model_bn = VGG_A_BatchNorm()
    optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=0.01, momentum=0.9)
    results_bn = train(model_bn, optimizer_bn, criterion, train_loader,
                       val_loader, epochs_n=EPOCHS,
                       best_model_path=os.path.join(models_path, 'vgg_a_bn_best.pth'))

    # ========================================
    # Part C: Visualizations
    # ========================================
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 1. Training curves comparison
    plot_training_curves(results_bn, results_no_bn,
                         save_path=os.path.join(figures_path, 'training_curves.png'))
    print("Saved training curves.")

    # 2. Loss landscape
    plot_loss_landscape(min_curve_bn, max_curve_bn, min_curve_no_bn, max_curve_no_bn,
                        save_path=os.path.join(figures_path, 'loss_landscape.png'))
    print("Saved loss landscape plot.")

    # 3. Gradient predictiveness
    plot_gradient_predictiveness(pred_bn, pred_no_bn, LR_LIST,
                                 save_path=os.path.join(figures_path, 'gradient_predictiveness.png'))
    print("Saved gradient predictiveness plot.")

    # 4. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"VGG-A (no BN)  - Best val acc: {results_no_bn['max_val_accuracy']:.4f}")
    print(f"VGG-A (with BN) - Best val acc: {results_bn['max_val_accuracy']:.4f}")
    print(f"\nGradient predictiveness (lower = smoother):")
    for i, lr in enumerate(LR_LIST):
        print(f"  LR={lr}: no_BN={pred_no_bn[i]:.6f}, with_BN={pred_bn[i]:.6f}")
    print(f"\nAll figures saved to: {figures_path}")
    print(f"Models saved to: {models_path}")
