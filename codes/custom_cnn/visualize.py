"""
Part 1: Visualization and Network Interpretation

Includes:
- Filter visualization (first conv layer weights)
- Loss landscape (variation across learning rates)
- Network interpretation (per-class activation maps, confusion matrix)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Ensure the script can be run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.custom_cnn import CustomCNN, CustomResNet, custom_cnn_medium, custom_resnet_medium
from data.loaders import get_cifar_loader
from utils.nn import set_random_seeds, get_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

figures_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures')
models_path = os.path.join(os.path.dirname(__file__), 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'VGG_BatchNorm', 'data')


def visualize_filters(model, save_path=None):
    """Visualize first conv layer filters as RGB images."""
    model.eval()

    # Find the first Conv2d layer
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        print("No Conv2d layer found in model")
        return

    weights = first_conv.weight.data.cpu().numpy()
    n_filters = weights.shape[0]
    n_channels = weights.shape[1]

    # Normalize weights to [0, 1] for visualization
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min + 1e-8)

    # Plot grid
    cols = 8
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if rows > 1 else [axes]

    for i in range(rows * cols):
        if i < n_filters:
            if n_channels == 3:
                # RGB filters: transpose to HxWxC
                filt = np.transpose(weights[i], (1, 2, 0))
            else:
                filt = weights[i, 0]  # show first channel

            axes[i].imshow(filt, cmap='gray' if n_channels == 1 else None)
            axes[i].set_title(f'F{i}', fontsize=8)
        axes[i].axis('off')

    plt.suptitle(f'First Layer Filters ({n_filters} filters, {n_channels} channels)',
                 fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved filter visualization: {save_path}")


def visualize_feature_maps(model, sample_input, save_path=None, max_maps=16):
    """Visualize feature maps of the first conv layer for a given input."""
    model.eval()

    # Extract intermediate features
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        return

    with torch.no_grad():
        x = sample_input.unsqueeze(0).to(device)
        # For CustomResNet
        try:
            features = first_conv(x)
        except:
            return

    features = features.cpu().numpy()[0]
    n_maps = min(features.shape[0], max_maps)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(rows * cols):
        if i < n_maps:
            axes[i].imshow(features[i], cmap='viridis')
        axes[i].axis('off')

    plt.suptitle('Feature Maps (First Conv Layer)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature maps: {save_path}")


def compute_loss_landscape(model, train_loader, criterion, device,
                           lr_list=None, num_steps=50):
    """Compute loss landscape by training with different learning rates.

    Measures how loss changes at each step for different step sizes.
    """
    if lr_list is None:
        lr_list = [1e-3, 2e-3, 1e-4, 5e-4]

    all_losses = []
    for lr in lr_list:
        # Create a fresh copy of the model
        if isinstance(model, CustomResNet):
            model_copy = custom_resnet_medium()
        else:
            model_copy = custom_cnn_medium()
        model_copy.to(device)

        optimizer = optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
        step_losses = []
        step_count = 0

        model_copy.train()
        for data in train_loader:
            if step_count >= num_steps:
                break
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model_copy(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
            step_count += 1

        all_losses.append(step_losses)

    # Compute min/max curves
    n_steps = min(len(l) for l in all_losses)
    min_curve = []
    max_curve = []
    for step in range(n_steps):
        step_vals = [all_losses[i][step] for i in range(len(lr_list))]
        min_curve.append(np.min(step_vals))
        max_curve.append(np.max(step_vals))

    return min_curve, max_curve


def plot_loss_landscape(min_curve, max_curve, model_name, save_path=None):
    """Plot loss landscape with fill_between."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    steps = np.arange(1, len(min_curve) + 1)

    ax.plot(steps, max_curve, 'r-', linewidth=1.5, label='Max Loss')
    ax.plot(steps, min_curve, 'b-', linewidth=1.5, label='Min Loss')
    ax.fill_between(steps, min_curve, max_curve, alpha=0.3, color='purple')
    ax.plot(steps, [(a + b) / 2 for a, b in zip(min_curve, max_curve)],
            'k--', linewidth=1.5, label='Mean Loss')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Landscape: {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss landscape: {save_path}")


def plot_confusion_matrix(model, val_loader, device, save_path=None):
    """Generate and plot confusion matrix for model predictions."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues')

    # Add text annotations
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                          ha='center', va='center',
                          fontsize=7,
                          color='white' if cm_normalized[i, j] > 0.5 else 'black')

    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_training_dynamics(histories, save_path=None):
    """Plot training dynamics: loss curves over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, history in histories.items():
        epochs = np.arange(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=name, linewidth=1.5)
        axes[1].plot(epochs, history['val_acc'], label=name, linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Dynamics: Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Training Dynamics: Accuracy')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_per_class_accuracy(model, val_loader, device, save_path=None):
    """Compute and visualize per-class accuracy."""
    model.eval()
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            c = (preds == y.to(device)).cpu().numpy()
            for i in range(len(y)):
                label = y[i].item()
                class_correct[label] += c[i]
                class_total[label] += 1

    per_class_acc = class_correct / class_total

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
    bars = ax.bar(CIFAR10_CLASSES, per_class_acc, color=colors, edgecolor='black')

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=9)

    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Classification Accuracy')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class accuracy: {save_path}")

    return per_class_acc


# ============================================================
# Main: Run all visualizations
# ============================================================
if __name__ == '__main__':
    print("Loading data...")
    val_loader = get_cifar_loader(batch_size=128, train=False, num_workers=4, root=DATA_ROOT)
    train_loader_small = get_cifar_loader(batch_size=128, train=True, num_workers=4, root=DATA_ROOT)

    # Get a sample input
    for x, y in val_loader:
        sample_input = x[0]
        sample_label = y[0]
        print(f"Sample: class={CIFAR10_CLASSES[sample_label]}, shape={sample_input.shape}")
        break

    # Try to load best model
    best_model_path = os.path.join(models_path, 'best_model_best.pth')
    models_to_viz = []

    # Create and visualize a fresh model
    print("\nVisualizing CustomCNN (Medium)...")
    cnn_model = custom_cnn_medium()
    cnn_model.to(device)
    models_to_viz.append(('CustomCNN', cnn_model))

    print("Visualizing CustomResNet (Medium)...")
    resnet_model = custom_resnet_medium()
    resnet_model.to(device)
    models_to_viz.append(('CustomResNet', resnet_model))

    # Run visualizations for each model
    for model_name, model in models_to_viz:
        print(f"\n{'='*60}")
        print(f"Visualizations for {model_name}")
        print(f"{'='*60}")

        # 1. Filter visualization
        visualize_filters(model,
                         save_path=os.path.join(figures_path,
                                               f'{model_name}_filters.png'))

        # 2. Feature maps
        visualize_feature_maps(model, sample_input,
                              save_path=os.path.join(figures_path,
                                                     f'{model_name}_feature_maps.png'))

        # 3. Loss landscape
        criterion = nn.CrossEntropyLoss()
        min_curve, max_curve = compute_loss_landscape(
            model, train_loader_small, criterion, device,
            lr_list=[1e-3, 2e-3, 1e-4, 5e-4])
        plot_loss_landscape(min_curve, max_curve, model_name,
                           save_path=os.path.join(figures_path,
                                                  f'{model_name}_loss_landscape.png'))

        # 4. Confusion matrix (only if model is trained, use random as placeholder)
        # For a fresh model, we train briefly first
        print(f"  Training {model_name} briefly for confusion matrix...")
        set_random_seeds(42, str(device))
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        steps = 0
        for data in train_loader_small:
            if steps >= 100:
                break
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            steps += 1

        plot_confusion_matrix(model, val_loader, device,
                             save_path=os.path.join(figures_path,
                                                    f'{model_name}_confusion.png'))
        visualize_per_class_accuracy(model, val_loader, device,
                                    save_path=os.path.join(figures_path,
                                                           f'{model_name}_per_class.png'))

    print("\nAll visualizations complete!")
    print(f"Figures saved to: {figures_path}")
