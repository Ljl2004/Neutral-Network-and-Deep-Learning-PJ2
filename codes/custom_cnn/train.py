"""
Part 1: Train a Custom Network on CIFAR-10

This script trains multiple network configurations and evaluates:
- Different model architectures (CNN, ResNet, different sizes)
- Different loss functions (CrossEntropy, with L2 regularization)
- Different activations (ReLU, LeakyReLU, GELU)
- Different optimizers (SGD, Adam, AdamW)
- Different numbers of filters/neurons

All experiments are logged and the best model is saved.
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
import json
from tqdm import tqdm

# Ensure the script can be run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.custom_cnn import (
    CustomCNN, CustomResNet,
    custom_cnn_small, custom_cnn_medium, custom_cnn_large,
    custom_resnet_small, custom_resnet_medium, custom_resnet_large,
    get_number_of_parameters,
)
from data.loaders import get_cifar_loader
from utils.nn import set_random_seeds, AverageMeter, get_accuracy

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 50
SEED = 42
LR = 0.01
WEIGHT_DECAY = 5e-4
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'VGG_BatchNorm', 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

figures_path = os.path.join(os.path.dirname(__file__), 'reports', 'figures')
models_path = os.path.join(os.path.dirname(__file__), 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)


# ============================================================
# Training function
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return losses.avg, correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            losses.update(loss.item(), x.size(0))
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return losses.avg, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, epochs=50, model_name='model',
                save_best=True):
    """Train model and return training history."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_val_acc': 0, 'best_epoch': 0,
        'model_name': model_name,
        'num_params': get_number_of_parameters(model),
    }

    model.to(device)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            if save_best:
                torch.save(model.state_dict(),
                          os.path.join(models_path, f'{model_name}_best.pth'))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return history


# ============================================================
# Experiment 1: Different model architectures (filters/neurons)
# ============================================================
def experiment_architecture():
    """Compare different model architectures (different numbers of filters)."""
    print("\n" + "=" * 60)
    print("Experiment 1: Different Model Architectures")
    print("=" * 60)

    train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True,
                                    num_workers=NUM_WORKERS, augmentation=True,
                                    root=DATA_ROOT)
    val_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False,
                                  num_workers=NUM_WORKERS, root=DATA_ROOT)

    model_factories = {
        'CNN-Small': custom_cnn_small,
        'CNN-Medium': custom_cnn_medium,
        'CNN-Large': custom_cnn_large,
        'ResNet-Small': custom_resnet_small,
        'ResNet-Medium': custom_resnet_medium,
        'ResNet-Large': custom_resnet_large,
    }

    results = {}
    for name, model_fn in model_factories.items():
        set_random_seeds(SEED, str(device))
        model = model_fn()
        print(f"\nTraining {name} ({get_number_of_parameters(model):,} params)...")

        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                              weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        history = train_model(model, train_loader, val_loader, criterion,
                             optimizer, scheduler, epochs=EPOCHS,
                             model_name=name)
        results[name] = history
        print(f"  Best val acc: {history['best_val_acc']:.4f} "
              f"(epoch {history['best_epoch']})")

    return results


# ============================================================
# Experiment 2: Different loss functions / regularization
# ============================================================
def experiment_loss_functions():
    """Compare different loss functions and regularization strengths."""
    print("\n" + "=" * 60)
    print("Experiment 2: Different Loss Functions & Regularization")
    print("=" * 60)

    train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True,
                                    num_workers=NUM_WORKERS, augmentation=True,
                                    root=DATA_ROOT)
    val_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False,
                                  num_workers=NUM_WORKERS, root=DATA_ROOT)

    configs = {
        'CrossEntropy (no reg)': {'criterion': nn.CrossEntropyLoss(), 'wd': 0},
        'CrossEntropy + L2 (1e-4)': {'criterion': nn.CrossEntropyLoss(), 'wd': 1e-4},
        'CrossEntropy + L2 (5e-4)': {'criterion': nn.CrossEntropyLoss(), 'wd': 5e-4},
        'CrossEntropy + L2 (1e-3)': {'criterion': nn.CrossEntropyLoss(), 'wd': 1e-3},
        'CrossEntropy + LabelSmoothing(0.1)': {
            'criterion': nn.CrossEntropyLoss(label_smoothing=0.1), 'wd': 5e-4
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\nTraining with {name}...")
        set_random_seeds(SEED, str(device))
        model = custom_cnn_medium()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                              weight_decay=config['wd'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        history = train_model(model, train_loader, val_loader, config['criterion'],
                             optimizer, scheduler, epochs=EPOCHS,
                             model_name=f"loss_{name[:20]}")
        results[name] = history
        print(f"  Best val acc: {history['best_val_acc']:.4f}")

    return results


# ============================================================
# Experiment 3: Different activations
# ============================================================
def experiment_activations():
    """Compare different activation functions."""
    print("\n" + "=" * 60)
    print("Experiment 3: Different Activation Functions")
    print("=" * 60)

    train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True,
                                    num_workers=NUM_WORKERS, augmentation=True,
                                    root=DATA_ROOT)
    val_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False,
                                  num_workers=NUM_WORKERS, root=DATA_ROOT)

    activations = ['relu', 'leaky_relu', 'gelu']
    results = {}

    for act in activations:
        print(f"\nTraining with {act} activation...")
        set_random_seeds(SEED, str(device))
        model = CustomCNN(filters=(64, 128, 256), num_blocks=(2, 2, 2),
                          fc_size=256, activation=act)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                              weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        history = train_model(model, train_loader, val_loader, criterion,
                             optimizer, scheduler, epochs=EPOCHS,
                             model_name=f"activation_{act}")
        results[act] = history
        print(f"  Best val acc: {history['best_val_acc']:.4f}")

    return results


# ============================================================
# Experiment 4: Different optimizers
# ============================================================
def experiment_optimizers():
    """Compare different optimization algorithms."""
    print("\n" + "=" * 60)
    print("Experiment 4: Different Optimizers")
    print("=" * 60)

    train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True,
                                    num_workers=NUM_WORKERS, augmentation=True,
                                    root=DATA_ROOT)
    val_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False,
                                  num_workers=NUM_WORKERS, root=DATA_ROOT)

    opt_configs = {
        'SGD+Momentum': lambda params: optim.SGD(params, lr=0.01, momentum=0.9,
                                                  weight_decay=WEIGHT_DECAY),
        'Adam': lambda params: optim.Adam(params, lr=0.001,
                                           weight_decay=WEIGHT_DECAY),
        'AdamW': lambda params: optim.AdamW(params, lr=0.001,
                                             weight_decay=WEIGHT_DECAY),
        'SGD+Nesterov': lambda params: optim.SGD(params, lr=0.01, momentum=0.9,
                                                  weight_decay=WEIGHT_DECAY,
                                                  nesterov=True),
    }

    results = {}
    for name, opt_fn in opt_configs.items():
        print(f"\nTraining with {name} optimizer...")
        set_random_seeds(SEED, str(device))
        model = custom_cnn_medium()
        optimizer = opt_fn(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        history = train_model(model, train_loader, val_loader, criterion,
                             optimizer, scheduler, epochs=EPOCHS,
                             model_name=f"opt_{name}")
        results[name] = history
        print(f"  Best val acc: {history['best_val_acc']:.4f}")

    return results


# ============================================================
# Final Best Model Training
# ============================================================
def train_best_model():
    """Train the best configuration for final evaluation."""
    print("\n" + "=" * 60)
    print("Training Best Model (CustomResNet-Medium + AdamW)")
    print("=" * 60)

    train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True,
                                    num_workers=NUM_WORKERS, augmentation=True,
                                    root=DATA_ROOT)
    val_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False,
                                  num_workers=NUM_WORKERS, root=DATA_ROOT)

    set_random_seeds(SEED, str(device))
    model = custom_resnet_medium(use_bn=True, use_dropout=True, dropout_rate=0.2)
    print(f"Model parameters: {get_number_of_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = train_model(model, train_loader, val_loader, criterion,
                         optimizer, scheduler, epochs=EPOCHS,
                         model_name='best_model')

    # Final evaluation on test set
    model.load_state_dict(torch.load(
        os.path.join(models_path, 'best_model_best.pth')))
    test_acc = get_accuracy(model, val_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    return model, history, test_acc


# ============================================================
# Plotting functions
# ============================================================
def plot_experiment_results(all_results, exp_name, save_path):
    """Plot results for a single experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, history in all_results.items():
        epochs = np.arange(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=name, linewidth=1.5)
        axes[1].plot(epochs, history['val_acc'], label=f"{name} ({history['best_val_acc']:.3f})", linewidth=1.5)

    # Also plot val loss
    for name, history in all_results.items():
        epochs = np.arange(1, len(history['val_loss']) + 1)
        axes[2].plot(epochs, history['val_loss'], label=name, linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'{exp_name}: Training Loss')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title(f'{exp_name}: Validation Accuracy')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title(f'{exp_name}: Validation Loss')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison_summary(summary, save_path):
    """Plot bar chart comparing best validation accuracy across experiments."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = list(summary.keys())
    values = list(summary.values())

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    all_summary = {}

    # Run all experiments
    try:
        arch_results = experiment_architecture()
        all_summary.update({
            f"{k} ({v['num_params']:,}p)": v['best_val_acc']
            for k, v in arch_results.items()
        })
        plot_experiment_results(arch_results, 'Architecture Comparison',
                               os.path.join(figures_path, 'exp1_architecture.png'))
    except Exception as e:
        print(f"Experiment 1 failed: {e}")

    try:
        loss_results = experiment_loss_functions()
        all_summary.update({
            k: v['best_val_acc'] for k, v in loss_results.items()
        })
        plot_experiment_results(loss_results, 'Loss Function Comparison',
                               os.path.join(figures_path, 'exp2_loss_functions.png'))
    except Exception as e:
        print(f"Experiment 2 failed: {e}")

    try:
        act_results = experiment_activations()
        all_summary.update({
            k: v['best_val_acc'] for k, v in act_results.items()
        })
        plot_experiment_results(act_results, 'Activation Comparison',
                               os.path.join(figures_path, 'exp3_activations.png'))
    except Exception as e:
        print(f"Experiment 3 failed: {e}")

    try:
        opt_results = experiment_optimizers()
        all_summary.update({
            k: v['best_val_acc'] for k, v in opt_results.items()
        })
        plot_experiment_results(opt_results, 'Optimizer Comparison',
                               os.path.join(figures_path, 'exp4_optimizers.png'))
    except Exception as e:
        print(f"Experiment 4 failed: {e}")

    # Train best model
    try:
        best_model, best_history, test_acc = train_best_model()
        all_summary['Best Model (ResNet-Medium)'] = test_acc
    except Exception as e:
        print(f"Best model training failed: {e}")

    # Plot summary
    plot_comparison_summary(all_summary,
                           os.path.join(figures_path, 'summary_comparison.png'))

    # Save results as JSON
    results_file = os.path.join(figures_path, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, acc in all_summary.items():
        print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
