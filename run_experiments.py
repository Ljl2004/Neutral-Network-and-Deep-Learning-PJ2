"""
Self-contained experiment runner for PJ2_2026.
Runs all experiments and generates figures for the report.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import sys
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(os.path.join(REPORTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, 'models'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
BATCH_SIZE = 64
NUM_WORKERS = 0
DATA_DIR = os.path.join(BASE_DIR, 'codes', 'VGG_BatchNorm', 'data')

# ================================================================
# Data Loading
# ================================================================
def get_train_loader(augmentation=True):
    if augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def get_val_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ================================================================
# VGG Models (inlined to avoid import issues)
# ================================================================
class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

# ================================================================
# Custom CNN Models (inlined)
# ================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class CustomCNN(nn.Module):
    def __init__(self, filters=(64, 128, 256), num_blocks=(2, 2, 2),
                 fc_size=256, num_classes=10, activation='relu', dropout_rate=0.3):
        super().__init__()
        act_fn = {'relu': nn.ReLU(inplace=True), 'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
                  'gelu': nn.GELU()}[activation]
        stages = []; in_ch = 3
        for si, (filt, n_blocks) in enumerate(zip(filters, num_blocks)):
            for _ in range(n_blocks):
                stages.append(nn.Conv2d(in_ch, filt, 3, padding=1, bias=False))
                stages.append(nn.BatchNorm2d(filt))
                stages.append(act_fn.__class__(inplace=True) if hasattr(act_fn.__class__, '__init__') else act_fn.__class__())
                in_ch = filt
            if si < len(filters) - 1:
                stages.append(nn.MaxPool2d(2, 2))
        stages.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(filters[-1], fc_size),
            act_fn.__class__(inplace=True) if hasattr(act_fn.__class__, '__init__') else act_fn.__class__(),
            nn.Dropout(dropout_rate), nn.Linear(fc_size, num_classes))
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

class CustomResNet(nn.Module):
    def __init__(self, filters=(64, 128, 256), num_blocks=(2, 2, 2),
                 fc_size=256, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, filters[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.stages = nn.ModuleList()
        in_ch = filters[0]
        for si, (filt, n_blocks) in enumerate(zip(filters, num_blocks)):
            stage = []
            for bi in range(n_blocks):
                stride = 2 if bi == 0 and si > 0 else 1
                stage.append(ResidualBlock(in_ch, filt, stride))
                in_ch = filt
            self.stages.append(nn.Sequential(*stage))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(filters[-1], fc_size), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(fc_size, num_classes))
        self._init_weights()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ================================================================
# Training Utilities
# ================================================================
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, epochs=15, save_path=None):
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'best_val_acc': 0, 'best_epoch': 0}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total, batches = 0.0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); batches += 1
            _, pred = loss.new_zeros(1).max(0)  # dummy; compute after
        # Recompute train stats properly
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                _, pred = model(x).max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        train_acc = correct / total if total > 0 else 0
        model.train()

        # Better training loss
        total_loss, batches = 0.0, 0
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); batches += 1
            correct += pred.max(1)[1].eq(y).sum().item()
            total += y.size(0)
        train_loss = total_loss / batches
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_loss, correct, total, batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                val_loss += loss.item(); batches += 1
                correct += loss.new_zeros(1).max(0)
                total += y.size(0)
        val_loss = val_loss / batches

        # Recompute validation properly
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, pred = model(x).max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total > 0 else 0

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
            if save_path:
                torch.save(model.state_dict(), save_path)

        print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f} Acc={train_acc:.4f} "
              f"Val_Loss={val_loss:.4f} Val_Acc={val_acc:.4f}")

    return history


# WARNING: the train_model above has a bug - it calls loss.backward() twice!
# It also computes train stats redundantly. Let me fix this properly below.
# Actually the code above is messy. Let me just rewrite a clean version.

# ================================================================
# Start fresh with clean training
# ================================================================
print("Loading data...")
train_loader = get_train_loader(augmentation=True)
val_loader = get_val_loader()
print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

EPOCHS = 12  # enough for meaningful results on CPU

all_results = {}

# ================================================================
# PART 2: BATCH NORMALIZATION
# ================================================================
print("\n" + "="*70)
print("PART 2: BATCH NORMALIZATION EXPERIMENTS")
print("="*70)

for name, model_cls in [('VGG-A (no BN)', VGG_A), ('VGG-A (with BN)', VGG_A_BatchNorm)]:
    print(f"\nTraining {name}...")
    set_seed(42)
    model = model_cls()
    print(f"  Parameters: {count_params(model):,}")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'best_val_acc': 0, 'best_epoch': 0}
    save_path = os.path.join(REPORTS_DIR, 'models', f'vgg_{name.replace(" ", "_")}.pth')

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total, batches = 0.0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); batches += 1
            correct += pred.max(1)[1].eq(y).sum().item()
            total += y.size(0)
        train_loss = total_loss / max(batches, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss, correct, total, batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item(); batches += 1
                correct += pred.max(1)[1].eq(y).sum().item()
                total += y.size(0)
        val_loss_avg = val_loss / max(batches, 1)
        val_acc = correct / max(total, 1)

        scheduler.step()
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)
        hist['val_loss'].append(val_loss_avg)
        hist['val_acc'].append(val_acc)

        if val_acc > hist['best_val_acc']:
            hist['best_val_acc'] = val_acc; hist['best_epoch'] = epoch
            torch.save(model.state_dict(), save_path)

        print(f"  Ep {epoch+1}: TL={train_loss:.4f} TA={train_acc:.4f} VL={val_loss_avg:.4f} VA={val_acc:.4f}")

    all_results[name] = hist

# Loss landscape
print("\nComputing loss landscape...")
lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
landscape = {}

for model_name, model_cls in [('VGG-A', VGG_A), ('VGG-A+BN', VGG_A_BatchNorm)]:
    all_losses = []
    for lr in lr_list:
        set_seed(42)
        m = model_cls().to(device)
        opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        step_losses = []
        step_count = 0
        m.train()
        for data in train_loader:
            if step_count >= 30: break
            x, y = data[0].to(device), data[1].to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
            step_count += 1
        all_losses.append(step_losses)

    n_steps = min(len(l) for l in all_losses)
    min_c, max_c = [], []
    for s in range(n_steps):
        vals = [all_losses[i][s] for i in range(len(lr_list))]
        min_c.append(np.min(vals)); max_c.append(np.max(vals))
    landscape[f'{model_name}_min'] = min_c
    landscape[f'{model_name}_max'] = max_c

# ================================================================
# PART 1: CUSTOM CNN
# ================================================================
print("\n" + "="*70)
print("PART 1: CUSTOM CNN EXPERIMENTS")
print("="*70)

# Experiment 1: Architecture
print("\n--- Exp 1: Architecture ---")
for name, model in [
    ('CNN-Small', CustomCNN(filters=(32,64,128), num_blocks=(2,2,2), fc_size=128)),
    ('CNN-Medium', CustomCNN(filters=(64,128,256), num_blocks=(2,2,2), fc_size=256)),
    ('CNN-Large', CustomCNN(filters=(64,128,256,512), num_blocks=(2,2,2,2), fc_size=512)),
    ('ResNet-Medium', CustomResNet(filters=(64,128,256), num_blocks=(2,2,2), fc_size=256)),
]:
    print(f"\n  Training {name} ({count_params(model):,} params)...")
    set_seed(42)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'best_val_acc':0, 'best_epoch':0}
    model.to(device); crit = nn.CrossEntropyLoss()
    sp = os.path.join(REPORTS_DIR, 'models', f'{name}.pth')
    for ep in range(EPOCHS):
        model.train()
        tl, c, tot, nb = 0,0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            p=model(x); l=crit(p,y); l.backward(); opt.step()
            tl+=l.item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        ta=c/max(tot,1); tla=tl/max(nb,1)
        model.eval(); vl,c,tot,nb=0,0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); p=model(x)
                vl+=crit(p,y).item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        va=c/max(tot,1); vla=vl/max(nb,1); sch.step()
        hist['train_loss'].append(tla); hist['train_acc'].append(ta)
        hist['val_loss'].append(vla); hist['val_acc'].append(va)
        if va>hist['best_val_acc']: hist['best_val_acc']=va; hist['best_epoch']=ep; torch.save(model.state_dict(),sp)
        print(f"    Ep{ep+1}: TL={tla:.4f} TA={ta:.4f} VL={vla:.4f} VA={va:.4f}")
    all_results[f'{name} ({count_params(model):,}p)'] = hist

# Experiment 2: Loss functions
print("\n--- Exp 2: Loss Functions ---")
for label, crit, wd in [
    ('CE (no reg)', nn.CrossEntropyLoss(), 0),
    ('CE+L2(5e-4)', nn.CrossEntropyLoss(), 5e-4),
    ('CE+L2(1e-3)', nn.CrossEntropyLoss(), 1e-3),
    ('CE+LabelSmooth(0.1)', nn.CrossEntropyLoss(label_smoothing=0.1), 5e-4),
]:
    print(f"\n  Training {label}...")
    set_seed(42)
    m = CustomCNN(filters=(64,128,256), num_blocks=(2,2,2), fc_size=256)
    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=wd)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[],'best_val_acc':0,'best_epoch':0}
    m.to(device); sp=os.path.join(REPORTS_DIR,'models',f'loss_{label[:10]}.pth')
    for ep in range(EPOCHS):
        m.train(); tl,c,tot,nb=0,0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            p=m(x); l=crit(p,y); l.backward(); opt.step()
            tl+=l.item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        ta=c/max(tot,1); tla=tl/max(nb,1)
        m.eval(); vl,c,tot,nb=0,0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); p=m(x)
                vl+=crit(p,y).item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        va=c/max(tot,1); vla=vl/max(nb,1); sch.step()
        hist['train_loss'].append(tla); hist['train_acc'].append(ta)
        hist['val_loss'].append(vla); hist['val_acc'].append(va)
        if va>hist['best_val_acc']: hist['best_val_acc']=va; hist['best_epoch']=ep; torch.save(m.state_dict(),sp)
        print(f"    Ep{ep+1}: TL={tla:.4f} TA={ta:.4f} VL={vla:.4f} VA={va:.4f}")
    all_results[label] = hist

# Experiment 3: Activations
print("\n--- Exp 3: Activations ---")
for act in ['relu', 'leaky_relu']:
    print(f"\n  Training activation={act}...")
    set_seed(42)
    m = CustomCNN(filters=(64,128,256), num_blocks=(2,2,2), fc_size=256, activation=act)
    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    hist = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[],'best_val_acc':0,'best_epoch':0}
    m.to(device); sp=os.path.join(REPORTS_DIR,'models',f'act_{act}.pth')
    for ep in range(EPOCHS):
        m.train(); tl,c,tot,nb=0,0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            p=m(x); l=crit(p,y); l.backward(); opt.step()
            tl+=l.item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        ta=c/max(tot,1); tla=tl/max(nb,1)
        m.eval(); vl,c,tot,nb=0,0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); p=m(x)
                vl+=crit(p,y).item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        va=c/max(tot,1); vla=vl/max(nb,1); sch.step()
        hist['train_loss'].append(tla); hist['train_acc'].append(ta)
        hist['val_loss'].append(vla); hist['val_acc'].append(va)
        if va>hist['best_val_acc']: hist['best_val_acc']=va; hist['best_epoch']=ep; torch.save(m.state_dict(),sp)
        print(f"    Ep{ep+1}: TL={tla:.4f} TA={ta:.4f} VL={vla:.4f} VA={va:.4f}")
    all_results[f'Activation={act}'] = hist

# Experiment 4: Optimizers
print("\n--- Exp 4: Optimizers ---")
for opt_name, opt_fn in [
    ('SGD+Momentum', lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=5e-4)),
    ('Adam', lambda p: optim.Adam(p, lr=0.001, weight_decay=5e-4)),
    ('AdamW', lambda p: optim.AdamW(p, lr=0.001, weight_decay=5e-4)),
]:
    print(f"\n  Training {opt_name}...")
    set_seed(42)
    m = CustomCNN(filters=(64,128,256), num_blocks=(2,2,2), fc_size=256)
    opt = opt_fn(m.parameters())
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    hist = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[],'best_val_acc':0,'best_epoch':0}
    m.to(device); sp=os.path.join(REPORTS_DIR,'models',f'opt_{opt_name}.pth')
    for ep in range(EPOCHS):
        m.train(); tl,c,tot,nb=0,0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            p=m(x); l=crit(p,y); l.backward(); opt.step()
            tl+=l.item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        ta=c/max(tot,1); tla=tl/max(nb,1)
        m.eval(); vl,c,tot,nb=0,0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); p=m(x)
                vl+=crit(p,y).item(); nb+=1; c+=p.max(1)[1].eq(y).sum().item(); tot+=y.size(0)
        va=c/max(tot,1); vla=vl/max(nb,1); sch.step()
        hist['train_loss'].append(tla); hist['train_acc'].append(ta)
        hist['val_loss'].append(vla); hist['val_acc'].append(va)
        if va>hist['best_val_acc']: hist['best_val_acc']=va; hist['best_epoch']=ep; torch.save(m.state_dict(),sp)
        print(f"    Ep{ep+1}: TL={tla:.4f} TA={ta:.4f} VL={vla:.4f} VA={va:.4f}")
    all_results[f'Optimizer={opt_name}'] = hist

# ================================================================
# GENERATE FIGURES
# ================================================================
print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

# Figure 1: BN training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = np.arange(1, EPOCHS+1)
axes[0].plot(ep, all_results['VGG-A (no BN)']['train_loss'], 'r-', lw=2, label='VGG-A (no BN)')
axes[0].plot(ep, all_results['VGG-A (with BN)']['train_loss'], 'b-', lw=2, label='VGG-A (with BN)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss: BN vs Without BN')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, all_results['VGG-A (no BN)']['train_acc'], 'r-', lw=2, label='VGG-A (no BN)')
axes[1].plot(ep, all_results['VGG-A (with BN)']['train_acc'], 'b-', lw=2, label='VGG-A (with BN)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Training Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(ep, all_results['VGG-A (no BN)']['val_acc'], 'r-', lw=2, label='VGG-A (no BN)')
axes[2].plot(ep, all_results['VGG-A (with BN)']['val_acc'], 'b-', lw=2, label='VGG-A (with BN)')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy'); axes[2].set_title('Validation Accuracy')
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_training_curves.png'),dpi=150); plt.close()
print("Saved: bn_training_curves.png")

# Figure 2: Loss landscape
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, mn in enumerate(['VGG-A', 'VGG-A+BN']):
    steps = np.arange(1, len(landscape[f'{mn}_min'])+1)
    axes[idx].plot(steps, landscape[f'{mn}_max'], 'r-', lw=1.5, label='Max Loss')
    axes[idx].plot(steps, landscape[f'{mn}_min'], 'b-', lw=1.5, label='Min Loss')
    axes[idx].fill_between(steps, landscape[f'{mn}_min'], landscape[f'{mn}_max'], alpha=0.3, color='purple')
    axes[idx].set_xlabel('Step'); axes[idx].set_ylabel('Loss'); axes[idx].set_title(f'Loss Landscape: {mn}')
    axes[idx].legend(); axes[idx].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_loss_landscape.png'),dpi=150); plt.close()
print("Saved: bn_loss_landscape.png")

# Figure 3: Combined landscape
fig, ax = plt.subplots(figsize=(10,6))
s1 = np.arange(1, len(landscape['VGG-A_min'])+1)
s2 = np.arange(1, len(landscape['VGG-A+BN_min'])+1)
ax.fill_between(s1, landscape['VGG-A_min'], landscape['VGG-A_max'], alpha=0.2, color='red', label='VGG-A (no BN)')
ax.fill_between(s2, landscape['VGG-A+BN_min'], landscape['VGG-A+BN_max'], alpha=0.2, color='blue', label='VGG-A (with BN)')
ax.plot(s1, [(a+b)/2 for a,b in zip(landscape['VGG-A_min'], landscape['VGG-A_max'])], 'r-', lw=2, label='VGG-A mean')
ax.plot(s2, [(a+b)/2 for a,b in zip(landscape['VGG-A+BN_min'], landscape['VGG-A+BN_max'])], 'b-', lw=2, label='VGG-A+BN mean')
ax.set_xlabel('Training Step'); ax.set_ylabel('Loss')
ax.set_title('Loss Landscape Comparison: BN vs Without BN')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_loss_landscape_combined.png'),dpi=150); plt.close()
print("Saved: bn_loss_landscape_combined.png")

# Figure 4-7: Part 1 experiments
def plot_p1(results_dict, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, hist in results_dict.items():
        ep = np.arange(1, len(hist['train_loss'])+1)
        axes[0].plot(ep, hist['train_loss'], lw=1.5, label=name)
        axes[1].plot(ep, hist['val_acc'], lw=1.5, label=f"{name} ({hist['best_val_acc']:.3f})")
        axes[2].plot(ep, hist['val_loss'], lw=1.5, label=name)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title(f'{title}: Train Loss')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title(f'{title}: Val Acc')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss'); axes[2].set_title(f'{title}: Val Loss')
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures',fname),dpi=150); plt.close()
    print(f"Saved: {fname}")

exp1 = {k:v for k,v in all_results.items() if 'p)' in k}
exp2 = {k:v for k,v in all_results.items() if 'CE' in k or 'Label' in k}
exp3 = {k:v for k,v in all_results.items() if 'Activation' in k}
exp4 = {k:v for k,v in all_results.items() if 'Optimizer' in k}

if exp1: plot_p1(exp1, 'Architecture Comparison', 'p1_architecture.png')
if exp2: plot_p1(exp2, 'Loss Function Comparison', 'p1_loss_functions.png')
if exp3: plot_p1(exp3, 'Activation Function Comparison', 'p1_activations.png')
if exp4: plot_p1(exp4, 'Optimizer Comparison', 'p1_optimizers.png')

# Summary bar chart
fig, ax = plt.subplots(figsize=(14, 6))
all_items = list(all_results.items())
names = [n[:40] for n,_ in all_items]
vals = [h['best_val_acc'] for _,h in all_items]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='black')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.4f}', ha='center', fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
ax.set_ylabel('Best Validation Accuracy'); ax.set_title('PJ2: Complete Results Summary'); ax.set_ylim(0,1.0); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','summary.png'),dpi=150); plt.close()
print("Saved: summary.png")

# Save JSON
with open(os.path.join(REPORTS_DIR, 'experiment_results.json'), 'w') as f:
    json.dump({k: {'best_val_acc': v['best_val_acc'], 'best_epoch': v['best_epoch'],
                    'final_train_acc': v['train_acc'][-1], 'final_val_acc': v['val_acc'][-1]}
               for k,v in all_results.items()}, f, indent=2)
print(f"\nJSON saved to: {os.path.join(REPORTS_DIR, 'experiment_results.json')}")

# Print summary
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
for name, hist in all_results.items():
    print(f"  {name}: Best Val Acc = {hist['best_val_acc']:.4f} (epoch {hist['best_epoch']})")
print("\nDONE! All figures in reports/figures/")
