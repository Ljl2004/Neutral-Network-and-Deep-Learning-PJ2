"""
Fast experiment runner - uses data subset for CPU training.
Gets real results quickly for the report.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np
import os, sys, json, random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(os.path.join(REPORTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, 'models'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
DATASET_SIZE = 10000  # Use subset for fast CPU training
BATCH_SIZE = 64
DATA_DIR = os.path.join(BASE_DIR, 'codes', 'VGG_BatchNorm', 'data')

# ---- Data ----
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
full_val = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

# Use subset for faster training
indices = np.random.RandomState(42).choice(len(full_train), DATASET_SIZE, replace=False)
train_set = Subset(full_train, indices)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(full_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_set)} samples ({len(train_loader)} batches), Val: {len(full_val)} samples")

# ---- Models (inlined for speed) ----
class VGG_A(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),nn.ReLU(True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,n_class))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): x=self.features(x); return self.classifier(x.view(x.size(0),-1))

class VGG_A_BN(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(True),
            nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(True),nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(True),nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,n_class))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): x=self.features(x); return self.classifier(x.view(x.size(0),-1))

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__()
        self.c1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(out_ch)
        self.c2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False); self.b2=nn.BatchNorm2d(out_ch)
        self.sc=nn.Sequential()
        if stride!=1 or in_ch!=out_ch: self.sc=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride,bias=False),nn.BatchNorm2d(out_ch))
    def forward(self,x):
        o=torch.relu(self.b1(self.c1(x))); o=self.b2(self.c2(o)); o+=self.sc(x); return torch.relu(o)

class CustomCNN(nn.Module):
    def __init__(self, filters=(64,128,256), fc_size=256, n_class=10, act='relu', dr=0.3):
        super().__init__()
        af = nn.ReLU(inplace=True) if act=='relu' else nn.LeakyReLU(0.1,inplace=True)
        stages=[]; ic=3
        for si,(f,_) in enumerate([(f,2) for f in filters]):
            for _ in range(2):
                stages.append(nn.Conv2d(ic,f,3,padding=1,bias=False))
                stages.append(nn.BatchNorm2d(f))
                stages.append(nn.ReLU(inplace=True) if act=='relu' else nn.LeakyReLU(0.1,inplace=True))
                ic=f
            if si<len(filters)-1: stages.append(nn.MaxPool2d(2,2))
        stages.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features=nn.Sequential(*stages)
        self.classifier=nn.Sequential(nn.Dropout(dr),nn.Linear(filters[-1],fc_size),nn.ReLU(inplace=True),nn.Dropout(dr),nn.Linear(fc_size,n_class))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): x=self.features(x); return self.classifier(x.view(x.size(0),-1))

class CustomResNet(nn.Module):
    def __init__(self, filters=(64,128,256), fc_size=256, n_class=10, dr=0.3):
        super().__init__()
        self.c1=nn.Conv2d(3,filters[0],3,1,1,bias=False); self.b1=nn.BatchNorm2d(filters[0])
        self.stages=nn.ModuleList(); ic=filters[0]
        for si,f in enumerate(filters):
            stage=[]
            for bi in range(2):
                stride=2 if bi==0 and si>0 else 1
                stage.append(ResBlock(ic,f,stride)); ic=f
            self.stages.append(nn.Sequential(*stage))
        self.ap=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Dropout(dr),nn.Linear(filters[-1],fc_size),nn.ReLU(inplace=True),nn.Dropout(dr),nn.Linear(fc_size,n_class))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x):
        x=torch.relu(self.b1(self.c1(x)))
        for s in self.stages: x=s(x)
        x=self.ap(x); return self.classifier(x.view(x.size(0),-1))

def count_p(m): return sum(p.numel() for p in m.parameters())

# ---- Training ----
def set_seed(s=42):
    np.random.seed(s); torch.manual_seed(s); random.seed(s)

def train_one(model, opt, sch, crit, epochs, name):
    model.to(device)
    hist={'tl':[],'ta':[],'vl':[],'va':[],'best_va':0,'best_ep':0}
    sp=os.path.join(REPORTS_DIR,'models',f'{name}.pth')
    for ep in range(epochs):
        model.train(); tl,c,t,b=0,0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            p=model(x); l=crit(p,y); l.backward(); opt.step()
            tl+=l.item(); b+=1; c+=p.max(1)[1].eq(y).sum().item(); t+=y.size(0)
        ta=c/max(t,1); tla=tl/max(b,1)
        model.eval(); vl,c,t,b=0,0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); p=model(x)
                vl+=crit(p,y).item(); b+=1; c+=p.max(1)[1].eq(y).sum().item(); t+=y.size(0)
        va=c/max(t,1); vla=vl/max(b,1); sch.step()
        hist['tl'].append(tla); hist['ta'].append(ta); hist['vl'].append(vla); hist['va'].append(va)
        if va>hist['best_va']: hist['best_va']=va; hist['best_ep']=ep; torch.save(model.state_dict(),sp)
        print(f"  Ep{ep+1}: TL={tla:.3f} TA={ta:.3f} VL={vla:.3f} VA={va:.3f}")
    return hist

EPOCHS=8
all_res={}
criterion=nn.CrossEntropyLoss()

# ============ PART 2: BN Experiments ============
print("\n"+"="*60)
print("PART 2: BATCH NORMALIZATION")
print("="*60)

for label, mcls in [('VGG-A',VGG_A),('VGG-A+BN',VGG_A_BN)]:
    print(f"\nTraining {label}...")
    set_seed(42); m=mcls()
    print(f"  Params: {count_p(m):,}")
    opt=optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    all_res[label]=train_one(m,opt,sch,criterion,EPOCHS,label.replace(' ','_'))

# Loss landscape
print("\nLoss Landscape...")
lrs=[1e-3,2e-3,1e-4,5e-4]; landscape={}
for mn, mcls in [('VGG-A',VGG_A),('VGG-A+BN',VGG_A_BN)]:
    all_losses=[]
    for lr in lrs:
        set_seed(42); m=mcls().to(device); opt=optim.SGD(m.parameters(),lr=lr,momentum=0.9)
        sl=[]; sc=0; m.train()
        for data in train_loader:
            if sc>=25: break
            x,y=data[0].to(device),data[1].to(device); opt.zero_grad()
            l=criterion(m(x),y); l.backward(); opt.step()
            sl.append(l.item()); sc+=1
        all_losses.append(sl)
    ns=min(len(l) for l in all_losses); mnc,mxc=[],[]
    for s in range(ns):
        vs=[all_losses[i][s] for i in range(len(lrs))]
        mnc.append(np.min(vs)); mxc.append(np.max(vs))
    landscape[f'{mn}_min']=mnc; landscape[f'{mn}_max']=mxc

# ============ PART 1: Custom CNN ============
print("\n"+"="*60)
print("PART 1: CUSTOM CNN EXPERIMENTS")
print("="*60)

# Exp1: Architecture
print("\n--- Exp1: Architecture ---")
for name, m in [
    ('CNN-Small',CustomCNN(filters=(32,64,128),fc_size=128)),
    ('CNN-Medium',CustomCNN(filters=(64,128,256),fc_size=256)),
    ('CNN-Large',CustomCNN(filters=(64,128,256,512),fc_size=512)),
    ('ResNet-Medium',CustomResNet(filters=(64,128,256),fc_size=256)),
]:
    print(f"\n  {name} ({count_p(m):,}p)")
    set_seed(42); opt=optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    all_res[f'{name} ({count_p(m):,}p)']=train_one(m,opt,sch,criterion,EPOCHS,name)

# Exp2: Loss functions
print("\n--- Exp2: Loss Functions ---")
for label, crit, wd in [
    ('CE (no reg)',nn.CrossEntropyLoss(),0),
    ('CE+L2(5e-4)',nn.CrossEntropyLoss(),5e-4),
    ('CE+L2(1e-3)',nn.CrossEntropyLoss(),1e-3),
    ('CE+LabelSmooth',nn.CrossEntropyLoss(label_smoothing=0.1),5e-4),
]:
    print(f"\n  {label}")
    set_seed(42); m=CustomCNN(filters=(64,128,256),fc_size=256)
    opt=optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    all_res[label]=train_one(m,opt,sch,crit,EPOCHS,f'loss_{label[:10]}')

# Exp3: Activations
print("\n--- Exp3: Activations ---")
for act in ['relu','leaky_relu']:
    print(f"\n  Activation={act}")
    set_seed(42); m=CustomCNN(filters=(64,128,256),fc_size=256,act=act)
    opt=optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    all_res[f'Activation={act}']=train_one(m,opt,sch,criterion,EPOCHS,f'act_{act}')

# Exp4: Optimizers
print("\n--- Exp4: Optimizers ---")
for oname, ofn in [
    ('SGD+Momentum',lambda p: optim.SGD(p,lr=0.01,momentum=0.9,weight_decay=5e-4)),
    ('Adam',lambda p: optim.Adam(p,lr=0.001,weight_decay=5e-4)),
    ('AdamW',lambda p: optim.AdamW(p,lr=0.001,weight_decay=5e-4)),
]:
    print(f"\n  Optimizer={oname}")
    set_seed(42); m=CustomCNN(filters=(64,128,256),fc_size=256)
    opt=ofn(m.parameters()); sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    all_res[f'Optimizer={oname}']=train_one(m,opt,sch,criterion,EPOCHS,f'opt_{oname}')

# ============ PLOTS ============
print("\n"+"="*60)
print("GENERATING PLOTS")
print("="*60)

# BN training curves
fig,axes=plt.subplots(1,3,figsize=(18,5))
ep=np.arange(1,EPOCHS+1)
axes[0].plot(ep,all_res['VGG-A']['tl'],'r-',lw=2,label='VGG-A'); axes[0].plot(ep,all_res['VGG-A+BN']['tl'],'b-',lw=2,label='VGG-A+BN')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep,all_res['VGG-A']['ta'],'r-',lw=2,label='VGG-A'); axes[1].plot(ep,all_res['VGG-A+BN']['ta'],'b-',lw=2,label='VGG-A+BN')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Train Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(ep,all_res['VGG-A']['va'],'r-',lw=2,label=f"VGG-A (best:{all_res['VGG-A']['best_va']:.3f})")
axes[2].plot(ep,all_res['VGG-A+BN']['va'],'b-',lw=2,label=f"VGG-A+BN (best:{all_res['VGG-A+BN']['best_va']:.3f})")
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy'); axes[2].set_title('Val Accuracy'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_training.png'),dpi=150); plt.close()
print("Saved: bn_training.png")

# Loss landscape
fig,axes=plt.subplots(1,2,figsize=(16,6))
for idx,mn in enumerate(['VGG-A','VGG-A+BN']):
    s=np.arange(1,len(landscape[f'{mn}_min'])+1)
    axes[idx].plot(s,landscape[f'{mn}_max'],'r-',lw=1.5,label='Max'); axes[idx].plot(s,landscape[f'{mn}_min'],'b-',lw=1.5,label='Min')
    axes[idx].fill_between(s,landscape[f'{mn}_min'],landscape[f'{mn}_max'],alpha=0.3,color='purple')
    axes[idx].set_xlabel('Step'); axes[idx].set_ylabel('Loss'); axes[idx].set_title(f'Loss Landscape: {mn}')
    axes[idx].legend(); axes[idx].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_landscape.png'),dpi=150); plt.close()
print("Saved: bn_landscape.png")

# Combined landscape
fig,ax=plt.subplots(figsize=(10,6))
s1=np.arange(1,len(landscape['VGG-A_min'])+1); s2=np.arange(1,len(landscape['VGG-A+BN_min'])+1)
ax.fill_between(s1,landscape['VGG-A_min'],landscape['VGG-A_max'],alpha=0.2,color='red',label='VGG-A (no BN)')
ax.fill_between(s2,landscape['VGG-A+BN_min'],landscape['VGG-A+BN_max'],alpha=0.2,color='blue',label='VGG-A+BN')
ax.plot(s1,[(a+b)/2 for a,b in zip(landscape['VGG-A_min'],landscape['VGG-A_max'])],'r-',lw=2,label='VGG-A mean')
ax.plot(s2,[(a+b)/2 for a,b in zip(landscape['VGG-A+BN_min'],landscape['VGG-A+BN_max'])],'b-',lw=2,label='VGG-A+BN mean')
ax.set_xlabel('Training Step'); ax.set_ylabel('Loss'); ax.set_title('Loss Landscape: BN vs Without BN')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','bn_landscape_combined.png'),dpi=150); plt.close()
print("Saved: bn_landscape_combined.png")

# Part 1 plots
def plot_exp(results_dict, title, fname):
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for name,hist in results_dict.items():
        ep=np.arange(1,len(hist['tl'])+1)
        axes[0].plot(ep,hist['tl'],lw=1.5,label=name)
        axes[1].plot(ep,hist['va'],lw=1.5,label=f"{name} (best:{hist['best_va']:.3f})")
        axes[2].plot(ep,hist['vl'],lw=1.5,label=name)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title(f'{title}: Train Loss'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title(f'{title}: Val Acc'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss'); axes[2].set_title(f'{title}: Val Loss'); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures',fname),dpi=150); plt.close()
    print(f"Saved: {fname}")

exp1={k:v for k,v in all_res.items() if 'p)' in k}
exp2={k:v for k,v in all_res.items() if 'CE' in k or 'Label' in k}
exp3={k:v for k,v in all_res.items() if 'Activation' in k}
exp4={k:v for k,v in all_res.items() if 'Optimizer' in k}
if exp1: plot_exp(exp1,'Architecture','p1_arch.png')
if exp2: plot_exp(exp2,'Loss Functions','p1_loss.png')
if exp3: plot_exp(exp3,'Activations','p1_act.png')
if exp4: plot_exp(exp4,'Optimizers','p1_opt.png')

# Summary bar chart
fig,ax=plt.subplots(figsize=(16,6))
items=list(all_res.items())
names=[n[:45] for n,_ in items]; vals=[h['best_va'] for _,h in items]
colors=plt.cm.viridis(np.linspace(0.15,0.95,len(names)))
bars=ax.bar(range(len(names)),vals,color=colors,edgecolor='black')
for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,f'{val:.3f}',ha='center',fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=60,ha='right',fontsize=6)
ax.set_ylabel('Best Val Accuracy'); ax.set_title('PJ2 Complete Results'); ax.set_ylim(0,1.0); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR,'figures','summary.png'),dpi=150); plt.close()
print("Saved: summary.png")

# Save JSON
with open(os.path.join(REPORTS_DIR,'results.json'),'w') as f:
    json.dump({k:{'best_va':v['best_va'],'best_ep':v['best_ep'],'final_ta':v['ta'][-1],'final_va':v['va'][-1]} for k,v in all_res.items()},f,indent=2)
print(f"Saved: results.json")

# Print summary
print("\n"+"="*60)
print("FINAL RESULTS")
print("="*60)
for k,v in all_res.items():
    print(f"  {k}: Best Val Acc = {v['best_va']:.4f} (ep {v['best_ep']})")
print("\nDONE!")
