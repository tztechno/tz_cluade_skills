```
 

```
```


```
  
**Wellbore Geology TCN**  
**T4*2 2026/05/10 10:21**  
```


```
  
++[https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271)++  
++[https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)++  
```

# =============================================================================
```
```


```
```
# Wellbore Geology TCN — v2 Improved
```
```


```
```
#
# Key improvements from v1 (Resolving 3 fundamental issues):
#   [Fix 1] Target (TVT) Normalization — Reduces MSE loss from 1,882,969 to ~1.0 scale
```
```


```
```
#   [Fix 2] Weighted Loss (using 'weight' column) — Aligns with competition evaluation metrics
#   [Fix 3] Added 'diff' features — Captures the rate of change in geological sequences
#
```
```


```
```
#   [Minor] Fixed torch.amp API deprecation warnings
```
```


```
```
#   [Minor] Expanded model architecture to [64, 128, 256]
```
```


```
```
#   [Minor] Increased well count from 400 to all wells (subset option commented out)
```
```


```
```
# =============================================================================
```
```


```
```

import
```
```
 numpy as np

```
```
import pandas as pd
import
```
```
 torch

```
```
import
```
```
 torch.nn as nn

```
```
import
```
```
 torch.optim as optim

```
```
from torch.utils.data import DataLoader, Dataset
from
```
```
 sklearn.preprocessing import StandardScaler

```
```
from
```
```
 sklearn.model_selection import GroupKFold

```
```
import
```
```
 glob

```
```
import
```
```
 os

```
```
import
```
```
 gc

```
```
import
```
```
 copy

```
```

# =============================================================================
```
```


```
```
# 1. TCN Model Definition
```
```


```
```
# =============================================================================
```
```


```
```

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        
```
```
super().__init__()

```
```
        self.chomp_size = chomp_size

    def forward(self, x):
        
```
```
return x[:, :, :-self.chomp_size].contiguous()

```
```


class CausalConv1dBlock(nn.Module):
    
```
```
def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):

```
```
        
```
```
super().__init__()

```
```
        self.conv = nn.utils.parametrize.register_parametrization if False else \
            nn
```
```
.utils.weight_norm(

```
```
                nn.Conv1d(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
            )
        
```
```
self.chomp = Chomp1d(padding)

```
```
        
```
```
self.relu  = nn.ReLU()

```
```
        self.net   = nn.Sequential(self.conv, self.chomp, self.relu)

    
```
```
def forward(self, x):

```
```
        
```
```
return self.net(x)

```
```


class
```
```
 TCNBlock(nn.Module):

```
```
    
```
```
def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):

```
```
        
```
```
super().__init__()

```
```
        padding       
```
```
= (kernel_size - 1) * dilation

```
```
        
```
```
self.conv1    = CausalConv1dBlock(n_inputs,   n_outputs, kernel_size, stride, dilation, padding)

```
```
        
```
```
self.dropout1 = nn.Dropout(dropout)

```
```
        
```
```
self.conv2    = CausalConv1dBlock(n_outputs,  n_outputs, kernel_size, stride, dilation, padding)

```
```
        
```
```
self.dropout2 = nn.Dropout(dropout)

```
```
        
```
```
self.net      = nn.Sequential(self.conv1, self.dropout1, self.conv2, self.dropout2)

```
```
        
```
```
self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

```
```
        self.relu     = nn.ReLU()

    
```
```
def forward(self, x):

```
```
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class
```
```
 TCNModel(nn.Module):

```
```
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers 
```
```
= []

```
```
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch  = input_size        if i == 0 else num_channels[i - 1]
            out_ch 
```
```
= num_channels[i]

```
```
            layers
```
```
.append(TCNBlock(in_ch, out_ch, kernel_size, 1, dilation_size, dropout))

```
```
        
```
```
self.tcn    = nn.Sequential(*layers)

```
```
        
```
```
self.linear = nn.Linear(num_channels[-1], output_size)

```
```

    
```
```
def forward(self, x):

```
```
        
```
```
return self.linear(self.tcn(x)[:, :, -1])

```
```

# =============================================================================
```
```


```
```
# 2. Dataset  ← [Fix 2] Modified to return the weight column
# =============================================================================

class
```
```
 WellSequenceDataset(Dataset):

```
```
    
```
```
"""Pre-generates sliding windows. Returns (x, y_normalized, weight)."""

```
```

    
```
```
def __init__(self, sequences_data, window_size=64):

```
```
        
```
```
self.samples     = []

```
```
        self.window_size = window_size

        for seq in sequences_data:
            X = seq['X']   # (T, F)
            y = seq['y']   # (T,)  ← Normalized target
            w 
```
```
= seq['w']   # (T,)  ← Weight column

```
```
            T 
```
```
= len(X)

```
```
            for i in range(T - window_size + 1):
                x_win    
```
```
= X[i:i + window_size].T           # (F, W)

```
```
                y_target 
```
```
= y[i + window_size - 1]

```
```
                w_target = w[i + window_size - 1]          # ← Added
                self.samples.append((x_win, y_target, w_target))

    
```
```
def __len__(self):

```
```
        
```
```
return len(self.samples)

```
```

    
```
```
def __getitem__(self, idx):

```
```
        x, y, w 
```
```
= self.samples[idx]

```
```
        
```
```
return (torch.from_numpy(x.copy()),

```
```
                torch
```
```
.tensor([y], dtype=torch.float32),

```
```
                torch
```
```
.tensor([w], dtype=torch.float32))  # ← Returning weights

```
```

# =============================================================================
```
```


```
```
# 3. Helpers
```
```


```
```
# =============================================================================
```
```


```
```

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred 
```
```
= np.asarray(y_pred, dtype=np.float64)

```
```
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# [Fix 2] Weighted Loss Function
```
```


```
```
def weighted_mse_loss(pred, target, weight):
    
```
```
"""Weighted MSE Loss. Matches competition metric (weighted RMSE)."""

```
```
    
```
```
return (weight * (pred - target).pow(2)).mean()

```
```


def _predict_well_batched(model, seq, window_size, batch_size=512, device=None, use_amp=False):
    """
    Batched sliding window inference (for one well).
    Returns values in normalized scale. Must call inverse_transform in the caller.
```
```


```
```
    """
    
```
```
if device is None:

```
```
        device 
```
```
= next(model.parameters()).device

```
```

    X 
```
```
= seq["X"]

```
```
    y = seq["y"] if seq["y"] is not None else np.zeros(len(seq["X"]), np.float32)
    T      
```
```
= len(X)

```
```
    n_wins 
```
```
= T - window_size + 1

```
```

    idx     = np.arange(window_size)[None, :] + np.arange(n_wins)[:, None]
    windows 
```
```
= X[idx].transpose(0, 2, 1).astype(np.float32)   # (n_wins, F, W)

```
```

    preds 
```
```
= np.empty(n_wins, dtype=np.float32)

```
```
    model
```
```
.eval()

```
```
    with torch.no_grad():
        
```
```
for start in range(0, n_wins, batch_size):

```
```
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device, non_blocking=True)
            
```
```
with torch.amp.autocast('cuda', enabled=use_amp):   # ← Fixed deprecation warning

```
```
                out = model(batch).squeeze(-1)
            preds[start:start 
```
```
+ batch_size] = out.cpu().numpy()

```
```

    well_preds 
```
```
= np.empty(T, dtype=np.float32)

```
```
    well_preds[window_size 
```
```
- 1:] = preds

```
```
    well_preds[:window_size 
```
```
- 1] = preds[0]

```
```
    
```
```
return well_preds, y

```
```

# =============================================================================
```
```


```
```
# 4. Configuration
# =============================================================================
```
```


```
```

BASE             = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
RAW_FEATURE_COLS 
```
```
= ["X", "Y", "Z", "GR", "MD"]

```
```
TARGET           = "TVT_input"
WEIGHT_COL       
```
```
= "weight"

```
```

WINDOW_SIZE    
```
```
= 64

```
```
BATCH_SIZE     = 128
EPOCHS         = 15          # Can be increased as convergence is stable after normalization
PATIENCE       = 5
N_FOLDS        
```
```
= 5

```
```
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   
```
```
= 1e-4

```
```
GRAD_CLIP      = 1.0
NUM_WORKERS    = 2

DEVICE  
```
```
= torch.device("cuda" if torch.cuda.is_available() else "cpu")

```
```
USE_AMP = torch.cuda.is_available()

print
```
```
(f"Device : {DEVICE}")

```
```
print(f"AMP    : {USE_AMP}")

# =============================================================================
# 5. Data Loading
```
```


```
```
# =============================================================================
```
```


```
```

def
```
```
 load_wells(directory, max_files=None):

```
```
    pattern 
```
```
= os.path.join(directory, "*__horizontal_well.csv")

```
```
    files   
```
```
= sorted(glob.glob(pattern))

```
```
    if max_files:
        files = files[:max_files]
    dfs = []
    
```
```
for f in files:

```
```
        df              = pd.read_csv(f)
        well_name       = os.path.basename(f).replace("__horizontal_well.csv", "")
        df["well"]      = well_name
        df["row_index"] = range(len(df))
        dfs
```
```
.append(df)

```
```
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print
```
```
("Loading train data...")

```
```
train = load_wells(os.path.join(BASE, "train"))
print
```
```
(f"  train shape: {train.shape}")

```
```

# Uncomment to use a subset (default uses all wells)
```
```


```
```
# well_ids     = train["well"].unique()
```
```


```
```
# subset_wells = well_ids[:400]
# train        = train[train["well"].isin(subset_wells)].copy()
```
```


```
```

print("Loading test data...")
test = load_wells(os.path.join(BASE, "test"))
print
```
```
(f"  test shape : {test.shape}")

```
```

# =============================================================================
# 6. STEP 1 — Scaling Raw Features (applied before lag/rolling)
```
```


```
```
# =============================================================================
```
```


```
```

print
```
```
("\n" + "="*60)

```
```
print("STEP 1: Raw feature scaling")
print
```
```
("="*60)

```
```

train_labeled 
```
```
= train[train[TARGET].notna()].copy()

```
```
scaler_raw    
```
```
= StandardScaler()

```
```
scaler_raw
```
```
.fit(train_labeled[RAW_FEATURE_COLS].fillna(0))

```
```

def
```
```
 scale_raw_features(df, scaler, raw_cols):

```
```
    scaled 
```
```
= scaler.transform(df[raw_cols].fillna(0))

```
```
    for i, col in enumerate(raw_cols):
        df[col] 
```
```
= scaled[:, i]

```
```
    
```
```
return df

```
```

train 
```
```
= scale_raw_features(train, scaler_raw, RAW_FEATURE_COLS)

```
```
test  = scale_raw_features(test,  scaler_raw, RAW_FEATURE_COLS)
print
```
```
("Raw features scaled (mean=0, std=1)")

```
```

# =============================================================================
```
```


```
```
# [Fix 1] STEP 1.5 — Target (TVT) Normalization
#   Problem: MSE loss was ~1,882,969 in Ep1 due to non-normalized TVT values.
```
```


```
```
#   Fix: Normalize TVT to N(0,1) using StandardScaler to keep loss around ~1.0 scale.
#   Effect: Stabilizes training, prevents gradient explosion, and drastically improves OOF RMSE.
```
```


```
```
# =============================================================================

print("\n" + "="*60)
print("STEP 1.5: [NEW] Target normalization")
print("="*60)

target_scaler = StandardScaler()
target_scaler.fit(train_labeled[[TARGET]])

def normalize_target_col(df, scaler, target_col):
    mask = df[target_col].notna()
    df
```
```
.loc[mask, target_col] = scaler.transform(df.loc[mask, [target_col]]).ravel()

```
```
    
```
```
return df

```
```

train = normalize_target_col(train, target_scaler, TARGET)
print
```
```
(f"Target normalized: mean={train[TARGET].mean():.4f}, std={train[TARGET].std():.4f}")

```
```
print(f"(Predictions will be inverse-transformed back to original scale)")

# =============================================================================
# [Fix 3] STEP 2a — Added 'diff' features (capturing rate of change)
```
```


```
```
#   Problem: Lag/rolling features alone missed the rate of change along depth.
```
```


```
```
#   Fix: Added diff1 (1-step difference) and diff2 (2-step difference).
```
```


```
```
#   Effect: Represents slope/acceleration between adjacent depths, improving boundary detection.
```
```


```
```
# =============================================================================
```
```


```
```

print
```
```
("\n" + "="*60)

```
```
print("STEP 2a: [NEW] Diff features (rate of change)")
print
```
```
("="*60)

```
```

DIFF_STEPS 
```
```
= [1, 2]

```
```

def add_diff_features(df, feature_cols, diff_steps):
    df      = df.sort_values(["well", "MD"]).copy()
    new_cols = []
    
```
```
for col in feature_cols:

```
```
        grp = df.groupby("well")[col]
        for d in diff_steps:
            cname    
```
```
= f"{col}_diff{d}"

```
```
            df[cname] 
```
```
= grp.diff(d).fillna(0)

```
```
            new_cols
```
```
.append(cname)

```
```
    return df, new_cols

train, diff_cols = add_diff_features(train, RAW_FEATURE_COLS, DIFF_STEPS)
test,  _         = add_diff_features(test,  RAW_FEATURE_COLS, DIFF_STEPS)
print
```
```
(f"Diff features added: {len(diff_cols)}")

```
```

# =============================================================================
```
```


```
```
# 7. STEP 2b — Lag / Rolling Features (applied to scaled data)
```
```


```
```
# =============================================================================
```
```


```
```

print
```
```
("\n" + "="*60)

```
```
print
```
```
("STEP 2b: Lag/rolling features")

```
```
print("="*60)

LAG_STEPS    = (1, 2, 3, 5)
ROLL_WINDOWS 
```
```
= [3, 5, 10]

```
```

def
```
```
 add_lag_roll_features(df, feature_cols, lag_steps, roll_windows):

```
```
    df       = df.sort_values(["well", "MD"]).copy()
    new_cols 
```
```
= []

```
```
    
```
```
for col in feature_cols:

```
```
        grp 
```
```
= df.groupby("well")[col]

```
```
        
```
```
for lag in lag_steps:

```
```
            cname     = f"{col}_lag{lag}"
            df[cname] = grp.shift(lag)
            new_cols.append(cname)
        for win in roll_windows:
            cname     = f"{col}_roll{win}"
            df[cname] 
```
```
= grp.shift(1).transform(lambda x: x.rolling(win, min_periods=1).mean())

```
```
            new_cols
```
```
.append(cname)

```
```
    
```
```
return df, new_cols

```
```

print("Building lag/rolling features for train...")
train, lag_cols 
```
```
= add_lag_roll_features(train, RAW_FEATURE_COLS, LAG_STEPS, ROLL_WINDOWS)

```
```
print
```
```
(f"  lag/rolling features added: {len(lag_cols)}")

```
```
gc
```
```
.collect()

```
```

print
```
```
("Building lag/rolling features for test...")

```
```
test, _  = add_lag_roll_features(test, RAW_FEATURE_COLS, LAG_STEPS, ROLL_WINDOWS)
gc.collect()

ALL_FEATURES = RAW_FEATURE_COLS + diff_cols + lag_cols   # ← Added diff_cols
missing      
```
```
= [c for c in ALL_FEATURES if c not in test.columns]

```
```
if
```
```
 missing:

```
```
    
```
```
print(f"[WARNING] Dropping {len(missing)} cols missing from test: {missing}")

```
```
    ALL_FEATURES 
```
```
= [c for c in ALL_FEATURES if c not in missing]

```
```

# Fill NaN (e.g., first rows of lag features)
```
```


```
```
train[ALL_FEATURES] 
```
```
= train[ALL_FEATURES].fillna(0)

```
```
test[ALL_FEATURES]  
```
```
= test[ALL_FEATURES].fillna(0)

```
```

print(f"\nTotal features: {len(ALL_FEATURES)}")
print
```
```
(f"  = {len(RAW_FEATURE_COLS)} raw + {len(diff_cols)} diff + {len(lag_cols)} lag/roll")

```
```

# =============================================================================
# 8. STEP 3 — Create Well Sequences  ← [Fix 2] Added weight column
```
```


```
```
# =============================================================================
```
```


```
```

print
```
```
("\n" + "="*60)

```
```
print("STEP 3: Per-well sequences")
print("="*60)

def
```
```
 create_well_sequence(df, well_id, feature_cols, target_col, weight_col, window_size=64):

```
```
    well_data = df[df["well"] == well_id].sort_values("MD")
    X = well_data[feature_cols].fillna(0).values.astype(np.float32)
    y 
```
```
= (well_data[target_col].fillna(0).values.astype(np.float32)

```
```
         
```
```
if target_col and target_col in well_data.columns

```
```
         else np.zeros(len(X), np.float32))
    
```
```
# [Fix 2] Extract weight column (fill with 1.0 if missing)

```
```
    w 
```
```
= (well_data[weight_col].fillna(1.0).values.astype(np.float32)

```
```
         
```
```
if weight_col and weight_col in well_data.columns

```
```
         
```
```
else np.ones(len(X), np.float32))

```
```
    return {"well_id": well_id, "X": X, "y": y, "w": w, "length": len(X)}

train_wells     
```
```
= train["well"].unique()

```
```
train_sequences 
```
```
= []

```
```
for
```
```
 well_id in train_wells:

```
```
    seq = create_well_sequence(train, well_id, ALL_FEATURES, TARGET, WEIGHT_COL, WINDOW_SIZE)
    
```
```
if seq["length"] >= WINDOW_SIZE:

```
```
        train_sequences
```
```
.append(seq)

```
```

print
```
```
(f"Wells with sufficient length : {len(train_sequences)}")

```
```
print
```
```
(f"Wells dropped (too short)    : {len(train_wells) - len(train_sequences)}")

```
```

# =============================================================================
# 9. STEP 4 — GroupKFold Cross-Validation
```
```


```
```
# =============================================================================

print
```
```
("\n" + "="*60)

```
```
print
```
```
("STEP 4: GroupKFold Cross-Validation (grouped by well)")

```
```
print("="*60)

well_ids = np.array([seq["well_id"] for seq in train_sequences])
gkf      = GroupKFold(n_splits=N_FOLDS)

fold_models          = []
fold_oof_predictions = []
fold_scores          
```
```
= []

```
```


def train_fold(fold_idx, train_seqs, val_seqs):
    """
    [Fix 1] Uses weighted_mse_loss (normalized target + weight column)
    [Fix 2] Dataset returns (x, y, weight), applying weights to the loss calculation
    """
```
```


```
```
    train_dataset 
```
```
= WellSequenceDataset(train_seqs, WINDOW_SIZE)

```
```
    val_dataset   
```
```
= WellSequenceDataset(val_seqs,   WINDOW_SIZE)

```
```

    train_loader 
```
```
= DataLoader(train_dataset, batch_size=BATCH_SIZE,

```
```
                              shuffle
```
```
=True,  num_workers=NUM_WORKERS, pin_memory=True)

```
```
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,
                              shuffle
```
```
=False, num_workers=NUM_WORKERS, pin_memory=True)

```
```

    # [Minor] Expanded model to [64, 128, 256] (minimal impact on computation time)
    model = TCNModel(
        input_size   = len(ALL_FEATURES),
        output_size  
```
```
= 1,

```
```
        num_channels 
```
```
= [64, 128, 256],   # v1: [32, 64, 128]

```
```
        kernel_size  
```
```
= 3,

```
```
        dropout      
```
```
= 0.1,

```
```
    ).to(DEVICE)

    optimizer  
```
```
= optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

```
```
    scheduler  
```
```
= optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

```
```
    scaler_amp = torch.amp.GradScaler('cuda', enabled=USE_AMP)   # ← Fix deprecation

    best_val_loss, best_state, patience_ctr 
```
```
= float('inf'), None, 0

```
```

    
```
```
print(f"\n  Fold {fold_idx} — {len(train_dataset):,} train / {len(val_dataset):,} val windows")

```
```

    for epoch in range(1, EPOCHS + 1):
        
```
```
# ── Training ────────────────────────────────────────────────

```
```
        model
```
```
.train()

```
```
        train_loss 
```
```
= 0.0

```
```
        
```
```
for batch_x, batch_y, batch_w in train_loader:   # ← Added batch_w

```
```
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y 
```
```
= batch_y.to(DEVICE, non_blocking=True).squeeze()

```
```
            batch_w = batch_w.to(DEVICE, non_blocking=True).squeeze()   # ← Weight
            optimizer
```
```
.zero_grad(set_to_none=True)

```
```
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                pred 
```
```
= model(batch_x).squeeze()

```
```
                loss = weighted_mse_loss(pred, batch_y, batch_w)   # ← [Fix 2]
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp
```
```
.update()

```
```
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss, val_preds_norm, val_targets_norm = 0.0, [], []
        
```
```
with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):

```
```
            
```
```
for batch_x, batch_y, batch_w in val_loader:

```
```
                batch_x 
```
```
= batch_x.to(DEVICE, non_blocking=True)

```
```
                batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
                batch_w = batch_w.to(DEVICE, non_blocking=True).squeeze()
                pred     = model(batch_x).squeeze()
                val_loss += weighted_mse_loss(pred, batch_y, batch_w).item()
                val_preds_norm.extend(pred.cpu().numpy())
                val_targets_norm
```
```
.extend(batch_y.cpu().numpy())

```
```

        train_loss 
```
```
/= len(train_loader)

```
```
        val_loss   /= len(val_loader)

        
```
```
# [Fix 1] Inverse transform to show real-scale RMSE

```
```
        val_preds_real   = target_scaler.inverse_transform(
            np
```
```
.array(val_preds_norm).reshape(-1, 1)).ravel()

```
```
        val_targets_real = target_scaler.inverse_transform(
            np
```
```
.array(val_targets_norm).reshape(-1, 1)).ravel()

```
```
        val_rmse_real    = rmse(val_targets_real, val_preds_real)

        scheduler.step()
        
```
```
print(f"    Ep {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f}"

```
```
              
```
```
f" | val_loss {val_loss:.4f} | RMSE {val_rmse_real:.4f}")

```
```

        
```
```
# ── Early Stopping ─────────────────────────────────────────────

```
```
        
```
```
if val_loss < best_val_loss:

```
```
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  
```
```
= 0

```
```
        
```
```
else:

```
```
            patience_ctr += 1
            
```
```
if patience_ctr >= PATIENCE:

```
```
                
```
```
print(f"    Early stop triggered at epoch {epoch}")

```
```
                break

    model
```
```
.load_state_dict(best_state)

```
```

    
```
```
# ── OOF Inference (Batched) → inverse_transform ─────────────────

```
```
    oof_preds, oof_targets, oof_wells = [], [], []
    
```
```
for seq in val_seqs:

```
```
        preds_norm, targets_norm 
```
```
= _predict_well_batched(

```
```
            model, seq, WINDOW_SIZE, batch_size
```
```
=BATCH_SIZE * 2, device=DEVICE, use_amp=USE_AMP)

```
```
        mask 
```
```
= targets_norm != 0

```
```
        
```
```
# Return to real scale

```
```
        preds_real   = target_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()
        targets_real = target_scaler.inverse_transform(targets_norm.reshape(-1, 1)).ravel()
        oof_preds.extend(preds_real[mask].tolist())
        oof_targets
```
```
.extend(targets_real[mask].tolist())

```
```
        oof_wells.extend([seq["well_id"]] * int(mask.sum()))

    return model, oof_preds, oof_targets, oof_wells


# ── Run CV ────────────────────────────────────────────────────────────────────
```
```


```
```
for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_sequences, groups=well_ids), 1):

    
```
```
print(f"\n{'='*50}")

```
```
    
```
```
print(f"FOLD {fold}/{N_FOLDS}")

```
```
    
```
```
print(f"{'='*50}")

```
```

    train_seqs 
```
```
= [train_sequences[i] for i in train_idx]

```
```
    val_seqs   
```
```
= [train_sequences[i] for i in val_idx]

```
```
    print(f"Train wells: {len(train_seqs)} | Val wells: {len(val_seqs)}")

    model, oof_preds, oof_targets, oof_wells = train_fold(fold, train_seqs, val_seqs)

    fold_rmse = rmse(oof_targets, oof_preds)
    fold_scores.append(fold_rmse)
    fold_models
```
```
.append(model)

```
```

    fold_oof_predictions.append(pd.DataFrame({
        
```
```
"well_id":    oof_wells,

```
```
        "target":     oof_targets,
        
```
```
"prediction": oof_preds,

```
```
        "fold":       fold,
    }))

    
```
```
print(f"\n  Fold {fold} RMSE (real scale): {fold_rmse:.4f}")

```
```

    gc
```
```
.collect()

```
```
    if torch.cuda.is_available():
        torch
```
```
.cuda.empty_cache()

```
```


oof_all      
```
```
= pd.concat(fold_oof_predictions, ignore_index=True)

```
```
overall_rmse 
```
```
= rmse(oof_all["target"], oof_all["prediction"])

```
```

print
```
```
("\n" + "="*60)

```
```
print("CROSS-VALIDATION RESULTS")
print
```
```
("="*60)

```
```
print
```
```
(f"Per-fold RMSEs  : {[f'{s:.4f}' for s in fold_scores]}")

```
```
print
```
```
(f"Mean fold RMSE  : {np.mean(fold_scores):.4f}  (+/- {np.std(fold_scores):.4f})")

```
```
print(f"Overall OOF RMSE: {overall_rmse:.4f}")

# =============================================================================
```
```


```
```
# 10. STEP 5 — Training Final Model on All Training Data
```
```


```
```
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Training final model on all training data")
print
```
```
("="*60)

```
```

final_dataset 
```
```
= WellSequenceDataset(train_sequences, WINDOW_SIZE)

```
```
final_loader  = DataLoader(final_dataset, batch_size=BATCH_SIZE,
                           shuffle
```
```
=True, num_workers=NUM_WORKERS, pin_memory=True)

```
```

final_model 
```
```
= TCNModel(

```
```
    input_size
```
```
=len(ALL_FEATURES), output_size=1,

```
```
    num_channels
```
```
=[64, 128, 256], kernel_size=3, dropout=0.1,

```
```
).to(DEVICE)

optimizer  = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler_amp = torch.amp.GradScaler('cuda', enabled=USE_AMP)

print
```
```
(f"Training on {len(final_dataset):,} windows...")

```
```

for epoch in range(1, EPOCHS + 1):
    final_model
```
```
.train()

```
```
    train_loss = 0.0
    for batch_x, batch_y, batch_w in final_loader:
        batch_x 
```
```
= batch_x.to(DEVICE, non_blocking=True)

```
```
        batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
        batch_w 
```
```
= batch_w.to(DEVICE, non_blocking=True).squeeze()

```
```
        optimizer.zero_grad(set_to_none=True)
        
```
```
with torch.amp.autocast('cuda', enabled=USE_AMP):

```
```
            pred 
```
```
= final_model(batch_x).squeeze()

```
```
            loss 
```
```
= weighted_mse_loss(pred, batch_y, batch_w)

```
```
        scaler_amp.scale(loss).backward()
        scaler_amp
```
```
.unscale_(optimizer)

```
```
        torch
```
```
.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP)

```
```
        scaler_amp
```
```
.step(optimizer)

```
```
        scaler_amp
```
```
.update()

```
```
        train_loss += loss.item()
    scheduler.step()
    
```
```
if epoch % 5 == 0:

```
```
        print(f"  Ep {epoch:2d}/{EPOCHS} | train loss: {train_loss/len(final_loader):.4f}")

print
```
```
("Final model training complete!")

```
```

# =============================================================================
# 11. STEP 6 — Inference on Test Set (Batched) → inverse_transform
# =============================================================================

print("\n" + "="*60)
print
```
```
("STEP 6: Inference on test set")

```
```
print("="*60)

test_wells     
```
```
= test["well"].unique()

```
```
test_sequences 
```
```
= []

```
```
for well_id in test_wells:
    seq = create_well_sequence(test, well_id, ALL_FEATURES, None, None, WINDOW_SIZE)
    if seq["length"] >= WINDOW_SIZE:
        test_sequences.append(seq)

print
```
```
(f"Test wells with sufficient length: {len(test_sequences)}")

```
```

test_predictions 
```
```
= []

```
```
final_model.eval()
for seq in test_sequences:
    well_id      
```
```
= seq["well_id"]

```
```
    well_test_df 
```
```
= test[test["well"] == well_id].sort_values("MD")

```
```
    row_indices  
```
```
= well_test_df["row_index"].values

```
```

    well_preds_norm, _ 
```
```
= _predict_well_batched(

```
```
        final_model, seq, WINDOW_SIZE,
        batch_size
```
```
=BATCH_SIZE * 2,

```
```
        device
```
```
=DEVICE, use_amp=USE_AMP,

```
```
    )
    
```
```
# [Fix 1] Revert normalization

```
```
    well_preds_real 
```
```
= target_scaler.inverse_transform(

```
```
        well_preds_norm.reshape(-1, 1)).ravel()

    for row_idx, pred_tvt in zip(row_indices, well_preds_real):
        test_predictions
```
```
.append({

```
```
            "well_id":   well_id,
            "row_index": row_idx,
            "tvt":       float(pred_tvt),
        })

test_pred_df = pd.DataFrame(test_predictions)
print(f"Generated {len(test_pred_df):,} test predictions")

# =============================================================================
```
```


```
```
# 12. STEP 7 — Generate Submission
```
```


```
```
# =============================================================================
```
```


```
```

sample_sub              = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
sample_sub[
```
```
"well"]      = sample_sub["id"].str.rsplit("_", n=1).str[0]

```
```
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)
sample_sub              = sample_sub.drop(columns=["tvt"], errors="ignore")

submission = sample_sub.merge(
    test_pred_df[[
```
```
"well_id", "row_index", "tvt"]],

```
```
    left_on
```
```
=["well", "row_index"],

```
```
    right_on
```
```
=["well_id", "row_index"],

```
```
    how
```
```
="left",

```
```
)
submission["tvt"] = submission["tvt"].fillna(
    submission
```
```
.groupby("well")["tvt"].transform("first")

```
```
)
submission = submission[["id", "tvt"]]
submission
```
```
.to_csv("submission.csv", index=False)

```
```
print
```
```
(f"\nSubmission saved: {len(submission):,} rows")

```
```
print(submission.head())

# =============================================================================
```
```


```
```
# 13. STEP 8 (Optional) — Ensemble Fold Models
# =============================================================================

print
```
```
("\n" + "="*60)

```
```
print
```
```
("STEP 8 (Optional): Ensemble predictions from all fold models")

```
```
print
```
```
("="*60)

```
```


def
```
```
 predict_with_ensemble(models, test_sequences, window_size=WINDOW_SIZE,

```
```
                          batch_size=BATCH_SIZE * 2, device=DEVICE, use_amp=USE_AMP):
    
```
```
"""Average predictions from all fold models (Ensemble after inverse_transform)"""

```
```
    all_fold_preds = []
    
```
```
for model in models:

```
```
        fold_rows 
```
```
= []

```
```
        for seq in test_sequences:
            well_id      = seq["well_id"]
            well_test_df 
```
```
= test[test["well"] == well_id].sort_values("MD")

```
```
            row_indices  
```
```
= well_test_df["row_index"].values

```
```
            well_preds_norm, _ 
```
```
= _predict_well_batched(

```
```
                model, seq, window_size, batch_size, device, use_amp)
            well_preds_real = target_scaler.inverse_transform(
                well_preds_norm.reshape(-1, 1)).ravel()
            for row_idx, pred_tvt in zip(row_indices, well_preds_real):
                fold_rows
```
```
.append({"well_id": well_id, "row_index": row_idx, "tvt": float(pred_tvt)})

```
```
        all_fold_preds.append(pd.DataFrame(fold_rows))

    combined      
```
```
= pd.concat(all_fold_preds)

```
```
    ensemble_pred 
```
```
= (combined

```
```
                     .groupby(["well_id", "row_index"])["tvt"]
                     
```
```
.mean()

```
```
                     
```
```
.reset_index())

```
```
    
```
```
return ensemble_pred

```
```


# To use the ensemble, uncomment the following:
```
```


```
```
# print("Running ensemble inference...")
# ensemble_test_pred = predict_with_ensemble(fold_models, test_sequences)
# ensemble_sub = sample_sub.merge(
```
```


```
```
#      ensemble_test_pred,
```
```


```
```
#      left_on=["well", "row_index"],
```
```


```
```
#      right_on=["well_id", "row_index"],
```
```


```
```
#      how="left",
# )
```
```


```
```
# ensemble_sub["tvt"] = ensemble_sub["tvt"].fillna(
```
```


```
```
#      ensemble_sub.groupby("well")["tvt"].transform("first")
# )
```
```


```
```
# ensemble_sub[["id", "tvt"]].to_csv("submission_ensemble.csv", index=False)
# print("Ensemble submission saved to submission_ensemble.csv")
```
```


```
```

print
```
```
("\n" + "="*60)

```
```
print
```
```
("DONE!")

```
```
print("="*60)

 

 

 
```
```


```
