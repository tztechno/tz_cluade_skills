```
 

```
```


```
  
## Wellbore Geology TCN++[¶](https://www.kaggle.com/code/stpeteishii/wellbore-geology-tcn?scriptVersionId=317973562#Wellbore-Geology-TCN)++  
**T4*2 2026/05/10 10:21**  
```


```
  
++[https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271)++  
++[https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)++  
++[https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)++  
```

# =============================================================================
```
```


```
```
# Wellbore Geology TCN — Optimized Version
```
```


```
```
# Key improvements over original:
#   1. Scale raw features BEFORE lag/rolling engineering (no leakage)
```
```


```
```
#   2. GroupKFold by well ID (no data leakage across wells)
```
```


```
```
#   3. Batched window inference          — ~30-50x faster inference
```
```


```
```
#   4. Automatic Mixed Precision (AMP)   — ~1.5-2x faster training on GPU
#   5. Early stopping                    — avoids wasted epochs
```
```


```
```
#   6. num_workers + pin_memory          — overlaps data loading with GPU
```
```


```
```
#   7. 3 folds instead of 5             — 40% less CV time
# =============================================================================

import
```
```
 numpy as np

```
```
import
```
```
 pandas as pd

```
```
import torch
import torch.nn as nn
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
# 1. TCN Model Definition
```
```


```
```
# =============================================================================

class
```
```
 Chomp1d(nn.Module):

```
```
    
```
```
def __init__(self, chomp_size):

```
```
        
```
```
super().__init__()

```
```
        self.chomp_size = chomp_size

    
```
```
def forward(self, x):

```
```
        
```
```
return x[:, :, :-self.chomp_size].contiguous()

```
```


class CausalConv1dBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        
```
```
super().__init__()

```
```
        
```
```
self.conv = nn.utils.weight_norm(

```
```
            nn
```
```
.Conv1d(n_inputs, n_outputs, kernel_size,

```
```
                      stride
```
```
=stride, padding=padding, dilation=dilation)

```
```
        )
        self.chomp = Chomp1d(padding)
        
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
        padding      = (kernel_size - 1) * dilation
        
```
```
self.conv1   = CausalConv1dBlock(n_inputs,   n_outputs, kernel_size, stride, dilation, padding)

```
```
        self.dropout1 = nn.Dropout(dropout)
        self.conv2   = CausalConv1dBlock(n_outputs, n_outputs, kernel_size, stride, dilation, padding)
        
```
```
self.dropout2 = nn.Dropout(dropout)

```
```
        self.net      = nn.Sequential(self.conv1, self.dropout1, self.conv2, self.dropout2)
        
```
```
self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

```
```
        
```
```
self.relu     = nn.ReLU()

```
```

    
```
```
def forward(self, x):

```
```
        out 
```
```
= self.net(x)

```
```
        res = x if self.downsample is None else self.downsample(x)
        
```
```
return self.relu(out + res)

```
```


class
```
```
 TCNModel(nn.Module):

```
```
    
```
```
def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):

```
```
        super().__init__()
        layers 
```
```
= []

```
```
        for i in range(len(num_channels)):
            dilation_size 
```
```
= 2 ** i

```
```
            in_ch  = input_size          if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 1, dilation_size, dropout))
        
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
# 2. Dataset
```
```


```
```
# =============================================================================
```
```


```
```

class
```
```
 WellSequenceDataset(Dataset):

```
```
    
```
```
"""Pre-builds all sliding windows into a flat list at construction time."""

```
```

    
```
```
def __init__(self, sequences_data, window_size=64):

```
```
        self.samples     = []
        
```
```
self.window_size = window_size

```
```

        
```
```
for seq in sequences_data:

```
```
            X 
```
```
= seq['X']   # (T, F)

```
```
            y 
```
```
= seq['y']   # (T,)

```
```
            T 
```
```
= len(X)

```
```
            
```
```
for i in range(T - window_size + 1):

```
```
                x_win    = X[i:i + window_size].T          # (F, W)
                y_target 
```
```
= y[i + window_size - 1]

```
```
                self.samples.append((x_win, y_target))

    
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

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        
```
```
return torch.from_numpy(x.copy()), torch.tensor([y], dtype=torch.float32)

```
```

# =============================================================================
```
```


```
```
# 3. Helpers
# =============================================================================

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
```
```
return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

```
```


def _predict_well_batched(model, seq, window_size, batch_size=512, device=None,
                           use_amp
```
```
=False):

```
```
    
```
```
"""

```
```
    Batched sliding-window inference for a single well.
```
```


```
```

    Instead of one forward pass per window (the original approach),
```
```


```
```
    this stacks all windows into a single tensor and processes them in
```
```


```
```
    large batches — typically 30-50x faster.

    Returns
    -------
    well_preds : np.ndarray, shape (T,)
        Prediction for every row.  Rows before the first full window
        are forward-filled from the first valid prediction.
```
```


```
```
    y : np.ndarray, shape (T,)
        Ground-truth targets (zeros where unavailable).
```
```


```
```
    """
```
```


```
```
    
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

    X = seq["X"]                                                       # (T, F)
    y 
```
```
= seq["y"] if seq["y"] is not None else np.zeros(len(X), np.float32)

```
```
    T      = len(X)
    n_wins 
```
```
= T - window_size + 1

```
```

    
```
```
# Build all windows at once with stride tricks — no Python loop needed

```
```
    
```
```
# idx shape: (n_wins, window_size)

```
```
    idx     = np.arange(window_size)[None, :] + np.arange(n_wins)[:, None]
    windows 
```
```
= X[idx].transpose(0, 2, 1).astype(np.float32)            # (n_wins, F, W)

```
```

    preds = np.empty(n_wins, dtype=np.float32)
    model.eval()
    
```
```
with torch.no_grad():

```
```
        
```
```
for start in range(0, n_wins, batch_size):

```
```
            batch 
```
```
= torch.from_numpy(windows[start:start + batch_size]).to(device,

```
```
                                                                            non_blocking
```
```
=True)

```
```
            
```
```
with torch.cuda.amp.autocast(enabled=use_amp):

```
```
                out = model(batch).squeeze(-1)
            preds[start:start + batch_size] = out.cpu().numpy()

    
```
```
# Align predictions to the full well length

```
```
    well_preds = np.empty(T, dtype=np.float32)
    well_preds[window_size - 1:] = preds
    well_preds[:window_size 
```
```
- 1] = preds[0]     # forward-fill early rows

```
```

    return well_preds, y

# =============================================================================
```
```


```
```
# 4. Configuration
```
```


```
```
# =============================================================================
```
```


```
```

BASE           = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
RAW_FEATURE_COLS = ["X", "Y", "Z", "GR", "MD"]
TARGET         
```
```
= "TVT_input"

```
```
WEIGHT_COL     = "weight"

WINDOW_SIZE    
```
```
= 64

```
```
BATCH_SIZE     = 128
EPOCHS         = 10     #5      # higher ceiling; early stopping cuts it short
PATIENCE       
```
```
= 5            # early-stop patience (epochs without improvement)

```
```
N_FOLDS        = 5            # 3 folds saves ~40% vs original 5 folds
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   
```
```
= 1e-4

```
```
GRAD_CLIP      
```
```
= 1.0

```
```
NUM_WORKERS    
```
```
= 2            # set 0 if you hit multiprocessing errors on Windows

```
```

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()    # automatic mixed precision (GPU only)

print
```
```
(f"Device : {DEVICE}")

```
```
print(f"AMP    : {USE_AMP}")

```
```
Device : cuda
AMP    : True

```
```

# =============================================================================
# 5. Data Loading
# =============================================================================

def
```
```
 load_wells(directory, max_files=None):

```
```
    pattern = os.path.join(directory, "*__horizontal_well.csv")
    files   
```
```
= sorted(glob.glob(pattern))

```
```
    if max_files:
        files 
```
```
= files[:max_files]

```
```

    dfs = []
    for f in files:
        df           = pd.read_csv(f)
        well_name    = os.path.basename(f).replace("__horizontal_well.csv", "")
        df[
```
```
"well"]   = well_name

```
```
        df[
```
```
"row_index"] = range(len(df))

```
```
        dfs
```
```
.append(df)

```
```

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


print("Loading train data...")
train = load_wells(os.path.join(BASE, "train"))
print
```
```
(f"  train shape: {train.shape}")

```
```

well_ids 
```
```
= train["well"].unique()

```
```
subset_wells 
```
```
= well_ids[:400] #200

```
```
train 
```
```
= train[train["well"].isin(subset_wells)].copy()

```
```

print
```
```
("Loading test data...")

```
```
test 
```
```
= load_wells(os.path.join(BASE, "test"))

```
```
print(f"  test shape : {test.shape}")

```
```
Loading train data...
  train shape: (5092255, 15)
Loading test data...
  test shape : (19221, 8)

```
```

# =============================================================================
```
```


```
```
# 6. STEP 1 — Scale raw features BEFORE any lag/rolling engineering
```
```


```
```
# =============================================================================
```
```


```
```

print("\n" + "="*60)
print("STEP 1: Scaling raw features BEFORE feature engineering")
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
scaler_raw    = StandardScaler()
scaler_raw
```
```
.fit(train_labeled[RAW_FEATURE_COLS].fillna(0))

```
```
print(f"Scaler fitted on {len(train_labeled):,} labeled rows — features: {RAW_FEATURE_COLS}")


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
    
```
```
for i, col in enumerate(raw_cols):

```
```
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


print
```
```
("Scaling train...")

```
```
train 
```
```
= scale_raw_features(train, scaler_raw, RAW_FEATURE_COLS)

```
```
print("Scaling test...")
test  = scale_raw_features(test,  scaler_raw, RAW_FEATURE_COLS)
print("Raw features scaled (mean=0, std=1)")

```
```
============================================================
STEP 1: Scaling raw features BEFORE feature engineering
============================================================
Scaler fitted on 677,231 labeled rows — features: ['X', 'Y', 'Z', 'GR', 'MD']
Scaling train...
Scaling test...
Raw features scaled (mean=0, std=1)

```
```

# =============================================================================
```
```


```
```
# 7. STEP 2 — Lag / rolling features on scaled data
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
("STEP 2: Building lag/rolling features on SCALED data")

```
```
print
```
```
("="*60)

```
```

LAG_STEPS    = (1, 2, 3, 5)
ROLL_WINDOWS = [3, 5, 10]


def add_lag_roll_features(df, feature_cols, lag_steps, roll_windows):
    df      = df.sort_values(["well", "MD"]).copy()
    new_cols = []
    
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
            cname    
```
```
= f"{col}_lag{lag}"

```
```
            df[cname] 
```
```
= grp.shift(lag)

```
```
            new_cols
```
```
.append(cname)

```
```
        for win in roll_windows:
            cname    
```
```
= f"{col}_roll{win}"

```
```
            df[cname] 
```
```
= grp.shift(1).transform(lambda x: x.rolling(win, min_periods=1).mean())

```
```
            new_cols.append(cname)
    
```
```
return df, new_cols

```
```


print
```
```
("Building lag/rolling features for train...")

```
```
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

print("Building lag/rolling features for test...")
test, _ = add_lag_roll_features(test, RAW_FEATURE_COLS, LAG_STEPS, ROLL_WINDOWS)
gc.collect()

ALL_FEATURES 
```
```
= RAW_FEATURE_COLS + lag_cols

```
```

missing = [c for c in ALL_FEATURES if c not in test.columns]
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

print
```
```
(f"\nTotal features (all scaled): {len(ALL_FEATURES)}")

```
```
============================================================
STEP 2: Building lag/rolling features on SCALED data
============================================================
Building lag/rolling features for train...
  lag/rolling features added: 35
Building lag/rolling features for test...

Total features (all scaled): 40

```
```

# =============================================================================
```
```


```
```
# 8. STEP 3 — Per-well sequences
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
("STEP 3: Preparing per-well sequences")

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
 create_well_sequence(df, well_id, feature_cols, target_col, window_size=64):

```
```
    well_data 
```
```
= df[df["well"] == well_id].sort_values("MD")

```
```
    X = well_data[feature_cols].fillna(0).values.astype(np.float32)
    y 
```
```
= (well_data[target_col].fillna(0).values.astype(np.float32)

```
```
         
```
```
if target_col and target_col in well_data.columns else None)

```
```
    
```
```
return {"well_id": well_id, "X": X, "y": y, "length": len(X)}

```
```


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
    seq 
```
```
= create_well_sequence(train, well_id, ALL_FEATURES, TARGET, WINDOW_SIZE)

```
```
    
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

print(f"Wells with sufficient length : {len(train_sequences)}")
print(f"Wells dropped (too short)    : {len(train_wells) - len(train_sequences)}")

```
```
============================================================
STEP 3: Preparing per-well sequences
============================================================
Wells with sufficient length : 400
Wells dropped (too short)    : 0

```
```

# =============================================================================
```
```


```
```
# 9. STEP 4 — GroupKFold cross-validation (grouped by well)
```
```


```
```
# =============================================================================

print("\n" + "="*60)
print("STEP 4: GroupKFold Cross-Validation (grouped by well)")
print
```
```
("="*60)

```
```

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
    
```
```
"""Train one fold with AMP, early stopping, and batched inference."""

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
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory
```
```
=True)

```
```
    val_loader   
```
```
= DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,

```
```
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory
```
```
=True)

```
```

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
= [32, 64, 128],

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
    )
```
```
.to(DEVICE)

```
```

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
    criterion  
```
```
= nn.MSELoss()

```
```
    scaler_amp 
```
```
= torch.cuda.amp.GradScaler(enabled=USE_AMP)

```
```

    best_val_loss, best_state, patience_ctr = float('inf'), None, 0

    print(f"\n  Fold {fold_idx} — {len(train_dataset):,} train / {len(val_dataset):,} val windows")

    
```
```
for epoch in range(1, EPOCHS + 1):

```
```

        # ── Training ───────────────────────────────────────────
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
for batch_x, batch_y in train_loader:

```
```
            batch_x 
```
```
= batch_x.to(DEVICE, non_blocking=True)

```
```
            batch_y 
```
```
= batch_y.to(DEVICE, non_blocking=True).squeeze()

```
```
            optimizer.zero_grad(set_to_none=True)
            
```
```
with torch.cuda.amp.autocast(enabled=USE_AMP):

```
```
                pred 
```
```
= model(batch_x).squeeze()

```
```
                loss = criterion(pred, batch_y)
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
.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

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

        
```
```
# ── Validation ─────────────────────────────────────────

```
```
        model.eval()
        val_loss, val_preds, val_targets 
```
```
= 0.0, [], []

```
```
        
```
```
with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):

```
```
            
```
```
for batch_x, batch_y in val_loader:

```
```
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
                pred     
```
```
= model(batch_x).squeeze()

```
```
                val_loss 
```
```
+= criterion(pred, batch_y).item()

```
```
                val_preds
```
```
.extend(pred.cpu().numpy())

```
```
                val_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_rmse    = rmse(val_targets, val_preds)
        scheduler
```
```
.step()

```
```

        print(f"    Ep {epoch:2d}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f}"
              f" | RMSE {val_rmse:.4f}")

        
```
```
# ── Early stopping ─────────────────────────────────────

```
```
        
```
```
if val_loss < best_val_loss:

```
```
            best_val_loss = val_loss
            best_state    
```
```
= copy.deepcopy(model.state_dict())

```
```
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
            patience_ctr 
```
```
+= 1

```
```
            if patience_ctr >= PATIENCE:
                
```
```
print(f"    Early stop triggered at epoch {epoch}")

```
```
                
```
```
break

```
```

    model
```
```
.load_state_dict(best_state)

```
```

    # ── OOF inference (batched) ────────────────────────────────
    oof_preds, oof_targets, oof_wells = [], [], []
    
```
```
for seq in val_seqs:

```
```
        preds, targets = _predict_well_batched(model, seq, WINDOW_SIZE,
                                               batch_size=BATCH_SIZE * 2,
                                               device
```
```
=DEVICE, use_amp=USE_AMP)

```
```
        mask 
```
```
= targets != 0

```
```
        oof_preds
```
```
.extend(preds[mask].tolist())

```
```
        oof_targets
```
```
.extend(targets[mask].tolist())

```
```
        oof_wells
```
```
.extend([seq["well_id"]] * int(mask.sum()))

```
```

    
```
```
return model, oof_preds, oof_targets, oof_wells

```
```


# ── Run CV ────────────────────────────────────────────────────────────────────
for
```
```
 fold, (train_idx, val_idx) in enumerate(

```
```
        gkf
```
```
.split(train_sequences, groups=well_ids), 1):

```
```

    print(f"\n{'='*50}")
    
```
```
print(f"FOLD {fold}/{N_FOLDS}")

```
```
    print(f"{'='*50}")

    train_seqs 
```
```
= [train_sequences[i] for i in train_idx]

```
```
    val_seqs   = [train_sequences[i] for i in val_idx]
    
```
```
print(f"Train wells: {len(train_seqs)} | Val wells: {len(val_seqs)}")

```
```

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
        
```
```
"target":     oof_targets,

```
```
        
```
```
"prediction": oof_preds,

```
```
        
```
```
"fold":       fold,

```
```
    }))

    print(f"\n  Fold {fold} RMSE: {fold_rmse:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch
```
```
.cuda.empty_cache()

```
```


oof_all      = pd.concat(fold_oof_predictions, ignore_index=True)
overall_rmse = rmse(oof_all["target"], oof_all["prediction"])

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
(f"Per-fold RMSEs : {[f'{s:.4f}' for s in fold_scores]}")

```
```
print(f"Mean fold RMSE : {np.mean(fold_scores):.4f}  (+/- {np.std(fold_scores):.4f})")
print
```
```
(f"Overall OOF RMSE: {overall_rmse:.4f}")

```
```
============================================================
STEP 4: GroupKFold Cross-Validation (grouped by well)
============================================================

==================================================
FOLD 1/5
==================================================
Train wells: 320 | Val wells: 80
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
  Fold 1 — 2,099,574 train / 514,771 val windows
/tmp/ipykernel_22/2514194365.py:41: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)
/tmp/ipykernel_22/2514194365.py:56: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=USE_AMP):
/tmp/ipykernel_22/2514194365.py:69: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
    Ep  1/10 | train 1882969.0149 | val 1702197.9441 | RMSE 1304.7401
    Ep  2/10 | train 1497319.9042 | val 1859601.0149 | RMSE 1363.7315
    Ep  3/10 | train 1403915.1117 | val 1719685.0520 | RMSE 1311.4249
    Ep  4/10 | train 1326638.8050 | val 1671875.1039 | RMSE 1293.0666
    Ep  5/10 | train 1272373.3627 | val 1755424.0140 | RMSE 1324.9821
    Ep  6/10 | train 1206872.5414 | val 1781926.9845 | RMSE 1334.9467
    Ep  7/10 | train 1162566.4798 | val 1835434.2156 | RMSE 1354.8412
    Ep  8/10 | train 1114426.6423 | val 1826803.8181 | RMSE 1351.6521
    Ep  9/10 | train 1079719.7413 | val 1939603.4232 | RMSE 1392.7573
    Early stop triggered at epoch 9
/tmp/ipykernel_22/627506093.py:47: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
  Fold 1 RMSE: 2097.9283

==================================================
FOLD 2/5
==================================================
Train wells: 320 | Val wells: 80

  Fold 2 — 2,095,194 train / 519,151 val windows
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
    Ep  1/10 | train 1814917.7346 | val 1893024.4180 | RMSE 1375.8948
    Ep  2/10 | train 1440273.8356 | val 2138432.1497 | RMSE 1462.3619
    Ep  3/10 | train 1352646.3762 | val 1906145.8196 | RMSE 1380.6550
    Ep  4/10 | train 1278674.8570 | val 1981282.6323 | RMSE 1407.6035
    Ep  5/10 | train 1212658.4136 | val 2049717.9087 | RMSE 1431.7070
    Ep  6/10 | train 1151722.1107 | val 2085218.2357 | RMSE 1444.0521
    Early stop triggered at epoch 6

  Fold 2 RMSE: 1709.3533

==================================================
FOLD 3/5
==================================================
Train wells: 320 | Val wells: 80

  Fold 3 — 2,081,775 train / 532,570 val windows
    Ep  1/10 | train 1780523.2272 | val 2015688.5237 | RMSE 1419.9707
    Ep  2/10 | train 1394931.7105 | val 2143246.9388 | RMSE 1464.2114
    Ep  3/10 | train 1301384.2270 | val 2044504.8512 | RMSE 1430.0846
    Ep  4/10 | train 1227262.2010 | val 2181310.3237 | RMSE 1477.1561
    Ep  5/10 | train 1151646.5346 | val 2272536.9038 | RMSE 1507.7285
    Ep  6/10 | train 1083489.2548 | val 2338184.7934 | RMSE 1529.3507
    Early stop triggered at epoch 6

  Fold 3 RMSE: 1725.4973

==================================================
FOLD 4/5
==================================================
Train wells: 320 | Val wells: 80

  Fold 4 — 2,100,703 train / 513,642 val windows
    Ep  1/10 | train 1930884.8320 | val 1671578.6115 | RMSE 1293.0842
    Ep  2/10 | train 1557970.8114 | val 1399055.5645 | RMSE 1182.9894
    Ep  3/10 | train 1472448.8593 | val 1487997.3042 | RMSE 1220.0131
    Ep  4/10 | train 1393470.5598 | val 1605609.4785 | RMSE 1267.3115
    Ep  5/10 | train 1314706.2777 | val 1502232.6592 | RMSE 1225.8349
    Ep  6/10 | train 1242957.6681 | val 1476635.7087 | RMSE 1215.3464
    Ep  7/10 | train 1179107.9161 | val 1456298.8313 | RMSE 1206.9483
    Early stop triggered at epoch 7

  Fold 4 RMSE: 1624.0711

==================================================
FOLD 5/5
==================================================
Train wells: 320 | Val wells: 80

  Fold 5 — 2,080,134 train / 534,211 val windows
    Ep  1/10 | train 1926712.5327 | val 1455654.1439 | RMSE 1206.5738
    Ep  2/10 | train 1545733.3551 | val 1583304.9755 | RMSE 1258.3663
    Ep  3/10 | train 1447766.2724 | val 1513473.5040 | RMSE 1230.3033
    Ep  4/10 | train 1374975.5071 | val 1586240.0577 | RMSE 1259.5321
    Ep  5/10 | train 1302190.0681 | val 1559859.1443 | RMSE 1249.0145
    Ep  6/10 | train 1224442.9732 | val 1538642.5544 | RMSE 1240.4911
    Early stop triggered at epoch 6

  Fold 5 RMSE: 1929.3464

============================================================
CROSS-VALIDATION RESULTS
============================================================
Per-fold RMSEs : ['2097.9283', '1709.3533', '1725.4973', '1624.0711', '1929.3464']
Mean fold RMSE : 1817.2393  (+/- 172.4670)
Overall OOF RMSE: 1823.6965

```
```

# =============================================================================
# 10. STEP 5 — Final model trained on all data
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
("STEP 5: Training final model on all training data")

```
```
print("="*60)

final_dataset = WellSequenceDataset(train_sequences, WINDOW_SIZE)
final_loader  
```
```
= DataLoader(final_dataset, batch_size=BATCH_SIZE,

```
```
                           shuffle
```
```
=True, num_workers=NUM_WORKERS, pin_memory=True)

```
```

final_model = TCNModel(
    input_size   
```
```
= len(ALL_FEATURES),

```
```
    output_size  = 1,
    num_channels 
```
```
= [32, 64, 128],

```
```
    kernel_size  
```
```
= 3,

```
```
    dropout      = 0.1,
)
```
```
.to(DEVICE)

```
```

optimizer  
```
```
= optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

```
```
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion  
```
```
= nn.MSELoss()

```
```
scaler_amp 
```
```
= torch.cuda.amp.GradScaler(enabled=USE_AMP)

```
```

print
```
```
(f"Training on {len(final_dataset):,} windows...")

```
```

for
```
```
 epoch in range(1, EPOCHS + 1):

```
```
    final_model
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
for batch_x, batch_y in final_loader:

```
```
        batch_x 
```
```
= batch_x.to(DEVICE, non_blocking=True)

```
```
        batch_y 
```
```
= batch_y.to(DEVICE, non_blocking=True).squeeze()

```
```
        optimizer
```
```
.zero_grad(set_to_none=True)

```
```
        
```
```
with torch.cuda.amp.autocast(enabled=USE_AMP):

```
```
            pred = final_model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP)
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

    train_loss 
```
```
/= len(final_loader)

```
```
    scheduler.step()

    if epoch % 5 == 0:
        
```
```
print(f"  Ep {epoch:2d}/{EPOCHS} | train loss: {train_loss:.4f}")

```
```

print
```
```
("Final model training complete!")

```
```


# =============================================================================
```
```


```
```
# 11. STEP 6 — Inference on test set (batched)
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
("STEP 6: Inference on test set")

```
```
print
```
```
("="*60)

```
```

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
    seq 
```
```
= create_well_sequence(test, well_id, ALL_FEATURES, None, WINDOW_SIZE)

```
```
    
```
```
if seq["length"] >= WINDOW_SIZE:

```
```
        test_sequences
```
```
.append(seq)

```
```

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

final_model
```
```
.eval()

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
    row_indices  = well_test_df["row_index"].values

    well_preds, _ 
```
```
= _predict_well_batched(

```
```
        final_model, seq, WINDOW_SIZE,
        batch_size=BATCH_SIZE * 2,
        device=DEVICE, use_amp=USE_AMP,
    )

    
```
```
for row_idx, pred_tvt in zip(row_indices, well_preds):

```
```
        test_predictions.append({
            
```
```
"well_id":   well_id,

```
```
            
```
```
"row_index": row_idx,

```
```
            
```
```
"tvt":       float(pred_tvt),

```
```
        })

test_pred_df = pd.DataFrame(test_predictions)
print(f"Generated {len(test_pred_df):,} test predictions")

```
```
============================================================
STEP 5: Training final model on all training data
============================================================
Training on 2,614,345 windows...
/tmp/ipykernel_22/772382643.py:24: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)
/tmp/ipykernel_22/772382643.py:36: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=USE_AMP):
  Ep  5/10 | train loss: 1273725.6984
  Ep 10/10 | train loss: 1054533.7993
Final model training complete!

============================================================
STEP 6: Inference on test set
============================================================
Test wells with sufficient length: 3
/tmp/ipykernel_22/627506093.py:47: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
Generated 19,221 test predictions

```
```

# =============================================================================
```
```


```
```
# 12. STEP 7 — Build submission file
```
```


```
```
# =============================================================================

sample_sub              = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
sample_sub[
```
```
"well"]      = sample_sub["id"].str.rsplit("_", n=1).str[0]

```
```
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)

# ★ Drop the placeholder column so it doesn't collide with our predictions
```
```


```
```
sample_sub = sample_sub.drop(columns=["tvt"], errors="ignore")

submission 
```
```
= sample_sub.merge(

```
```
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
    right_on=["well_id", "row_index"],
    how
```
```
="left",

```
```
)

# Now "tvt" unambiguously refers to our predictions ✓
```
```


```
```
submission["tvt"] = submission["tvt"].fillna(
    submission.groupby("well")["tvt"].transform("first")
)

submission = submission[["id", "tvt"]]
submission
```
```
.to_csv("submission.csv", index=False)

```
```

# =============================================================================
# 13. STEP 8 (optional) — Ensemble all fold models
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


def predict_with_ensemble(models, test_sequences, window_size=WINDOW_SIZE,
                          batch_size
```
```
=BATCH_SIZE * 2, device=DEVICE,

```
```
                          use_amp
```
```
=USE_AMP):

```
```
    
```
```
"""

```
```
    Average predictions from every fold model.
    Uses _predict_well_batched so it is just as fast as single-model inference.
```
```


```
```
    """
```
```


```
```
    all_fold_preds 
```
```
= []

```
```

    for model in models:
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
            row_indices  = well_test_df["row_index"].values

            well_preds, _ 
```
```
= _predict_well_batched(

```
```
                model, seq, window_size, batch_size, device, use_amp
            )

            for row_idx, pred_tvt in zip(row_indices, well_preds):
                fold_rows
```
```
.append({

```
```
                    
```
```
"well_id":   well_id,

```
```
                    "row_index": row_idx,
                    "tvt":       float(pred_tvt),
                })
        all_fold_preds
```
```
.append(pd.DataFrame(fold_rows))

```
```

    combined      = pd.concat(all_fold_preds)
    ensemble_pred 
```
```
= (combined

```
```
                     
```
```
.groupby(["well_id", "row_index"])["tvt"]

```
```
                     
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


# Uncomment the block below to produce an ensemble submission:
```
```


```
```
#
# print("Running ensemble inference...")
```
```


```
```
# ensemble_test_pred = predict_with_ensemble(fold_models, test_sequences)
```
```


```
```
#
```
```


```
```
# ensemble_sub = sample_sub.merge(
#     ensemble_test_pred,
#     left_on=["well", "row_index"],
#     right_on=["well_id", "row_index"],
```
```


```
```
#     how="left",
# )
```
```


```
```
# ensemble_sub["tvt"] = ensemble_sub["tvt"].fillna(
#     ensemble_sub.groupby("well")["tvt"].transform("first")
# )
```
```


```
```
# ensemble_sub[["id", "tvt"]].to_csv("submission_ensemble.csv", index=False)
# print("Ensemble submission saved to submission_ensemble.csv")


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
============================================================
STEP 8 (Optional): Ensemble predictions from all fold models
============================================================

============================================================
DONE!
============================================================

```
```

 

 
```
```


```
