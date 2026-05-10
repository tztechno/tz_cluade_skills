---
name: wellbore-geology-tcn
description: >
  Wellbore地質データにTCN（Temporal Convolutional Network）を適用してTVT（True Vertical Thickness）を
  予測するKaggleコンペ向けPythonパイプラインを生成するスキル。
  次のようなリクエストで必ずこのスキルを使うこと：
  - 「ウェルボア / 井戸 / well log データを使って機械学習」
  - 「TCNで時系列予測」「Temporal Convolutional Network」
  - 「地質データ予測」「TVT予測」「well geology」
  - 「Kaggle rogii-wellbore-geology-prediction」
  - 「スライディングウィンドウ + 時系列CNN」
  ユーザーが明示的にTCNと言わなくても、wellデータやdrilling logの予測タスクに言及したら積極的に提案すること。
---

# Wellbore Geology TCN スキル

## 概要

このスキルは、水平井（horizontal well）の測定データ（X, Y, Z, GR, MD等）から
TVT（True Vertical Thickness）を予測するエンドツーエンドのPythonパイプラインを生成する。

アーキテクチャ：**TCN（Temporal Convolutional Network）**
参考論文：https://arxiv.org/abs/1803.01271
参考実装：https://github.com/locuslab/TCN

---

## パイプライン全体構成（8ステップ）

```
STEP 1: 生の特徴量をスケーリング（lag/rolling前に実施 ← リーケージ防止）
STEP 2: lag / rolling 特徴量の生成（スケーリング済みデータに対して）
STEP 3: 井戸ごとのシーケンス作成
STEP 4: GroupKFold クロスバリデーション（well IDでグループ化）
STEP 5: 全訓練データで最終モデルを学習
STEP 6: テストセットへの推論（バッチ化スライディングウィンドウ）
STEP 7: submission.csv の生成
STEP 8: （オプション）フォールドモデルのアンサンブル
```

---

## 1. TCNモデル定義

```python
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConv1dBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp = Chomp1d(padding)
        self.relu  = nn.ReLU()
        self.net   = nn.Sequential(self.conv, self.chomp, self.relu)

    def forward(self, x):
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding      = (kernel_size - 1) * dilation
        self.conv1   = CausalConv1dBlock(n_inputs,   n_outputs, kernel_size, stride, dilation, padding)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2   = CausalConv1dBlock(n_outputs, n_outputs, kernel_size, stride, dilation, padding)
        self.dropout2 = nn.Dropout(dropout)
        self.net      = nn.Sequential(self.conv1, self.dropout1, self.conv2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu     = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch  = input_size          if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 1, dilation_size, dropout))
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (batch, features, window)  →  output: (batch, 1)
        return self.linear(self.tcn(x)[:, :, -1])
```

**ポイント**：
- `Chomp1d`でパディングを削除し因果性を保証（未来の情報を使わない）
- `weight_norm`で学習安定化
- 各`TCNBlock`に残差接続（residual connection）
- dilation = 2^i で指数的に受容野を拡大

---

## 2. WellSequenceDataset（スライディングウィンドウ）

```python
from torch.utils.data import Dataset
import torch

class WellSequenceDataset(Dataset):
    """コンストラクタ時にすべてのウィンドウを事前生成してフラットなリストに保持"""

    def __init__(self, sequences_data, window_size=64):
        self.samples     = []
        self.window_size = window_size

        for seq in sequences_data:
            X = seq['X']   # (T, F)
            y = seq['y']   # (T,)
            T = len(X)
            for i in range(T - window_size + 1):
                x_win    = X[i:i + window_size].T          # (F, W)
                y_target = y[i + window_size - 1]
                self.samples.append((x_win, y_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x.copy()), torch.tensor([y], dtype=torch.float32)
```

---

## 3. ヘルパー関数

### RMSE
```python
import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
```

### バッチ化推論（30-50x高速化）
```python
def _predict_well_batched(model, seq, window_size, batch_size=512, device=None, use_amp=False):
    """
    1つの井戸に対してバッチスライディングウィンドウ推論を実行。
    従来の1ウィンドウずつの推論より30-50倍高速。
    
    Returns
    -------
    well_preds : np.ndarray, shape (T,)
    y          : np.ndarray, shape (T,)
    """
    if device is None:
        device = next(model.parameters()).device

    X = seq["X"]
    y = seq["y"] if seq["y"] is not None else np.zeros(len(seq["X"]), np.float32)
    T      = len(X)
    n_wins = T - window_size + 1

    # stride tricksで全ウィンドウをまとめてテンソル化（Pythonループなし）
    idx     = np.arange(window_size)[None, :] + np.arange(n_wins)[:, None]
    windows = X[idx].transpose(0, 2, 1).astype(np.float32)  # (n_wins, F, W)

    preds = np.empty(n_wins, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_wins, batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(batch).squeeze(-1)
            preds[start:start + batch_size] = out.cpu().numpy()

    # 全行に予測値を割り当て（最初のwindow_size-1行は最初の予測値で前埋め）
    well_preds = np.empty(T, dtype=np.float32)
    well_preds[window_size - 1:] = preds
    well_preds[:window_size - 1] = preds[0]

    return well_preds, y
```

---

## 4. 設定（ハイパーパラメータ）

```python
BASE           = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
RAW_FEATURE_COLS = ["X", "Y", "Z", "GR", "MD"]
TARGET         = "TVT_input"
WEIGHT_COL     = "weight"

WINDOW_SIZE    = 64
BATCH_SIZE     = 128
EPOCHS         = 10
PATIENCE       = 5      # 早期終了の辛抱エポック数
N_FOLDS        = 5      # GroupKFoldのfold数（3にすると40%時間短縮）
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
NUM_WORKERS    = 2      # Windowsでエラーが出た場合は0に設定

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # GPU時のみAuto Mixed Precision有効
```

**モデルアーキテクチャ設定**（`train_fold`内）：
```python
model = TCNModel(
    input_size   = len(ALL_FEATURES),
    output_size  = 1,
    num_channels = [32, 64, 128],   # 3層、チャネル数を増やす
    kernel_size  = 3,
    dropout      = 0.1,
).to(DEVICE)
```

---

## 5. データ読み込み

```python
import glob, os, pandas as pd

def load_wells(directory, max_files=None):
    pattern = os.path.join(directory, "*__horizontal_well.csv")
    files   = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    dfs = []
    for f in files:
        df           = pd.read_csv(f)
        well_name    = os.path.basename(f).replace("__horizontal_well.csv", "")
        df["well"]   = well_name
        df["row_index"] = range(len(df))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

train = load_wells(os.path.join(BASE, "train"))
# 必要に応じてサブセット（メモリ節約）
well_ids     = train["well"].unique()
subset_wells = well_ids[:400]
train        = train[train["well"].isin(subset_wells)].copy()

test = load_wells(os.path.join(BASE, "test"))
```

---

## 6. STEP 1 — 生の特徴量をスケーリング（リーケージ防止）

```python
from sklearn.preprocessing import StandardScaler

# ラベルあり行のみでscalerをfitする（リーケージ防止）
train_labeled = train[train[TARGET].notna()].copy()
scaler_raw    = StandardScaler()
scaler_raw.fit(train_labeled[RAW_FEATURE_COLS].fillna(0))

def scale_raw_features(df, scaler, raw_cols):
    scaled = scaler.transform(df[raw_cols].fillna(0))
    for i, col in enumerate(raw_cols):
        df[col] = scaled[:, i]
    return df

train = scale_raw_features(train, scaler_raw, RAW_FEATURE_COLS)
test  = scale_raw_features(test,  scaler_raw, RAW_FEATURE_COLS)
```

> **重要**: スケーリングをlag/rolling特徴量生成の**前**に行う。
> 逆順にするとlag列にスケーリング前の値が混入し、リーケージが発生する。

---

## 7. STEP 2 — Lag / Rolling 特徴量（スケーリング済みデータに適用）

```python
LAG_STEPS    = (1, 2, 3, 5)
ROLL_WINDOWS = [3, 5, 10]

def add_lag_roll_features(df, feature_cols, lag_steps, roll_windows):
    df      = df.sort_values(["well", "MD"]).copy()
    new_cols = []
    for col in feature_cols:
        grp = df.groupby("well")[col]
        for lag in lag_steps:
            cname    = f"{col}_lag{lag}"
            df[cname] = grp.shift(lag)
            new_cols.append(cname)
        for win in roll_windows:
            cname    = f"{col}_roll{win}"
            df[cname] = grp.shift(1).transform(lambda x: x.rolling(win, min_periods=1).mean())
            new_cols.append(cname)
    return df, new_cols

train, lag_cols = add_lag_roll_features(train, RAW_FEATURE_COLS, LAG_STEPS, ROLL_WINDOWS)
test,  _        = add_lag_roll_features(test,  RAW_FEATURE_COLS, LAG_STEPS, ROLL_WINDOWS)

ALL_FEATURES = RAW_FEATURE_COLS + lag_cols
# testに存在しない列を除外
missing      = [c for c in ALL_FEATURES if c not in test.columns]
ALL_FEATURES = [c for c in ALL_FEATURES if c not in missing]
# → 合計40特徴量（5生 + 35 lag/rolling）
```

---

## 8. STEP 3 — 井戸ごとのシーケンス作成

```python
def create_well_sequence(df, well_id, feature_cols, target_col, window_size=64):
    well_data = df[df["well"] == well_id].sort_values("MD")
    X = well_data[feature_cols].fillna(0).values.astype(np.float32)
    y = (well_data[target_col].fillna(0).values.astype(np.float32)
         if target_col and target_col in well_data.columns else None)
    return {"well_id": well_id, "X": X, "y": y, "length": len(X)}

train_wells     = train["well"].unique()
train_sequences = []
for well_id in train_wells:
    seq = create_well_sequence(train, well_id, ALL_FEATURES, TARGET, WINDOW_SIZE)
    if seq["length"] >= WINDOW_SIZE:
        train_sequences.append(seq)
```

---

## 9. STEP 4 — GroupKFold クロスバリデーション

```python
import copy
from sklearn.model_selection import GroupKFold
import torch.optim as optim

well_ids = np.array([seq["well_id"] for seq in train_sequences])
gkf      = GroupKFold(n_splits=N_FOLDS)

fold_models          = []
fold_oof_predictions = []
fold_scores          = []

def train_fold(fold_idx, train_seqs, val_seqs):
    """AMP・早期終了・バッチ推論付きで1フォールドを学習"""
    train_dataset = WellSequenceDataset(train_seqs, WINDOW_SIZE)
    val_dataset   = WellSequenceDataset(val_seqs,   WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = TCNModel(
        input_size=len(ALL_FEATURES), output_size=1,
        num_channels=[32, 64, 128], kernel_size=3, dropout=0.1,
    ).to(DEVICE)

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion  = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_loss, best_state, patience_ctr = float('inf'), None, 0

    for epoch in range(1, EPOCHS + 1):
        # ── 訓練 ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()

        # ── 検証 ──────────────────────────────────────────
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
                pred     = model(batch_x).squeeze()
                val_loss += criterion(pred, batch_y).item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_rmse    = rmse(val_targets, val_preds)
        scheduler.step()

        # ── 早期終了 ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model

# フォールドループ
for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(np.zeros(len(train_sequences)), groups=well_ids)):
    train_seqs = [train_sequences[i] for i in train_idx]
    val_seqs   = [train_sequences[i] for i in val_idx]
    fold_model = train_fold(fold_idx + 1, train_seqs, val_seqs)
    fold_models.append(fold_model)
```

> **重要**: `GroupKFold`でwellをグループ化することで、同じ井戸のデータが
> 訓練・検証に同時に入るデータリーケージを完全に防ぐ。

---

## 10. STEP 5 — 全訓練データで最終モデル学習

```python
final_model    = TCNModel(
    input_size=len(ALL_FEATURES), output_size=1,
    num_channels=[32, 64, 128], kernel_size=3, dropout=0.1,
).to(DEVICE)
final_dataset  = WellSequenceDataset(train_sequences, WINDOW_SIZE)
final_loader   = DataLoader(final_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
optimizer      = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler      = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion      = nn.MSELoss()
scaler_amp     = torch.amp.GradScaler('cuda', enabled=USE_AMP)

for epoch in range(1, EPOCHS + 1):
    final_model.train()
    train_loss = 0.0
    for batch_x, batch_y in final_loader:
        batch_x = batch_x.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True).squeeze()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            pred = final_model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        train_loss += loss.item()
    scheduler.step()
    if epoch % 5 == 0:
        print(f"  Ep {epoch:2d}/{EPOCHS} | train loss: {train_loss/len(final_loader):.4f}")
```

---

## 11. STEP 6 — テスト推論（バッチ化）

```python
test_wells     = test["well"].unique()
test_sequences = []
for well_id in test_wells:
    seq = create_well_sequence(test, well_id, ALL_FEATURES, None, WINDOW_SIZE)
    if seq["length"] >= WINDOW_SIZE:
        test_sequences.append(seq)

test_predictions = []
final_model.eval()
for seq in test_sequences:
    well_id      = seq["well_id"]
    well_test_df = test[test["well"] == well_id].sort_values("MD")
    row_indices  = well_test_df["row_index"].values

    well_preds, _ = _predict_well_batched(
        final_model, seq, WINDOW_SIZE,
        batch_size=BATCH_SIZE * 2,
        device=DEVICE, use_amp=USE_AMP,
    )

    for row_idx, pred_tvt in zip(row_indices, well_preds):
        test_predictions.append({
            "well_id":   well_id,
            "row_index": row_idx,
            "tvt":       float(pred_tvt),
        })

test_pred_df = pd.DataFrame(test_predictions)
```

---

## 12. STEP 7 — Submission生成

```python
sample_sub              = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
sample_sub["well"]      = sample_sub["id"].str.rsplit("_", n=1).str[0]
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)
sample_sub = sample_sub.drop(columns=["tvt"], errors="ignore")   # プレースホルダー列を除去

submission = sample_sub.merge(
    test_pred_df[["well_id", "row_index", "tvt"]],
    left_on=["well", "row_index"],
    right_on=["well_id", "row_index"],
    how="left",
)
submission["tvt"] = submission["tvt"].fillna(
    submission.groupby("well")["tvt"].transform("first")
)
submission = submission[["id", "tvt"]]
submission.to_csv("submission.csv", index=False)
```

---

## 13. STEP 8 — アンサンブル（オプション）

```python
def predict_with_ensemble(models, test_sequences, window_size=WINDOW_SIZE,
                          batch_size=BATCH_SIZE * 2, device=DEVICE, use_amp=USE_AMP):
    """全フォールドモデルの予測を平均化"""
    all_fold_preds = []
    for model in models:
        fold_rows = []
        for seq in test_sequences:
            well_id      = seq["well_id"]
            well_test_df = test[test["well"] == well_id].sort_values("MD")
            row_indices  = well_test_df["row_index"].values
            well_preds, _ = _predict_well_batched(
                model, seq, window_size, batch_size, device, use_amp
            )
            for row_idx, pred_tvt in zip(row_indices, well_preds):
                fold_rows.append({"well_id": well_id, "row_index": row_idx, "tvt": float(pred_tvt)})
        all_fold_preds.append(pd.DataFrame(fold_rows))

    combined      = pd.concat(all_fold_preds)
    ensemble_pred = (combined
                     .groupby(["well_id", "row_index"])["tvt"]
                     .mean()
                     .reset_index())
    return ensemble_pred

# 使用する場合はコメントアウトを外す：
# ensemble_test_pred = predict_with_ensemble(fold_models, test_sequences)
# ensemble_sub = sample_sub.merge(ensemble_test_pred, ...)
# ensemble_sub[["id", "tvt"]].to_csv("submission_ensemble.csv", index=False)
```

---

## 主要な改善点（オリジナルからの変更）

| # | 改善内容 | 効果 |
|---|---------|------|
| 1 | **生の特徴量を先にスケーリング**してからlag/rolling生成 | データリーケージ排除 |
| 2 | **GroupKFold**でwell IDごとにグループ化 | well間のデータリーケージ排除 |
| 3 | **バッチ化ウィンドウ推論**（stride tricks） | 推論速度30-50倍向上 |
| 4 | **AMP**（Automatic Mixed Precision） | 学習速度1.5-2倍向上 |
| 5 | **早期終了**（Early Stopping） | 無駄なエポックを削減 |
| 6 | **num_workers + pin_memory** | データロードとGPUを並列化 |
| 7 | フォールド数を**3に削減**（必要に応じて） | CV時間40%削減 |

---

## 必要なライブラリ

```bash
pip install torch torchvision pandas numpy scikit-learn
```

Kaggle環境では全て利用可能。`torch.cuda.amp`の一部APIは非推奨のため、
`torch.amp.autocast('cuda', ...)` と `torch.amp.GradScaler('cuda', ...)` を使用すること。

---

## カスタマイズポイント

ユーザーが変更したい可能性の高い設定：
- `RAW_FEATURE_COLS`：使用する生の特徴量（GR, X, Y, Z, MD等）
- `WINDOW_SIZE`：スライディングウィンドウの長さ（デフォルト64）
- `num_channels`：TCNの各層のチャネル数（デフォルト`[32, 64, 128]`）
- `N_FOLDS`：フォールド数（時間を節約するには3に設定）
- `subset_wells`：使用するwell数（メモリ制限がある場合）
- `EPOCHS / PATIENCE`：最大エポック数と早期終了のパラメータ
