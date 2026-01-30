# PhysicsForTimePrediction

時系列予測モデル（Transformer系 + NeuralODE）のSoTA比較ベンチマークリポジトリ

## 概要

このリポジトリは、論文の性能比較のために以下のモデルを統一インターフェースで実装しています：

### 実装済みモデル

**Transformer系**
- **Informer** - ProbSparse self-attention による O(L log L) 複雑度
- **Autoformer** - Auto-Correlation mechanism と Series Decomposition

**NeuralODE系**
- **Latent ODE** - VAE構造 + ODE dynamics（Adjoint法でO(1)メモリ）
- **ODE-RNN** - GRU更新 + ODE連続時間進化

### 対応データセット

- Weather (気象データ)
- ILI (インフルエンザ様疾患)
- Exchange (為替レート)

## インストール

```bash
# リポジトリのクローン
cd PhysicsForTimePrediction

# 依存関係のインストール
pip install -e .

# または requirements.txt を使用
pip install -r requirements.txt
```

## 使い方

### 1. データセットのダウンロード

```bash
# 全データセットをダウンロード
python scripts/download_data.py --dataset all

# 特定のデータセットのみ
python scripts/download_data.py --dataset weather
```

### 2. 訓練

```bash
# Informer + Weather
python scripts/run.py \
    --config configs/models/informer.yaml \
    --data_config configs/datasets/weather.yaml \
    --mode train

# Latent ODE + Weather
python scripts/run.py \
    --config configs/models/latent_ode.yaml \
    --data_config configs/datasets/weather.yaml \
    --mode train
```

### 3. 評価

```bash
python scripts/run.py \
    --config configs/models/informer.yaml \
    --data_config configs/datasets/weather.yaml \
    --mode test
```

### 4. ベンチマーク実行

```bash
# 全モデル × 全データセットの比較
python scripts/benchmark.py \
    --models informer autoformer latent_ode ode_rnn \
    --datasets weather ili exchange
```

## プロジェクト構造

```
PhysicsForTimePrediction/
├── configs/                    # 設定ファイル
│   ├── base/default.yaml       # デフォルト設定
│   ├── models/                 # モデル別設定
│   └── datasets/               # データセット別設定
│
├── src/
│   ├── models/                 # モデル実装
│   │   ├── base.py             # BaseModel, BaseODEModel
│   │   ├── transformers/       # Informer, Autoformer
│   │   └── neural_ode/         # Latent ODE, ODE-RNN
│   │
│   ├── data/                   # データローディング
│   │   ├── base.py             # BaseTimeSeriesDataset
│   │   ├── datasets/           # 各データセット実装
│   │   └── download.py         # データダウンロード
│   │
│   ├── training/               # 訓練インフラ
│   │   ├── trainer.py          # StandardTrainer
│   │   └── ode_trainer.py      # AdjointTrainer
│   │
│   ├── evaluation/             # 評価
│   │   ├── metrics.py          # MSE, MAE, RMSE
│   │   └── evaluator.py        # 評価パイプライン
│   │
│   └── utils/                  # ユーティリティ
│
└── scripts/                    # 実行スクリプト
    ├── run.py                  # 訓練/評価
    ├── download_data.py        # データダウンロード
    └── benchmark.py            # ベンチマーク
```

## 設定

設定はYAMLファイルで管理され、継承をサポートしています：

```yaml
# configs/experiments/informer_weather.yaml
_base_:
  - ../base/default.yaml
  - ../models/informer.yaml
  - ../datasets/weather.yaml

training:
  epochs: 100
  batch_size: 32
```

## 重要な設計ポイント

### NeuralODE の制約

ODE dynamics 関数では **Tanh/Softplus** 等の滑らかな活性化関数を使用する必要があります。ReLU は Lipschitz 連続でないため使用できません。

```python
# OK
self.activation = nn.Tanh()

# NG - ODE dynamics では使用不可
self.activation = nn.ReLU()
```

### Adjoint法

NeuralODE モデルは `odeint_adjoint` を使用してO(1)メモリで学習を行います。

## 評価指標

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

## 参考文献

- [Informer (AAAI 2021)](https://arxiv.org/abs/2012.07436)
- [Autoformer (NeurIPS 2021)](https://arxiv.org/abs/2106.13008)
- [Neural ODEs (NeurIPS 2018)](https://arxiv.org/abs/1806.07366)
- [Latent ODEs (NeurIPS 2019)](https://arxiv.org/abs/1907.03907)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
