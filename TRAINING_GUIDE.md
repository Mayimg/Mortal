# Mortal AI 学習ガイド: MJAI形式データから学習済みモデルまで

このガイドでは、MJAI形式の麻雀牌譜を使用してMortal AIモデルを学習するための包括的なステップバイステップのプロセスを提供します。

## 目次
1. [前提条件](#前提条件)
2. [データ形式の理解](#データ形式の理解)
3. [環境構築](#環境構築)
4. [データ準備](#データ準備)
5. [学習設定](#学習設定)
6. [学習プロセス](#学習プロセス)
7. [モニタリングと検証](#モニタリングと検証)
8. [トラブルシューティング](#トラブルシューティング)

## 前提条件

開始前に以下を確認してください：
- MJAI形式の麻雀牌譜（`.json.gz`ファイル）
- CUDA対応GPU（推奨：RTX 3090以上）
- 十分なディスク容量（大規模データセットの場合100GB以上）
- LinuxまたはMSYS2環境のWindows

## データ形式の理解

### MJAI形式の構造
MJAI（Mahjong AI）は、麻雀ゲームを表現するためのJSONベースのプロトコルです。各ゲームログは一連のイベントで構成されています：

```json
{"type": "start_game", "names": ["Player1", "Player2", "Player3", "Player4"]}
{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0, 
 "scores": [25000, 25000, 25000, 25000], "tehais": [...]}
{"type": "tsumo", "actor": 0, "pai": "1m"}
{"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": false}
{"type": "hora", "actor": 1, "target": 0, "deltas": [-1000, 1000, 0, 0]}
{"type": "end_kyoku"}
{"type": "end_game"}
```

### 主な要件
- ファイルはgzip圧縮されたJSON（`.json.gz`）である必要があります
- 各行に1つのイベントが含まれます
- スコア計算のため、イベントには適切な`deltas`フィールドが必要です
- すべてのプレイヤーアクションは日本麻雀のルールに従って有効である必要があります

## 環境構築

### 1. リポジトリのクローンとセットアップ
```bash
git clone https://github.com/Equim-chan/Mortal.git
cd Mortal
```

### 2. Conda環境の作成
```bash
conda env create -f environment.yml
conda activate mortal
```

### 3. PyTorchのインストール
CUDAバージョンに基づいてPyTorchをインストール：
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Rustコンポーネントのビルド
```bash
# Rustライブラリのビルド
cargo build -p libriichi --lib --release

# Pythonモジュールディレクトリにコピー
# Linux:
cp target/release/libriichi.so mortal/libriichi.so

# Windows MSYS2:
cp target/release/riichi.dll mortal/libriichi.pyd
```

## データ準備

### 1. データの整理
MJAIデータ用のディレクトリ構造を作成：
```
data/
├── training/
│   ├── 2023/
│   │   ├── game_001.json.gz
│   │   ├── game_002.json.gz
│   │   └── ...
│   └── 2024/
│       └── ...
└── validation/
    └── ...
```

### 2. データ形式の検証
データをチェックするための簡単な検証スクリプトを作成：

```python
import gzip
import json

def validate_mjai_file(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        game_started = False
        kyoku_started = False
        
        for line in f:
            event = json.loads(line)
            event_type = event.get('type')
            
            if event_type == 'start_game':
                game_started = True
            elif event_type == 'start_kyoku':
                kyoku_started = True
            elif event_type == 'end_kyoku':
                kyoku_started = False
            elif event_type == 'end_game':
                game_started = False
            
            # 終端イベントでのdeltasをチェック
            if event_type in ['hora', 'ryukyoku']:
                if 'deltas' not in event:
                    print(f"警告: {event_type}イベントにdeltasがありません")
        
        return not (game_started or kyoku_started)

# すべてのファイルを検証
for file in Path('data/training').rglob('*.json.gz'):
    if not validate_mjai_file(file):
        print(f"無効なファイル: {file}")
```

### 3. データ品質チェックリスト
- [ ] すべてのゲームで完全なスコア追跡がある
- [ ] 各終端イベントでスコアのdeltasの合計がゼロになる
- [ ] 切り詰められたり破損したファイルがない
- [ ] ゲームが目標スキルレベルを表している（必要に応じて初心者ゲームをフィルタリング）
- [ ] 十分なデータ量（推奨：良好な結果のために10万ゲーム以上）

## 学習設定

### 1. GRP学習設定の作成
`config_grp.toml`を作成：

```toml
[control]
version = 4
state_file = 'models/grp.pth'
enable_amp = false
print_states = true

[grp.control]
device = 'cuda:0'
batch_size = 512
save_every = 2000
print_every = 10
val_interval = 100
val_steps = 400

[grp.dataset]
train_globs = ['data/training/**/*.json.gz']
val_globs = ['data/validation/**/*.json.gz']
enable_augmentation = true
file_batch_size = 50
file_index = 'cache/grp_file_index.pth'

[grp.network]
hidden_size = 64
num_layers = 2

[grp.optim]
lr = 0.002
weight_decay = 0.00001
```

### 2. メイン学習設定の作成
`config_main.toml`を作成：

```toml
[control]
version = 4
state_file = 'models/mortal.pth'
steps = 1000000
batch_size = 512
save_every = 10000
print_every = 100
device = 'cuda:0'
enable_amp = true
print_states = true

[dataset]
globs = ['data/training/**/*.json.gz']
enable_augmentation = true
file_batch_size = 30
num_workers = 4
file_index = 'cache/main_file_index.pth'

[grp]
state_file = 'models/grp.pth'  # 学習済みGRPモデルへのパス

[resnet]
conv_channels = 192
num_blocks = 40
initial_conv_channels = 64

[cql]
target_update_interval = 1000
min_q_weight = 5.0
num_samples = 10

[dqn]
hidden_size = 1024

[aux]
hidden_size = 1024

[optim]
lr = 0.00001
weight_decay = 0.00001
lr_scheduler = 'constant'
```

## 学習プロセス

### ステップ1: 必要なディレクトリの作成
```bash
mkdir -p models cache logs
```

### ステップ2: GRPモデルの学習
GRP（Game Result Predictor）はメイン学習で報酬計算に使用されるため、最初に学習する必要があります。

```bash
python -m mortal.train_grp --config config_grp.toml 2>&1 | tee logs/grp_training.log
```

**学習時間**: 
- 通常は5万～10万ステップで収束
- 検証精度を監視（>70%に達するべき）
- 全データセットの10%での学習で通常は十分

**期待される出力**:
```
[2024-01-01 12:00:00] step 1000, t_loss: 2.456, v_loss: 2.389, t_acc: 0.234, v_acc: 0.245
[2024-01-01 12:05:00] step 2000, t_loss: 1.823, v_loss: 1.756, t_acc: 0.456, v_acc: 0.467
...
```

### ステップ3: メインモデルの学習
GRP学習が完了し、検証精度が満足できるレベルに達したら：

```bash
python -m mortal.train --config config_main.toml 2>&1 | tee logs/main_training.log
```

**学習時間**:
- 完全な学習には通常50万～100万ステップが必要
- RTX 3090では1万ステップあたり約1～2時間
- 競技レベルのモデルの合計学習時間：2～4日

**期待される出力**:
```
[2024-01-01 14:00:00] step 1000, loss: 45.123, entropy: 2.345, grad_norm: 12.456
[2024-01-01 14:10:00] step 2000, loss: 38.456, entropy: 2.123, grad_norm: 8.234
...
```

## モニタリングと検証

### 1. 学習進捗の監視
以下の主要メトリクスを追跡：

**GRP学習の場合**:
- `t_acc`（学習精度）：>70%まで上昇するべき
- `v_acc`（検証精度）：学習精度に追従するべき
- t_accとv_accの大きなギャップは過学習を示す

**メイン学習の場合**:
- `loss`：着実に減少するべき
- `entropy`：安定しているべき（0に収束しない）
- `grad_norm`：適切な範囲内にとどまるべき（<100）

### 2. Tensorboardモニタリング（オプション）
```python
# 可視化のために学習スクリプトに追加
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/tensorboard')
```

### 3. チェックポイント管理
- モデルは`save_every`ステップごとに保存されます
- 後退に備えて複数のチェックポイントを保持
- ベストプラクティス：検証メトリクスと一緒にチェックポイントを保存

### 4. モデルのテスト
学習後、モデルをテスト：

```bash
# tsumogiri（ランダム捨て牌）ベースラインに対してテスト
python -m mortal.mahjongsoul --mortal-cfg models/mortal.pth

# または体系的な評価のためにarenaを使用
cargo run -p libriichi --bin arena -- \
    --agent mortal:models/mortal.pth \
    --agent tsumogiri \
    --games 1000
```

## トラブルシューティング

### 一般的な問題と解決策

#### 1. メモリ不足（OOM）エラー
**解決策**: 設定でbatch_sizeまたはfile_batch_sizeを減らす
```toml
batch_size = 256  # 512から減らす
file_batch_size = 15  # 30から減らす
```

#### 2. データ読み込みが遅い
**解決策**: num_workersを増やし、データがSSD上にあることを確認
```toml
num_workers = 8  # CPUコア数に基づいて増やす
```

#### 3. 学習損失が減少しない
**考えられる原因**:
- 学習率が高すぎる/低すぎる
- データ品質が不十分
- GRPモデルが適切に学習されていない

**解決策**:
- 学習率を調整: `lr = 0.00005`
- データ品質を検証
- より多くのデータでGRPを再学習

#### 4. ファイルインデックスエラー
**解決策**: キャッシュを削除して再生成
```bash
rm cache/*.pth
```

#### 5. libriichiのインポートエラー
**解決策**: Rustライブラリが正しくビルドされコピーされていることを確認
```bash
# 再ビルド
cargo clean
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
```

### パフォーマンス最適化のヒント

1. **データ読み込み**:
   - 高速アクセスのためデータをSSDに保存
   - 並列読み込みのため複数のワーカーを使用
   - より良い汎化のためデータ拡張を有効化

2. **GPU利用率**:
   - `nvidia-smi`でGPU使用状況を監視
   - 高速学習のためAMP（自動混合精度）を有効化
   - GPUメモリ使用を最大化するようバッチサイズを調整

3. **学習効率**:
   - 利用可能な場合は事前学習済みモデルから開始
   - より大きな実効バッチサイズのため勾配累積を使用
   - 検証メトリクスに基づいた早期停止を実装

## 高度なトピック

### カスタムプレイヤーフィルタリング
特定のプレイヤーでゲームをフィルタリング：
```toml
[dataset]
player_names_file = 'target_players.txt'  # 1行に1名
exclude_player_names_file = 'exclude_players.txt'
```

### 分散学習
マルチGPUセットアップの場合：
```python
python -m torch.distributed.launch --nproc_per_node=4 \
    -m mortal.train --config config_main.toml
```

### ハイパーパラメータ調整
実験すべき主要パラメータ：
- `min_q_weight`: CQL正則化 (5.0-10.0)
- `num_blocks`: ResNetの深さ (20-50)
- `conv_channels`: ネットワークの幅 (128-256)

## まとめ

Mortal AIモデルの学習は複雑ですがやりがいのあるプロセスです。成功の鍵となる要因：
1. 適切なMJAI形式の高品質な学習データ
2. 報酬計算のための適切に学習されたGRPモデル
3. 十分な計算リソースと忍耐
4. 定期的な監視とチェックポイント管理

このガイドを使用すれば、競技レベルの麻雀AIモデルを学習できるはずです。学習の成功をお祈りします！