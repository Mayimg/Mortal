# Mortal AI 学習クイックリファレンスチェックリスト

## 学習前チェックリスト
- [ ] MJAI形式データファイルの準備（`.json.gz`）
- [ ] データの検証（完全なゲーム、適切なdeltas）
- [ ] Conda環境の作成と有効化
- [ ] CUDAサポート付きPyTorchのインストール
- [ ] Rustライブラリのビルド（`cargo build -p libriichi --lib --release`）
- [ ] libriichi.soをmortal/ディレクトリにコピー
- [ ] ディレクトリ構造の作成（models/, cache/, logs/）

## GRP学習
```bash
# 1. GRP設定ファイルの作成（config_grp.toml）
# 2. GRP学習の実行
python -m mortal.train_grp --config config_grp.toml 2>&1 | tee logs/grp_training.log

# 検証精度が70%を超えるまで監視
```

## メインモデル学習
```bash
# 1. メイン設定ファイルの作成（config_main.toml）
# 2. 設定内のGRPモデルパスが正しいことを確認
# 3. メイン学習の実行
python -m mortal.train --config config_main.toml 2>&1 | tee logs/main_training.log

# 50万～100万ステップ学習
```

## 監視すべき主要メトリクス
- **GRP**: t_acc, v_acc（目標: >70%）
- **メイン**: loss（減少）、entropy（安定）、grad_norm（<100）

## よく使うコマンド
```bash
# GPU使用状況の監視
watch -n 1 nvidia-smi

# 学習ログの追跡
tail -f logs/main_training.log

# モデルのテスト
python -m mortal.mahjongsoul --mortal-cfg models/mortal.pth
```

## トラブルシューティングクイックフィックス
- **OOM**: 設定でbatch_sizeを減らす
- **読み込みが遅い**: num_workersを増やす
- **インポートエラー**: libriichi.soを再ビルドしてコピー
- **ファイルインデックスエラー**: cache/*.pthを削除

## 推奨パラメータ
- batch_size: 512（OOMの場合は減らす）
- file_batch_size: 30（OOMの場合は減らす）
- min_q_weight: 5.0
- learning_rate: 0.00001
- num_workers: 4-8