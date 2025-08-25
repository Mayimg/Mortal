# ACH 使い方とパラメータ説明

このドキュメントは、ACH（Actor-Critic Hedge）学習ループの実行手順と、主なパラメータの説明をまとめたものです。アルゴリズム詳細は ACH_Codex.md を参照してください。


## 実行前提

- オンライン学習パイプライン（mortal.server / mortal.client）が有効であること。
- libriichi 側（Rust）は変更不要。Python 側のみで完結します。
- 既存の DQN/CQL 学習（mortal.train.py）はそのまま併存します。


## 最小設定

`config.toml` に以下のキーを追加/確認してください。

```
[control]
algo = 'ach'     # ACH トレーナを利用
online = true   # ACH はオンライン収集のみ対応

[ach]
eta = 1.0              # Hedge 係数
logit_threshold = 6.0  # ロジットしきい値（±値でクリップ）
ratio_clip = 0.5       # PPO の比率クリップ ε
gae_lambda = 0.95      # 便宜上保持（実質 MC）
entropy_coef = 1e-2    # エントロピー係数 β
value_coef = 0.5       # 価値損失の係数 α
batch_size = 8192

[optim]
lr = 2.5e-4
lr_scheduler = 'constant'

[env]
gamma = 0.995
```

クライアント側（自己対戦生成）の温度/εは、ACH 方策に合わせてできるだけ素直なサンプリングにすることを推奨します（例: `boltzmann_epsilon=0.0`, `boltzmann_temp=1.0`, `top_p=1.0`）。


## 実行手順

1) サーバ起動（サンプル受け渡し）
- `python -m mortal.server`

2) クライアント起動（自己対戦ログ収集）
- `python -m mortal.client`
- `control.algo='ach'` のとき、クライアントは `PolicyNet` をロードします（互換のため server 側キーは `dqn` を使用）。

3) 学習起動（トレーナ）
- `python -m mortal.run_train`
- `control.algo='ach'` により `mortal.train_ach.train()` が呼ばれます。学習中は `submit_every` 間隔で最新パラメータをサーバへ送信し、クライアントが常に新しい方策で自己対戦します。


## ログと保存物

- TensorBoard 出力: `logs/tensorboard_*`（`config.control.tensorboard_dir`）
  - `ach/policy_loss`, `ach/value_loss`, `ach/entropy`, `ach/ratio_mean`, `ach/ratio_clip_rate` など
- チェックポイント: `config.control.state_file` に `mortal`, `current_dqn(=policy)`, `value`, `optimizer`, `steps` を保存。
- ベスト更新時: `config.control.best_state_file` にコピーし、server にパラメータを送信（自己対戦側が自動更新）。


## 主なパラメータ（ACH）

- `ach.eta`（float, 既定: 1.0）
  - Hedge 係数。ロジット y に対し `Softmax(eta * y)` をとるスケール。
- `ach.logit_threshold`（float, 既定: 6.0）
  - 状態ごとにロジット平均を減算後、`[-th, +th]` にクリップ。数値安定と学習安定化に寄与。
- `ach.ratio_clip`（float, 既定: 0.5）
  - PPO の比率クリップ ε。`π/π_old` の逸脱が大きい更新をゲートで遮断します。
- `ach.gae_lambda`（float, 既定: 0.95）
  - 論文値。麻雀の報酬構造では MC と実質等価のため、`A = G - V` 実装で問題ありません。
- `ach.entropy_coef`（float, 既定: 1e-2）
  - エントロピー正則化の係数。探索と安定化のトレードオフ。
- `ach.value_coef`（float, 既定: 0.5）
  - 価値損失の係数。
- `ach.batch_size`（int, 既定: `control.batch_size`）
  - 1 ミニバッチのサイズ。

- `optim.lr`（float, 既定: 2.5e-4）
  - AdamW の学習率。
- `env.gamma`（float, 既定: 0.995）
  - エピソード内割引。局末報酬への重みづけに影響。


## 互換性と注意事項

- 既存 DQN と共存します。`current_dqn` キーには互換のため PolicyNet の状態を保存します。
- server 側のパラメータフィールド名は従来通り `mortal` と `dqn` を利用（`dqn` に PolicyNet を格納）。
- 方策サンプリングを論文の `Softmax(eta·y)` に近づけるには、クライアントの `boltzmann_epsilon=0.0`, `boltzmann_temp=1.0` を推奨します。


## 参考

- アルゴリズム全体と理論背景は ACH_Codex.md を参照。
