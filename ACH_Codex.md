# Actor-Critic Hedge (ACH) 実装設計書（Mortal）

本書は、ICLR 2022 掲載の「Actor-Critic Policy Optimization in a Large Scale Imperfect-Information Game」にて提案された Actor-Critic Hedge（ACH）を、本プロジェクト（Mortal/libriichi）に実装するための完全な技術仕様です。元論文を再参照せずに実装できることを目的に、アルゴリズムの数式・擬似コード・ハイパーパラメータ・既存コードへの組み込み手順を詳細に記載します。

対象要件:
- online=true のときに ACH での学習が可能にする
- 既存モデル/エージェントは維持（ACH は追加の新規アルゴリズム/モデルとして実装）
- 可能な限り既存実装の再利用（Brain 特徴抽出やオンライン収集の枠組みなど）
- libriichi（Rust の対戦エンジン）へ変更は加えない（mortal 側のみ拡張）
- 論文のハイパーパラメータに最大限一致


## 1. アルゴリズム概要

ACH は、CFR の収束性（Hedge による後悔最小化）と、Deep RL のスケーラビリティ（actor-critic）を両立するオンライン自己対戦アルゴリズムです。状態 s に対する方策 π の更新を「Hedge（指数重み付け）」で行い、その際に使用する累積（加重）反事実後悔量をニューラルネットで近似します。実践的実装（ACH）は、以下の特徴を持ちます。

- 方策ネット y(a|s; θ)（出力は各合法手 a のロジット）と、価値ネット V(s; ω) を併用（多くのパラメータを共有）。
- 現行ポリシー π を用いた自己対戦でサンプルを収集（µp,t = πp,t）。
- 方策更新は Mirror Descent（KL 正則化）に対応する Hedge で、閉形式解 π(a|s) ∝ exp(η(s)·y(a|s; θ)) を用いる。
- 学習は分散 Actor-Learner で非同期を許容し、PPO の比率クリップで安定化。
- エントロピー正則化を加えて現在方策の安定化と探索を促進。


## 2. 数式定義と完全仕様

記法:
- s: 状態（infoset）。A(s): 合法手集合。a ∈ A(s)。
- πt: 反復 t での現在方策。µt: 行動方策（本実装では µt = πt）。
- y(a|s; θ): 方策ネット出力（ロジット）。
- η(s) > 0: Hedge 係数（状態依存でも定数でも可）。
- V(s; ω): 価値ネット。G: 割引累積報酬（エピソード内で得られる実サンプルのリターン）。
- A(s, a): 優位度（アドバンテージ）。本実装では GAE もしくは MC に相当。

2.1 方策の生成（Hedge）
- 反復 t の実行時方策は、直前の θt−1 を固定して
  πt(a|s) = Softmax(η(s)·y(a|s; θt−1)) （合法手のみ正規化）

2.2 価値とアドバンテージ推定
- 学習用サンプル [s, a, G] から価値損失を
  L_value = 1/2 · (V(s; ω) − G)^2
 で最小化。
- アドバンテージは GAE(λ) が原則。ただし本プロジェクトのデータは局収益（kyoku_rewards）で最終時にのみ報酬が生じるため、GAE は MC に一致します。実装では簡潔に
  A(s, a) = G − V(s; ω)
 で代替可能（論文表現の GAE(λ=0.95)とも整合性が取れる：中間報酬が 0 のとき同等）。

2.3 非同期安定化（PPO クリップとロジットしきい値）
- サンプル時の方策確率 πold(a|s) を用い、重要度比 r = π(a|s; θ) / πold(a|s) を導入。PPO と同様に r を [1−ε, 1+ε] にクリップした係数 c を使う：
  - A(s,a) ≥ 0 のとき、r ≤ 1+ε かつ ロジット中心化後が < l_th のとき寄与。
  - A(s,a) < 0 のとき、r ≥ 1−ε かつ ロジット中心化後が > −l_th のとき寄与。
- 数値安定化として、各状態ごとにロジットから平均 ȳ(·|s; θ) を減算し、さらに [−l_th, l_th] にクリップ。

2.4 方策（Actor）損失と全体損失
- 方策損失（期待値の符号に注意。論文に従い最大化方向の符号で表記）
  L_policy = − c · η(s) · (y(a|s; θ) / πold(a|s)) · A(s, a)
- エントロピー正則化（β>0）
  L_ent = β · Σ_a π(a|s; θ) · log π(a|s; θ)
- 価値損失（係数 α）
  L_value = α/2 · (V(s; ω) − G)^2
- 1 ミニバッチに対する合計損失：
  L_ACH = Σ_{(s,a)} [ L_policy + L_value + L_ent ]

2.5 近似 NW-CFR と Hedge との関係
- 累積（加重）反事実後悔 R^a_t(s,a) に対し、y(a|s; θ) ≈ R^a_t(s,a) を回帰し、Hedge で π を生成する枠組み（NW-CFR）。実践実装の ACH は「現在方策＋エントロピー正則化＋非同期クリップ」による近似で、前節の損失で 1 ミニバッチごとに θ, ω を同時に 1 回更新する。


## 3. 擬似コード（本リポジトリ適用版）

スレッド/プロセス構成: 既存の online クライアント・サーバ・トレーナの枠組みを流用。

Actors（mortal.client → mortal.player.TrainPlayer → mortal.engine.MortalEngine）
1. 学習器から最新パラメータ（θ, ω）を受領し、πold(a|s) = Softmax(η·y(a|s; θ)) で行動。
2. 自己対戦ログから時系列サンプルを抽出し、(obs=s, action=a, mask, G, …) を保存。
   - 追加で πold は保存不要。学習側で「当該バッファ生成に用いた θ を固定」して再計算できるため（本枠組みではサンプル生成直後に学習に供する）。
3. サーバへアップロード（現行のバッファ転送を使用）。

Learner（mortal.train_ach（新規））
1. サーバから drain() でログを取得し、DataLoader（オンライン分岐）でミニバッチを組む。
2. ミニバッチごとに：
   - s を Brain に通して特徴 φ を得る。
   - πold(a|s) は固定スナップショット θ_old から再計算（θ_old は drain 時点の方策）。
   - 価値 V(s; ω) を前向き計算。G から A(s,a)=G−V(s) を計算。
   - 重要度比 r=π(a|s;θ)/πold(a|s) を計算し、PPO 方式で c を決定。y を平均減算し [−l_th, l_th] にクリップ。
   - L_ACH を計算し、θ と ω を同時に 1 回更新。
3. 保存/評価/送信は既存の仕組みに準拠（TensorBoard、best_state、submit_param 等）。加えて、ACH の要件（Algorithm 2: Actors fetch the latest model）に合わせ、`submit_every` ごとに学習中の最新パラメータをサーバへ定期送信し、自己対戦側が常に最新方策を取得できるようにする。


## 4. ハイパーパラメータ（1-on-1 Mahjong 実験の推奨値）

論文に準拠（ICLR 2022 付録 H.1 Table 5）:
- Ratio clip ε: 0.5
- GAE λ: 0.95（本実装では MC と等価。A=G−V）
- 学習率（Adam）: 2.5e-4（方策・価値を同時更新の単一路線）
- 割引率 γ: 0.995
- 価値損失係数 α: 0.5
- エントロピー係数 β: 1e-2
- バッチサイズ: 8192
- ロジットしきい値 l_th: 6.0
- Hedge 係数 η(s): 1.0（定数運用）

注意:
- 既存 config は scheduler を使っていますが、ACH では論文同様「定学習率」での安定運用が基本です（`optim.lr_scheduler='constant'` を推奨）。
- 既存の `env.gamma` は 1 になっているため、ACH 実験時は 0.995 に変更することを推奨します。


## 5. 既存コードへの統合方針（差分設計）

libriichi（Rust）には一切変更を加えない。mortal 側に ACH 用の最小限の新規ファイルと分岐を追加する。

5.1 新規ファイル（例）
- `mortal/policy_ach.py`
  - `class PolicyNet(nn.Module)`: Brain 出力 φ とアクションマスクからロジット y を生成。出力後に「状態ごとの平均減算」「[−l_th, l_th] クリップ」は損失計算側で実施可。
- `mortal/value_head.py`
  - `class ValueHead(nn.Module)`: φ からスカラー V(s) を出力（軽量 MLP）。
- `mortal/train_ach.py`
  - DataLoader(オンライン)からバッチを受け取り、L_ACH で θ, ω を同時更新する学習ループ実装。
  - 既存の保存、評価、TensorBoard 出力（loss, KL/ratio 分布, entropy, value MSE など）を踏襲。

5.2 既存ファイルの変更点（最小限）
- `mortal/run_train.py`
  - `config['control']['algo']` が `'ach'` 且つ `online=true` の場合、`from mortal.train_ach import train as train_ach` を呼ぶ分岐を追加。既存 `mortal.train.train()` は不変更。
- `mortal/engine.py`
  - ACH 実行時に、`MortalEngine` が DQN ではなく `PolicyNet` を用いて `π(a|s)=Softmax(η·y)` からサンプリング/greedy を行う分岐を追加（例: `policy_mode='ach'`）。
  - 既存の DQN ベース（greedy/ボルツマン/top-p）には影響しないようオプション化。
- `mortal/model.py`（または新規ファイル）
  - 既存 `Brain` を再利用し、`PolicyNet` と `ValueHead` を同じ φ から分岐させる薄いラッパ（初期化と forward の整合）を追加。
- `mortal/config.py` / `config.toml`
  - ACH 用の設定ブロックを追加（下記 5.3）。

変更しないもの:
- 既存の `mortal/train.py`（CQL+DQN 学習）、`mortal/player.py`（評価ロジック）等は保持。
- librichi の Rust 側（対戦エンジン、ログ/データセットフォーマット）は変更しない。

5.3 設定（config.toml）への追加キー（例）
```toml
[control]
algo = 'ach'  # 'ach' のときに ACH 学習ループへ分岐（online=true 前提）

[ach]
eta = 1.0            # Hedge 係数 η(s)
logit_threshold = 6.0  # l_th
ratio_clip = 0.5     # ε
gae_lambda = 0.95
entropy_coef = 1e-2  # β
value_coef = 0.5     # α
batch_size = 8192

[optim]
lr = 2.5e-4
lr_scheduler = 'constant'

[env]
gamma = 0.995
```


## 6. 実装詳細（ロス計算とバッチ整形）

6.1 バッチ構造（オンライン・データローダ）
- 既存のオンライン分岐は `drain()` → `FileDatasetsIter` → DataLoader で
  `(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)` を返します。
- ACH では `G = (gamma ** steps_to_done) * kyoku_rewards` をそのまま利用（既存 `train.py` と同じ計算）。
- πold(a|s) は「当該 drain 周期で固定した θ_old」で再計算するため、ログ拡張は不要。

6.2 ロジット前処理
- y ← PolicyNet(φ, mask)
- 状態ごとに平均減算: y ← y − mean(y[mask])
- 区間クリップ: y ← clamp(y, −l_th, l_th)
- π(a|s; θ) = Softmax(η·y)[mask]

6.3 重要度比と係数 c
- r = π(a|s; θ) / πold(a|s)
- if A ≥ 0 かつ r ≤ 1+ε かつ y_centered < l_th なら c = 1、else c = 0。
- if A < 0 かつ r ≥ 1−ε かつ y_centered > −l_th なら c = 1、else c = 0。

6.4 損失
- L_policy = − c · η · (π(a|s; θ) / πold(a|s)) · A
- L_value = α/2 · (V(s; ω) − G)^2
- L_ent = β · Σ_a π(a|s; θ) log π(a|s; θ)
- L_ACH = Σ(L_policy + L_value + L_ent)

6.5 勾配更新
- 1 ミニバッチにつき 1 回、θ と ω を同時に更新（Adam, lr=2.5e-4）
- クリッピング: 既存の `max_grad_norm` 設定と統合可。
- AMP/scheduler/TensorBoard は既存の仕組みを活用。


## 7. 評価とログ

- 既存の `TestPlayer` と `OneVsThree` により、ベースライン DQN モデルとの対戦評価を維持。
- TensorBoard: policy/value loss、entropy、比率 r のヒストグラム、V と G の分布、平均 KL（任意）を追跡。


## 8. 実装差分一覧（ファイル単位）

新規:
- `mortal/policy_ach.py`: PolicyNet 実装（φ→ロジット）。
- `mortal/value_head.py`: ValueHead 実装（φ→V）。
- `mortal/train_ach.py`: ACH 学習ループ（オンライン専用）。

更新:
- `mortal/run_train.py`: `control.algo=='ach'` かつ `online==true` で `train_ach()` を呼ぶ分岐追加。
- `mortal/engine.py`: `policy_mode='ach'` 時に PolicyNet + Softmax(η·y) で行動（従来 DQN 分岐と共存）。
- `config.toml`: 5.3 のキーを追加。既存のキーは保持。

非変更:
- `mortal/train.py`（DQN/CQL 学習）や既存モデル類はそのまま。
- libriichi（Rust コア）への変更は一切不要。


## 9. 既存実装の再利用と設計上の注意

- Brain（特徴抽出）は共通化し、Actor/Value は軽量ヘッドで分岐。これにより既存の重い ResNet を使い回しつつ、ヘッド差し替えのみで DQN/ACH を切り替え可能。
- オンライン収集の枠組み（server/client/drain）はそのまま利用。πold は θ_old で再計算できるため、ログ形式変更不要。
- GAE(λ) は局末報酬の性質上、実質 MC と等価。論文値（λ=0.95）を保持しつつ、実装は A=G−V で簡潔にできる。
- 非同期性は本枠組み（サンプル生成→即学習→次パラメータ配布）で小さく、PPO クリップ（ε=0.5）で追加安定化。
- 行動マスクは Softmax 正規化前に無合法手へ −inf を与える処理で完全に対応。


## 10. 実装手順（推奨ロードマップ）

1) 方策・価値ヘッドの実装
- PolicyNet と ValueHead を追加。Brain 出力 φ を受け取り、マスク対応のロジット/価値を出す。

2) 学習ループ（train_ach.py）
- DataLoader(オンライン)からのミニバッチで L_ACH を計算し、θ, ω を同時更新。
- TensorBoard と checkpoint/save/best 提交は既存と同様に実装。

3) 実行系（engine.py, run_train.py）
- `policy_mode='ach'` を追加し、π=Softmax(η·y) で行動。
- `control.algo='ach'` 分岐を `run_train.py` に追加。

4) 設定
- `config.toml` に 5.3 の ACH ブロックを追加。online=true を維持。

5) 検証
- オンライン短時間スモーク（8192 バッチ）で loss/entropy/比率クリップ挙動を確認。
- 既存ベースラインとの対戦（`test_play`）で勝率/平均順位/平均点を記録。


## 11. 既知の落とし穴と対処

- πold の取り扱い: ログへ保存せずとも、サンプル収集直後の θ_old を保持して再計算可能。非同期が大きい構成に拡張したい場合は、πold をログに埋め込む案もあるが、本要件では不要。
- ロジットの数値安定: 状態ごとの平均減算とクリップ（±6.0）を必ず行う。
- γ と報酬の疎性: γ=0.995 を推奨。局末一括報酬でも A=G−V で問題なく作動。
- 既存 DQN と共存: ファイル分離と分岐で完全共存。既存スクリプト・評価も不変更。


## 12. 参考設定（最小追加差分の例）

config.toml（差分）:
- [control] に `algo='ach'`
- [env] に `gamma=0.995`
- [ach] ブロック（η, l_th, ε, λ, α, β, batch_size）
- [optim] を `lr=2.5e-4`, `lr_scheduler='constant'` に設定

実行:
- サーバ起動: `python -m mortal.server`
- クライアント起動: `python -m mortal.client`
- 学習起動: `python -m mortal.run_train`（`control.algo='ach'` かつ `online=true`）


---

本設計に従えば、libriichi（Rust コア）を変更せず、mortal 側に ACH を追加実装できます。既存 DQN/CQL 系と完全に共存し、オンライン学習モードのみで ACH を有効化可能です。ハイパーパラメータは論文に整合し、将来的な調整も設定ファイルから可能です。
