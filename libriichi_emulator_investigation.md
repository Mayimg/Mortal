# libriichi 麻雀エミュレータ調査報告書

## 概要

libriichiは、Mortal（凡夫）プロジェクトの中核となるRust製の高性能麻雀エミュレータライブラリです。深層強化学習による麻雀AIの学習・推論に必要な機能を包括的に提供し、PyO3を通じてPythonから利用可能となっています。

## アーキテクチャ

### ライブラリ構成

libriichiは以下の主要モジュールで構成されています：

1. **state**: ゲーム状態管理
2. **algo**: 麻雀アルゴリズム
3. **agent**: AIエージェント実装
4. **arena**: 対戦シミュレーション
5. **mjai**: mjaiプロトコルインターフェース
6. **dataset**: 学習データ処理

### ビルドシステム

- **言語**: Rust (2024 edition)
- **ライブラリタイプ**: cdylib（Python用）とrlib（Rust用）の両方
- **Python統合**: PyO3を使用
- **パフォーマンス**: mimallocメモリアロケータをオプションで使用

## 主要機能

### 1. ゲーム状態管理（state モジュール）

#### PlayerState クラス
libriichiの中核となるクラスで、特定のプレイヤー視点からの完全なゲーム状態を保持します。

**主な状態情報**：
- 手牌（`tehai`）: 赤牌を除く34種類の牌の枚数
- 待ち牌（`waits`）: 各牌種が和了牌かどうか
- ドラ情報（`dora_factor`, `doras_owned`, `doras_seen`）
- 捨て牌（`kawa`）: 各プレイヤーの河の状態
- 副露（`fuuro_overview`）: 各プレイヤーの鳴き情報
- リーチ状態（`riichi_declared`, `riichi_accepted`）
- ゲーム進行状態（場風、自風、局数、本場、供託、点数など）
- シャンテン数（`shanten`）
- フリテン状態（`at_furiten`）

**主要メソッド**：
- `update_json(mjai_json)`: mjaiイベントを受け取り状態を更新
- `validate_reaction_json(mjai_json)`: アクションの妥当性を検証
- `brief_info()`: デバッグ用の状態表示
- `single_player_tables()`: 期待値計算テーブルの取得

#### ActionCandidate クラス
プレイヤーが取れる合法手のリストを管理します。

### 2. 麻雀アルゴリズム（algo モジュール）

#### agari（和了判定）
- 和了形の判定
- 役の計算
- 符計算

#### shanten（シャンテン数計算）
- 一般手のシャンテン数
- 七対子のシャンテン数
- 国士無双のシャンテン数

#### point（点数計算）
- 和了時の点数計算
- 積み棒、供託を含む

#### sp（シングルプレイヤー計算）
- 期待値ベースの最適打牌選択
- 向聴数維持/進行の判断

### 3. AIエージェント（agent モジュール）

#### 実装されているエージェント

1. **MortalBatchAgent**: 
   - Mortal AIのメインエージェント
   - ニューラルネットワークを使用
   - バッチ処理対応

2. **AkochanAgent**:
   - Akochan AIとの互換実装

3. **MjaiLogBatchAgent**:
   - mjaiログからの再現用

4. **Tsumogiri**:
   - ツモ切りのみの単純なエージェント

5. **BatchifiedAgent**:
   - 複数エージェントのバッチ処理ラッパー

### 4. 対戦シミュレーション（arena モジュール）

#### OneVsThree
- 1対3の対戦形式
- 特定のAIの性能評価用

#### TwoVsTwo
- 2対2のチーム戦形式

#### Board
- ゲームボードの管理
- 天鳳ルールに準拠した進行

### 5. mjaiプロトコル（mjai モジュール）

#### Bot クラス
- mjaiプロトコルでの通信
- 標準入出力を通じたイベントの送受信
- 他のmjai互換AIとの対戦を可能に

#### Event
- mjaiイベントの定義と処理
- 各種アクション（ツモ、打牌、ポン、チー、カン、リーチ、和了など）

### 6. データセット処理（dataset モジュール）

- mjaiログからの学習データ生成
- 特徴量の抽出とエンコーディング
- バッチ処理による効率的なデータ準備

## Python統合

### インポート方法
```python
from mortal.libriichi import (
    state,      # PlayerStateなど
    dataset,    # データセット処理
    arena,      # 対戦シミュレーション
    mjai,       # mjaiプロトコル
    stat,       # 統計処理
    consts      # 定数定義
)
```

### 使用例

#### 1. ゲーム状態の管理
```python
from mortal.libriichi.state import PlayerState

# プレイヤー0の状態を初期化
player_state = PlayerState(0)

# mjaiイベントで状態を更新
action_candidates = player_state.update('{"type": "tsumo", "actor": 0, "pai": "1m"}')

# アクションの検証
player_state.validate_reaction('{"type": "dahai", "actor": 0, "pai": "1m"}')
```

#### 2. mjaiボットの実行
```python
from mortal.libriichi.mjai import Bot

# エンジンを設定してボットを起動
bot = Bot(engine, player_id)
bot.run()  # 標準入出力でmjaiプロトコル通信
```

#### 3. 対戦シミュレーション
```python
from mortal.libriichi.arena import OneVsThree

# 1対3の対戦を設定
arena = OneVsThree(engines, config)
results = arena.run(num_games)
```

## パフォーマンス特性

- **高速処理**: Rustの低レベル最適化により、40,000半荘/時の処理速度を実現（最新ハードウェア）
- **並列処理**: Rayonによる並列化
- **メモリ効率**: カスタムメモリアロケータ（mimalloc）の使用オプション
- **バッチ処理**: 複数ゲームの同時処理による効率化

## 拡張性

libriichiは以下の点で高い拡張性を持ちます：

1. **新しいエージェントの追加**: Agentトレイトを実装することで新しいAIを追加可能
2. **ルールのカスタマイズ**: Boardクラスを拡張してルールバリエーションに対応
3. **データ形式の追加**: 新しいデータセット形式のサポート
4. **統計機能の拡張**: 新しい統計分析機能の追加

## まとめ

libriichiは、麻雀AIの研究開発に必要な機能を網羅的に提供する高性能なエミュレータライブラリです。Rustによる実装により高速な処理を実現しつつ、PyO3を通じてPythonから簡単に利用できる設計となっています。特に、完全なゲーム状態管理、高度なアルゴリズム、柔軟なエージェントシステム、mjaiプロトコル対応により、麻雀AI開発の基盤として優れた機能を提供しています。