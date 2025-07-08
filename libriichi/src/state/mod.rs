// ===== libriichi/src/state/mod.rs =====
// 
// このファイルは、麻雀ゲームの状態管理に関するモジュールを定義し、整理するためのモジュール定義ファイルです。
// 主な役割：
// 1. ゲーム状態に関連する各サブモジュールの宣言
// 2. 重要な型（ActionCandidate、PlayerState、SinglePlayerTables）の再エクスポート
// 3. Python バインディング（PyO3）を使用して、Rust の型を Python から使用可能にする登録関数の提供
//
// このモジュールは、麻雀ゲームの状態を表現し、プレイヤーの視点から見たゲーム情報を管理するための
// 中核的な機能を提供します。

// action モジュール：プレイヤーが取り得るすべての行動（打牌、チー、ポン、カン、リーチ、和了など）を
// 表現する ActionCandidate 構造体を定義
mod action;

// agent_helper モジュール：AI エージェントやボットが状態を扱う際の補助機能を提供
mod agent_helper;

// getter モジュール：PlayerState からの情報取得用のゲッターメソッド群を定義
mod getter;

// item モジュール：ゲーム内のアイテムや要素に関する定義
mod item;

// obs_repr モジュール：観測可能な状態の表現（representation）に関する機能
// 深層学習モデルへの入力として使用される特徴量のエンコーディングなどを扱う
mod obs_repr;

// player_state モジュール：プレイヤーの視点から見たゲーム状態を管理する
// PlayerState 構造体を定義（手牌、捨て牌、鳴き、ドラ、リーチ状態など）
mod player_state;

// sp_tables モジュール：シングルプレイヤー用の期待値テーブルを管理する
// SinglePlayerTables 構造体を定義
mod sp_tables;

// update モジュール：ゲーム状態の更新処理に関する機能を提供
mod update;

// テスト環境でのみコンパイルされるテストモジュール
// cargo test 実行時にのみ含まれる
#[cfg(test)]
mod test;

// Python バインディング用のヘルパー関数をインポート
// add_submodule は Python のモジュールシステムにサブモジュールを登録する
use crate::py_helper::add_submodule;

// action モジュールから ActionCandidate を公開再エクスポート
// これにより、外部からは libriichi::state::ActionCandidate として使用可能
pub use action::ActionCandidate;

// player_state モジュールから PlayerState を公開再エクスポート
// ゲーム状態管理の中核となる構造体
pub use player_state::PlayerState;

// sp_tables モジュールから SinglePlayerTables を公開再エクスポート
// シングルプレイヤーモードでの期待値計算に使用
pub use sp_tables::SinglePlayerTables;

// PyO3 ライブラリのプレリュード（よく使う型や関数）をインポート
// Python との相互運用性を提供
use pyo3::prelude::*;

// Python モジュールとして登録するための関数
// crate 内部からのみアクセス可能（pub(crate)）
pub(crate) fn register_module(
    py: Python<'_>,      // Python インタプリタへの参照（ライフタイム付き）
    prefix: &str,        // モジュールのプレフィックス（例："libriichi"）
    super_mod: &Bound<'_, PyModule>, // 親モジュールへの参照（Bound は PyO3 の新しい API）
) -> PyResult<()> {      // PyResult は Python エラーを Rust で扱うための Result 型
    // "state" という名前の新しい Python モジュールを作成
    let m = PyModule::new(py, "state")?;
    
    // ActionCandidate クラスを Python モジュールに追加
    // これにより Python から ActionCandidate を使用可能になる
    m.add_class::<ActionCandidate>()?;
    
    // PlayerState クラスを Python モジュールに追加
    // Python から麻雀ゲームの状態を操作できるようになる
    m.add_class::<PlayerState>()?;
    
    // 作成したモジュールを親モジュールに追加し、Python の sys.modules に登録
    // これにより、Python から import libriichi.state などでアクセス可能になる
    add_submodule(py, prefix, super_mod, &m)
}
