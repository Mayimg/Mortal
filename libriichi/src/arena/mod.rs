// arenaモジュールのエントリポイント。麻雀ゲームのアリーナ（対戦場）システムを提供する。
// このモジュールは、複数のAIエージェントが対戦できる環境を実装し、
// バッチゲーム実行、ゲーム結果の記録、Python バインディングなどを提供する。

mod board;         // ゲームボード（盤面）の状態管理と基本的なゲームロジック
mod game;          // ゲーム実行エンジンとバッチ処理
mod one_vs_three;  // 1対3の対戦モード（1人のチャレンジャー vs 3人のチャンピオン）
mod result;        // ゲーム結果とログ出力
mod two_vs_two;    // 2対2の対戦モード（2人のチャレンジャー vs 2人のチャンピオン）

pub use board::Board;         // ゲームボードの公開インターフェース
pub use result::GameResult;   // ゲーム結果構造体の公開

use crate::py_helper::add_submodule;  // Pythonサブモジュール追加用のヘルパー関数
use one_vs_three::OneVsThree;         // 1対3対戦クラス
use two_vs_two::TwoVsTwo;             // 2対2対戦クラス

use pyo3::prelude::*;                 // Python バインディング用のPyO3ライブラリ

// Pythonモジュールとして登録するための関数
// このモジュールをPythonから使用可能にする
pub(crate) fn register_module(
    py: Python<'_>,                   // Python インタープリタへの参照
    prefix: &str,                     // モジュールプレフィックス（通常は親モジュール名）
    super_mod: &Bound<'_, PyModule>,  // 親モジュールへの参照
) -> PyResult<()> {
    let m = PyModule::new(py, "arena")?;  // 新しいPythonモジュール "arena" を作成
    m.add_class::<OneVsThree>()?;         // OneVsThreeクラスをPythonクラスとして登録
    m.add_class::<TwoVsTwo>()?;           // TwoVsTwoクラスをPythonクラスとして登録
    add_submodule(py, prefix, super_mod, &m)  // 親モジュールのサブモジュールとして追加
}
