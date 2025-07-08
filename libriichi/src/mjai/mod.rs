// mjaiモジュール - 麻雀AIプロトコルであるmjaiフォーマットのサポートを提供
// 
// このモジュールは、麻雀AIの対戦プロトコルであるmjaiフォーマットでの通信をサポートします。
// mjaiは日本の麻雀AI研究で広く使われているJSON形式のイベントベースプロトコルです。
// 
// 主な機能：
// - mjaiイベントの定義とシリアライズ/デシリアライズ
// - イベントの処理と状態管理
// - PythonバインディングのためのBot実装

mod bot; // mjaiボットの実装を含むモジュール
mod event; // mjaiイベントの定義を含むモジュール

// eventモジュールから主要な型を再エクスポート
// Event: mjaiイベントの列挙型
// EventExt: メタデータ付きの拡張イベント型
// EventWithCanAct: アクション可能フラグ付きイベント型
// Metadata: イベントに付随するメタデータ（AI評価値など）
// OutOfBoundError: 範囲外エラー
pub use event::{Event, EventExt, EventWithCanAct, Metadata, OutOfBoundError};

use crate::py_helper::add_submodule; // Pythonサブモジュール追加のヘルパー関数
use bot::Bot; // mjaiボットクラス

use pyo3::prelude::*; // Python-Rustバインディング用のマクロと型

// Pythonモジュールとして登録するための関数
// py: Python実行環境への参照
// prefix: モジュールのプレフィックス（パス）
// super_mod: 親モジュールへの参照
pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "mjai")?; // "mjai"という名前で新しいPythonモジュールを作成
    m.add_class::<Bot>()?; // BotクラスをPythonモジュールに追加（Python側から利用可能にする）
    add_submodule(py, prefix, super_mod, &m) // 作成したモジュールを親モジュールのサブモジュールとして追加
}
