// モジュール概要：
// このファイルは麻雀ゲームデータセット用のモジュールの定義ファイルです。
// ゲームプレイ、GRP（Game Record Parser）、不可視データ（他家の手牌情報）などの
// サンプルデータ抽出機能を提供し、PythonからアクセスできるようPyO3バインディングを
// 通じてPythonモジュールとして公開しています。

//! Sample extractions.
//! サンプルデータの抽出機能を提供するモジュール

// gameplay.rsモジュールを宣言
// ゲームプレイに関するデータ処理を担当
mod gameplay;

// grp.rsモジュールを宣言  
// GRP（Game Record Parser）形式のデータ処理を担当
mod grp;

// invisible.rsモジュールを宣言
// 不可視情報（他プレイヤーの手牌など見えない情報）の処理を担当
mod invisible;

// Pythonサブモジュールの追加を支援するヘルパー関数をインポート
use crate::py_helper::add_submodule;

// gameplayモジュールから主要な構造体を公開
// Gameplay: ゲームプレイデータの本体
// GameplayLoader: ゲームプレイデータのローダー
pub use gameplay::{Gameplay, GameplayLoader};

// grpモジュールからGrp構造体を公開
// GRP形式のデータを扱う構造体
pub use grp::Grp;

// invisibleモジュールからInvisible構造体を公開
// 不可視情報を扱う構造体
pub use invisible::Invisible;

// PyO3（PythonのRustバインディング）のプレリュードをインポート
// Pythonインターフェースを作成するために必要
use pyo3::prelude::*;

// Pythonモジュールとして登録する関数
// py: Python実行環境のGIL（Global Interpreter Lock）を保持したPythonコンテキスト
// prefix: モジュール名のプレフィックス（親モジュールのパス）
// super_mod: 親となるPythonモジュール（このモジュールが追加される先）
// 戻り値: PyResult<()> - Python操作の成功/失敗を表す結果型
pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    // "dataset"という名前の新しいPythonモジュールを作成
    let m = PyModule::new(py, "dataset")?;
    
    // GameplayクラスをPythonモジュールに追加
    // ゲームプレイデータを扱うPythonクラスとして公開
    m.add_class::<Gameplay>()?;
    
    // GameplayLoaderクラスをPythonモジュールに追加
    // ゲームプレイデータを読み込むローダークラスとして公開
    m.add_class::<GameplayLoader>()?;
    
    // GrpクラスをPythonモジュールに追加
    // GRP形式のデータを扱うPythonクラスとして公開
    m.add_class::<Grp>()?;
    
    // 作成したモジュールを親モジュールのサブモジュールとして追加
    // py_helperのヘルパー関数を使用してモジュールの階層構造を構築
    add_submodule(py, prefix, super_mod, &m)
}
