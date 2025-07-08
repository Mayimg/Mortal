// ========================================================================
// libriichi 定数定義モジュール
// 
// このファイルは麻雀AIライブラリlibriichi内で使用される定数を定義しています。
// 主に以下を含みます：
// - バージョン管理用定数（MAX_VERSION）
// - 麻雀のアクション空間サイズ（ACTION_SPACE）
// - グループサイズ（GRP_SIZE）
// - 観測データの形状を返す関数（obs_shape, oracle_obs_shape）
// - Python バインディング用のモジュール登録関数
// ========================================================================

// Python バインディング用のヘルパー関数をインポート
// add_submodule: Python サブモジュールを追加するためのユーティリティ関数
use crate::py_helper::add_submodule;

// PyO3ライブラリのプレリュードモジュールをインポート
// PyO3はRustでPythonバインディングを作成するためのライブラリ
use pyo3::prelude::*;

// サポートされる最大バージョン番号を定義
// この定数は観測データ形状の決定などに使用される
pub const MAX_VERSION: u32 = 4;

// 麻雀における全アクションの種類数を定義
// 麻雀AIが選択可能な全てのアクションを合計した数
pub const ACTION_SPACE: usize = 37 // 0-33: 捨て牌（34種類の牌）、34-36: カン選択時の赤牌選択（5mr, 5pr, 5sr）
                              + 1  // リーチ宣言
                              + 3  // チー（上家から牌を鳴く、3種類の組み合わせ）
                              + 1  // ポン（他家から牌を鳴く）
                              + 1  // カン決定（カンを実行するかの決定）
                              + 1  // アガリ（和了宣言）
                              + 1  // 流局（九種九牌などによる流局宣言）
                              + 1; // パス（アクションをスキップ）
// = 46（合計46種類のアクション）

// ゲーム結果予測（GRP: Game Result Predictor）の特徴量数を定義
// 各局の状態を表す7つの特徴量：
// 1. grand_kyoku（通算局数：東1=0, 東2=1, ..., 南4=7, 西1=8, ...）
// 2. honba（本場：連続親番・流局回数）
// 3. kyotaku（供託：未回収のリーチ棒数）
// 4-7. 各プレイヤーのスコア（万点単位）
// これらはGRU-RNNの入力として使用され、ゲーム最終順位を予測する
pub const GRP_SIZE: usize = 7;

// Python関数として公開される属性マクロ
// この関数はPythonから呼び出し可能になる
#[pyfunction]
// インライン化のヒントをコンパイラに与える
// 小さな関数なので呼び出しオーバーヘッドを削減
#[inline]
// 観測データの形状を返すコンパイル時定数関数
// version: データフォーマットのバージョン番号
// 戻り値: (特徴量の数, チャンネル数)のタプル
// 34チャンネルは34種類の牌を表す（萬子9種+筒子9種+索子9種+字牌7種）
pub const fn obs_shape(version: u32) -> (usize, usize) {
    // バージョンに応じて異なる観測データの形状を返す
    match version {
        1 => (938, 34),   // v1: 基本的なエンコーディング（一部バグあり）
        2 => (942, 34),   // v2: ゲーム進行状況、見えている牌数、他家の手出し牌とリーチ宣言牌を追加
        3 => (934, 34),   // v3: RBFエンコーディングを削除、捨て牌の時間的エンコーディングを追加
        4 => (1012, 34),  // v4: 期待値計算機能を大幅追加（最大期待値、必要牌、確率テーブル等）
        _ => unreachable!(),  // 未定義のバージョンは到達不可能（パニック）
    }
}

// Python関数として公開される属性マクロ
#[pyfunction]
// インライン化のヒントをコンパイラに与える
#[inline]
// オラクル（完全情報）観測データの形状を返すコンパイル時定数関数
// オラクルモードでは通常より多くの情報（他家の手牌など）が観測可能
// version: データフォーマットのバージョン番号
// 戻り値: (特徴量の数, チャンネル数)のタプル
// 通常の観測データより大幅に少ない特徴量数は、完全情報により推測が不要になるため
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    // バージョンに応じて異なるオラクル観測データの形状を返す
    match version {
        1 => (211, 34),           // v1: 基本的な完全情報エンコーディング
        2 | 3 | 4 => (217, 34),   // v2-4: 改良版（6特徴量追加）、v2以降は統一仕様
        _ => unreachable!(),      // 未定義のバージョンは到達不可能（パニック）
    }
}

// クレート内部でのみ使用可能なPythonモジュール登録関数
// py: Pythonインタープリタへの参照
// prefix: モジュールのプレフィックス（名前空間）
// super_mod: 親モジュールへの参照
// 戻り値: PyResult型（成功/エラー）
pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    // "consts"という名前の新しいPythonモジュールを作成
    let m = PyModule::new(py, "consts")?;
    
    // obs_shape関数をPythonモジュールに追加
    // wrap_pyfunction!マクロでRust関数をPython関数にラップ
    m.add_function(wrap_pyfunction!(obs_shape, &m)?)?;
    
    // oracle_obs_shape関数をPythonモジュールに追加
    m.add_function(wrap_pyfunction!(oracle_obs_shape, &m)?)?;
    
    // MAX_VERSION定数をPythonモジュールに追加
    m.add("MAX_VERSION", MAX_VERSION)?;
    
    // ACTION_SPACE定数をPythonモジュールに追加
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    
    // GRP_SIZE定数をPythonモジュールに追加
    m.add("GRP_SIZE", GRP_SIZE)?;
    
    // 作成したモジュールを親モジュールのサブモジュールとして追加
    add_submodule(py, prefix, super_mod, &m)
}
