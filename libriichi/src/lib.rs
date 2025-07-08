// libriichiクレート: 日本麻雀(リーチ麻雀)の実装を提供するRustライブラリ
// このライブラリは、麻雀のゲームロジック、AI開発、ログ解析などの機能を提供し、
// Python向けのバインディング(PyO3)も含んでいます。

// clippyの警告設定: manual_range_patternsはmatches_tu8マクロのために無効化
#![allow(clippy::manual_range_patterns)] // because of matches_tu8
// 厳格なコード品質チェックの設定
// Rust 2018エディションのイディオムやClippyの様々なリントルールを有効化
// これにより、コードの品質、パフォーマンス、保守性を向上させています
#![deny(
    rust_2018_idioms,                       // Rust 2018の推奨イディオムに従うことを強制
    let_underscore_drop,                     // let _ = ... でのdropを禁止（明示的なdropを推奨）
    clippy::assertions_on_result_states,     // Result型に対するアサーションを禁止
    clippy::bool_to_int_with_if,            // if式でboolをintに変換することを禁止
    clippy::borrow_as_ptr,                   // borrowをポインタに変換することを禁止
    clippy::cloned_instead_of_copied,        // cloned()の代わりにcopied()を使うべき場合を検出
    clippy::create_dir,                      // create_dirの代わりにcreate_dir_allの使用を推奨
    clippy::debug_assert_with_mut_call,      // debug_assert内での可変参照呼び出しを禁止
    clippy::default_union_representation,    // union型のデフォルト表現を禁止
    clippy::deref_by_slicing,                // スライシングによる参照外しを禁止
    clippy::derive_partial_eq_without_eq,    // PartialEqを実装する際にEqも実装することを推奨
    clippy::empty_drop,                      // 空のdrop実装を禁止
    clippy::empty_line_after_outer_attr,     // 外部属性の後の空行を禁止
    clippy::empty_structs_with_brackets,     // 空の構造体で{}を使用することを禁止
    clippy::equatable_if_let,                // if letの代わりに==を使うべき場合を検出
    clippy::expl_impl_clone_on_copy,         // Copy型に対する明示的なClone実装を禁止
    clippy::explicit_deref_methods,          // 明示的なderef()メソッドの使用を禁止
    clippy::explicit_into_iter_loop,         // for文での明示的なinto_iter()呼び出しを禁止
    clippy::explicit_iter_loop,              // for文での明示的なiter()呼び出しを禁止
    clippy::filetype_is_file,                // filetype.is_file()の使用を禁止
    clippy::filter_map_next,                 // filter_map().next()の代わりにfind_map()を推奨
    clippy::flat_map_option,                 // Option型に対するflat_mapを禁止
    clippy::float_cmp,                       // 浮動小数点数の直接比較を禁止
    clippy::float_cmp_const,                 // 浮動小数点数と定数の比較を禁止
    clippy::format_push_string,              // format!の結果をpush_strすることを禁止
    clippy::from_iter_instead_of_collect,    // from_iterの代わりにcollectを使うべき場合を検出
    clippy::get_unwrap,                      // get().unwrap()の代わりに[]を使うべき場合を検出
    clippy::implicit_clone,                  // 暗黙的なclone()呼び出しを禁止
    clippy::implicit_saturating_sub,         // 暗黙的な飽和減算を禁止
    clippy::imprecise_flops,                 // 不正確な浮動小数点演算を禁止
    clippy::index_refutable_slice,           // refutableなスライスインデックスを禁止
    clippy::inefficient_to_string,           // 非効率なto_string()の使用を禁止
    clippy::invalid_upcast_comparisons,      // 無効なアップキャスト比較を禁止
    clippy::iter_on_empty_collections,       // 空のコレクションに対するiterを禁止
    clippy::iter_on_single_items,            // 単一要素に対するiterを禁止
    clippy::large_types_passed_by_value,     // 大きな型を値渡しすることを禁止
    clippy::let_unit_value,                  // let _ = ()のような単位値への束縛を禁止
    clippy::lossy_float_literal,             // 精度を失う浮動小数点リテラルを禁止
    clippy::macro_use_imports,               // #[macro_use]の使用を禁止（明示的なインポートを推奨）
    clippy::manual_assert,                   // 手動でのassert実装を禁止
    clippy::manual_clamp,                    // 手動でのclamp実装を禁止
    clippy::manual_instant_elapsed,          // 手動でのInstant::elapsed実装を禁止
    clippy::manual_let_else,                 // 手動でのlet-else実装を禁止
    clippy::manual_ok_or,                    // 手動でのok_or実装を禁止
    clippy::manual_string_new,               // 手動でのString::new実装を禁止
    clippy::map_unwrap_or,                   // map().unwrap_or()の代わりにmap_or()を推奨
    clippy::match_bool,                      // bool値に対するmatchを禁止
    clippy::match_same_arms,                 // 同じ処理を行うmatchアームを禁止
    clippy::missing_const_for_fn,            // const fnにできる関数を検出
    clippy::mut_mut,                         // &mut &mutを禁止
    clippy::mutex_atomic,                    // Mutexの代わりにatomicを使うべき場合を検出
    clippy::mutex_integer,                   // 整数値のMutexを禁止（Atomicを推奨）
    clippy::naive_bytecount,                 // 非効率なバイトカウントを禁止
    clippy::needless_bitwise_bool,           // 不要なビット演算を禁止
    clippy::needless_collect,                // 不要なcollect()を禁止
    clippy::needless_continue,               // 不要なcontinueを禁止
    clippy::needless_for_each,               // 不要なfor_each()を禁止
    clippy::nonstandard_macro_braces,        // 非標準のマクロ括弧を禁止
    clippy::or_fun_call,                     // or()の代わりにor_else()を使うべき場合を検出
    clippy::path_buf_push_overwrite,         // PathBuf::pushによる上書きを禁止
    clippy::ptr_as_ptr,                      // ポインタからポインタへのキャストを禁止
    clippy::range_minus_one,                 // range - 1の代わりに..=を推奨
    clippy::range_plus_one,                  // range + 1の代わりに..=を推奨
    clippy::redundant_else,                  // 冗長なelseブロックを禁止
    clippy::rest_pat_in_fully_bound_structs, // 完全に束縛された構造体での..パターンを禁止
    clippy::semicolon_if_nothing_returned,   // 何も返さない場合のセミコロンを要求
    clippy::significant_drop_in_scrutinee,   // match式の検査対象での重要なdropを禁止
    clippy::str_to_string,                   // &strに対するto_string()を禁止
    clippy::string_add,                      // String + &strを禁止（push_strを推奨）
    clippy::string_add_assign,               // String += &strを禁止
    clippy::string_lit_as_bytes,             // 文字列リテラルのas_bytes()を禁止
    clippy::string_to_string,                // Stringに対するto_string()を禁止
    clippy::suboptimal_flops,                // 最適でない浮動小数点演算を禁止
    clippy::suspicious_to_owned,             // 疑わしいto_owned()の使用を禁止
    clippy::trait_duplication_in_bounds,     // トレイト境界の重複を禁止
    clippy::trivially_copy_pass_by_ref,      // Copy型の参照渡しを禁止
    clippy::type_repetition_in_bounds,       // 型境界での型の重複を禁止
    clippy::unchecked_duration_subtraction,  // チェックなしのDuration減算を禁止
    clippy::undocumented_unsafe_blocks,      // ドキュメントなしのunsafeブロックを禁止
    clippy::unicode_not_nfc,                 // NFC正規化されていないUnicodeを禁止
    clippy::uninlined_format_args,           // インライン化されていないformat引数を禁止
    clippy::unnecessary_join,                // 不要なjoin()を禁止
    clippy::unnecessary_self_imports,        // 不要なself importを禁止
    clippy::unneeded_field_pattern,          // 不要なフィールドパターンを禁止
    clippy::unnested_or_patterns,            // ネストされていないORパターンを禁止
    clippy::unseparated_literal_suffix,      // 区切りのないリテラルサフィックスを禁止
    clippy::unused_peekable,                 // 使用されていないPeekableを禁止
    clippy::unused_rounding,                 // 使用されていない丸め処理を禁止
    clippy::use_self,                        // Self型の使用を推奨
    clippy::used_underscore_binding,         // _で始まる束縛の使用を禁止
    clippy::useless_let_if_seq               // 無意味なlet-if連鎖を禁止
)]

// プライベートモジュール（クレート内部でのみ使用）
mod arena;      // 対戦環境（自己対戦など）の実装
mod array;      // 配列操作のユーティリティ
mod consts;     // 定数定義（観測空間・行動空間など）
mod dataset;    // 訓練データセットの生成・処理
mod macros;     // マクロ定義
mod py_helper;  // Python連携用のヘルパー関数
mod rankings;   // ランキング関連の処理
mod vec_ops;    // ベクトル演算のユーティリティ

// バイナリクレート（実行可能ファイル）から使用するための公開モジュール
pub mod chi_type;  // チー（鳴き）の種類の定義
pub mod mjai;      // MJAIプロトコルの実装（外部AIとの通信）
pub mod stat;      // 統計処理（ログ解析など）
pub mod state;     // ゲーム状態管理（プレイヤー状態、盤面状態など）

// テスト以外のコードから使用するための公開モジュール
pub mod agent;     // AIエージェントの基底実装
pub mod tile;      // 牌の定義と操作

// ベンチマーク用に公開するモジュール
pub mod algo;      // アルゴリズム（シャンテン数計算、和了判定など）
pub mod hand;      // 手牌の表現と操作

// PyO3: PythonとRustのバインディングライブラリ
// これによりRustコードをPythonから呼び出せるようになる
use pyo3::prelude::*;

// mimallocフィーチャーが有効な場合、グローバルアロケータとしてMiMallocを使用
// MiMallocは高性能なメモリアロケータで、特にマルチスレッド環境での性能が優れている
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// libriichiモジュール: リーチ麻雀の実装を提供
/// 
/// 主な機能:
/// - コア機能: mjaiイベントによるプレイヤー状態管理（`state.PlayerState`経由）
/// - mjaiログの読み込みと訓練用インスタンスのバッチ生成（`dataset`経由）
/// - 天鳳ルールでの自己対戦（`arena`経由）
/// - Mortal用の観測空間・行動空間の定義（`consts`経由）
/// - mjaiログの統計処理（`stat.Stat`経由）
/// - mjaiインターフェース（`mjai.Bot`経由）
#[pymodule]  // PyO3のアトリビュート: このモジュールをPythonモジュールとして公開
fn libriichi(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // pyo3_logの初期化: RustのログをPythonのloggingモジュールに転送
    pyo3_log::init();
    // シャンテン数計算テーブルの初期化（一度だけ実行される）
    algo::shanten::ensure_init();
    // 和了判定テーブルの初期化（一度だけ実行される）
    algo::agari::ensure_init();

    // モジュール名を取得
    let name = m.name()?;
    let name = name.extract()?;
    // デバッグビルドかリリースビルドかを判定し、警告とプロファイル情報を設定
    if cfg!(debug_assertions) {
        // デバッグビルドの場合は標準エラー出力に警告を表示
        eprintln!("{name}: this is a debug build.");
        // Pythonモジュールに__profile__属性として"debug"を追加
        m.add("__profile__", "debug")?;
    } else {
        // リリースビルドの場合は__profile__属性として"release"を追加
        m.add("__profile__", "release")?;
    }
    // Cargo.tomlのバージョン情報をPythonモジュールの__version__属性として追加
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // 各サブモジュールをPythonモジュールに登録
    // それぞれのregister_module関数内で、Pythonから呼び出せる関数やクラスが定義される
    consts::register_module(py, name, m)?;   // 定数定義（観測・行動空間など）
    state::register_module(py, name, m)?;    // ゲーム状態管理
    dataset::register_module(py, name, m)?;  // データセット処理
    arena::register_module(py, name, m)?;    // 対戦環境
    stat::register_module(py, name, m)?;     // 統計処理
    mjai::register_module(py, name, m)?;     // MJAIインターフェース

    Ok(())  // 成功を返す
}
