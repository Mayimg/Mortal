//! シングルプレイヤー麻雀計算機のRust実装モジュール
//! 
//! このモジュールは、nekobeanのC++実装をRustに移植したもので、
//! 一人麻雀における最適な打牌戦略を計算するためのアルゴリズムを提供します。
//! 
//! 主な機能：
//! - 向聴数計算（通常手、七対子、国士無双を含む）
//! - ツモ確率の計算
//! - 期待値計算
//! - リーチ/ダマの選択
//! - 裏ドラ確率の動的計算
//!
//! Rust port of nekobean's C++ implementation of his single-player mahjong
//! calculator. Some of the original comments are included.
//!
//! Source: <https://github.com/nekobean/mahjong-cpp>
//!
//! Major differences compared to the C++ version:
//! - Whenever shanten calculation is involved, all types of shanten will be
//!   considered (using `shanten::calc_all`). In the original version, you can
//!   only choose one of normal, chitoi and kokushi.
//! - When calculating uradora probs, the actual the number of tiles left is
//!   calculated and used, while the original version uses a fixed value of 121.
//! - Riichi is optional, so you can calculate values of a dama-preferred hand,
//!   although dama has no benefit at all in single-player mahjong.
//! - `max_tsumo` is set to the actual value, instead of the hardcoded 17 or 18
//!   in the original version. Not only does this reduce the amount of
//!   calculations, but more importantly, I think this is the theoretically
//!   correct way to calculate, since we keep track of the actual `tiles_seen`
//!   on board so we can have the accurate denominator when building the
//!   `tsumo_prob_table`.
//!
//! Other improvements:
//! - More aggressive compile-time optimizations.
//!
//! To reproduce the behavior of the original C++ version, set feature
//! `sp_reproduce_cpp_ver`.

// サブモジュールの宣言
// calc: 期待値計算のコアロジック
mod calc;
// candidate: 打牌候補とその評価結果を表現
mod candidate;
// state: 初期状態や計算状態の管理
mod state;
// tile: 必要牌の定義と操作
mod tile;

// 外部APIとして公開する型のre-export
// SPCalculator: メインの計算機クラス
pub use calc::SPCalculator;
// Candidate: 打牌候補、CandidateColumn: 結果表示用の列定義
pub use candidate::{Candidate, CandidateColumn};
// InitState: 計算開始時の初期状態
pub use state::InitState;
// RequiredTile: アガリに必要な牌の情報
pub use tile::RequiredTile;

// 残りツモ数の最大値定義
// C++版再現モードでは18（元のC++実装に合わせる）
#[cfg(feature = "sp_reproduce_cpp_ver")]
pub const MAX_TSUMOS_LEFT: usize = 18;
/// 実際のゲームでは、親の第一ツモは必須なので最大17回が正しい
/// In practice, the max number of tsumos left should be 17, since the first
/// tsumo of oya is mandatory.
#[cfg(not(feature = "sp_reproduce_cpp_ver"))]
pub const MAX_TSUMOS_LEFT: usize = 17;

// 向聴数計算関数の選択
// C++版再現モードでは通常手のみを計算（元のC++実装の動作）
#[cfg(feature = "sp_reproduce_cpp_ver")]
const CALC_SHANTEN_FN: fn(&[u8; 34], u8) -> i8 = super::shanten::calc_normal;
// 通常モードでは全ての手牌形（通常手、七対子、国士無双）を考慮
#[cfg(not(feature = "sp_reproduce_cpp_ver"))]
const CALC_SHANTEN_FN: fn(&[u8; 34], u8) -> i8 = super::shanten::calc_all;
