//! 麻雀の基本アルゴリズム群を提供するモジュール
//!
//! このモジュールには麻雀プログラムで必要となる以下の主要なアルゴリズムが含まれています：
//! - agari: 和了判定と役・符・飜数の計算
//! - point: 点数計算
//! - shanten: 向聴数計算
//! - sp: シングルプレイヤー向けの計算（立直判断、打牌選択など）
//!
//! This module includes essential mahjong algorithms including agari, shanten,
//! single-player calculators and score lookups.

// 和了判定と役計算モジュール
pub mod agari;
// 点数計算モジュール
pub mod point;
// 向聴数計算モジュール
pub mod shanten;
// シングルプレイヤー向け計算モジュール
pub mod sp;
