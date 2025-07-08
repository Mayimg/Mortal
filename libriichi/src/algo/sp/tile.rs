//! 打牌・ツモ牌・必要牌に関する型定義
//!
//! このモジュールは、シングルプレイヤー麻雀計算において使用される
//! 牌に関する構造体を定義します。主に以下の3つの構造体を提供：
//! - DiscardTile: 打牌候補とその評価
//! - DrawTile: ツモ可能な牌とその情報
//! - RequiredTile: アガリに必要な牌の情報

// 基本的な牌の型をインポート
use crate::tile::Tile;

/// 打牌候補を表す構造体
/// 
/// どの牌を切るかの候補と、その打牌による向聴数の変化を保持
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DiscardTile {
    /// 打牌候補の牌
    pub(super) tile: Tile,
    /// この牌を切った時の向聴数の変化量
    /// 正の値: 向聴数が増加（悪化）
    /// 0: 向聴数不変
    /// 負の値: 向聴数が減少（改善）
    pub(super) shanten_diff: i8,
}

/// ツモ可能な牌を表す構造体
/// 
/// ツモることができる牌とその枚数、ツモった場合の向聴数変化を保持
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DrawTile {
    /// ツモ可能な牌の種類
    pub(super) tile: Tile,
    /// その牌の残り枚数（山に残っている枚数）
    pub(super) count: u8,
    /// この牌をツモった時の向聴数の変化量
    /// 正の値: 向聴数が増加（悪化）
    /// 0: 向聴数不変
    /// 負の値: 向聴数が減少（改善）
    pub(super) shanten_diff: i8,
}

/// アガリに必要な牌を表す構造体
/// 
/// この構造体は外部APIとして公開され、
/// どの牌が何枚必要かという情報を提供する
#[derive(Debug, Default, Clone, Copy)]
pub struct RequiredTile {
    /// 必要な牌の種類
    pub tile: Tile,
    /// 必要な枚数（通常は1枚、刻子待ちなら2枚など）
    pub count: u8,
}
