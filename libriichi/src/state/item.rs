// このファイルは麻雀の河（捨て牌）に関連するデータ構造を定義しています。
// 河には通常の捨て牌（Sutehai）だけでなく、チー・ポン・カンの情報も含まれます。
// KawaItemは河の1つのアイテムを表し、捨て牌とそれに関連する鳴き情報を保持します。
// これらの構造体は、ゲーム状態の一部として河の状態を管理するために使用されます。

// タイル（牌）を表す型をインポート
use crate::tile::Tile;
// フォーマット表示用のトレイトをインポート
use std::fmt;

// シリアライズ機能を提供するマクロをインポート
use serde::Serialize;
// 固定サイズの小さな配列を効率的に扱うための型をインポート
use tinyvec::ArrayVec;

// 河の1つのアイテムを表す構造体
// Debug: デバッグ出力を可能にする
// Clone: 構造体のコピーを可能にする
// Serialize: JSONなどへのシリアライズを可能にする
#[derive(Debug, Clone, Serialize)]
pub(super) struct KawaItem {
    // チー・ポンの情報（ある場合）
    // Optionで包まれているため、チー・ポンがない通常の捨て牌の場合はNone
    pub(super) chi_pon: Option<ChiPon>,
    // カンされた牌の配列
    // ArrayVecを使用して最大4枚まで（大明槓の場合）の牌を効率的に格納
    pub(super) kan: ArrayVec<[Tile; 4]>,
    // 捨て牌の情報（必須）
    pub(super) sutehai: Sutehai,
}

// 捨て牌を表す構造体
// Debug: デバッグ出力を可能にする
// Clone: 構造体のコピーを可能にする
// Copy: 値のコピーセマンティクスを有効にする（軽量な構造体のため）
// Serialize: JSONなどへのシリアライズを可能にする
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct Sutehai {
    // 捨てられた牌
    pub(super) tile: Tile,
    // ドラ表示牌かどうか（通常のドラのみ、赤ドラは含まない）
    // only for normal dora, aka is not included
    pub(super) is_dora: bool,
    // 手出しかどうか（true: 手出し、false: ツモ切り）
    pub(super) is_tedashi: bool,
    // リーチ宣言牌かどうか
    pub(super) is_riichi: bool,
}

// チー・ポンの情報を表す構造体
// Debug: デバッグ出力を可能にする
// Clone: 構造体のコピーを可能にする
// Serialize: JSONなどへのシリアライズを可能にする
#[derive(Debug, Clone, Serialize)]
pub(super) struct ChiPon {
    // 手牌から晒した2枚の牌
    pub(super) consumed: [Tile; 2],
    // チー・ポンの対象となった牌（他家が捨てた牌）
    pub(super) target_tile: Tile,
}

// Sutehaiの表示形式を定義
impl fmt::Display for Sutehai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 捨て牌を文字列として出力
        write!(
            f,
            "{}{}{}{}",
            // 牌そのもの
            self.tile,
            // ドラ表示牌の場合は"!"を付ける
            if self.is_dora { "!" } else { "" },
            // ツモ切りの場合は"^"を付ける（手出しの場合は何も付けない）
            if self.is_tedashi { "" } else { "^" },
            // リーチ宣言牌の場合は"|"を付ける
            if self.is_riichi { "|" } else { "" },
        )
    }
}

// ChiPonの表示形式を定義
impl fmt::Display for ChiPon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // チー・ポンを"(手牌1手牌2+鳴いた牌)"の形式で出力
        write!(
            f,
            "({}{}+{})",
            // 手牌から晒した1枚目
            self.consumed[0], 
            // 手牌から晒した2枚目
            self.consumed[1], 
            // 他家から鳴いた牌
            self.target_tile,
        )
    }
}

// KawaItemの表示形式を定義
impl fmt::Display for KawaItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // カンがある場合の処理
        if !self.kan.is_empty() {
            // カンの開始を"{"で表す
            f.write_str("{")?;
            // 各カン牌を順に出力
            for kan in self.kan {
                write!(f, "{kan}")?;
            }
            // カンの終了を"}"で表す
            f.write_str("}")?;
        }

        // チー・ポンがある場合は出力
        if let Some(chi_pon) = &self.chi_pon {
            write!(f, "{chi_pon}")?;
        }

        // 最後に捨て牌を出力（必須要素）
        write!(f, "{}", self.sutehai)
    }
}
