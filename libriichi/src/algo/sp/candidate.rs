//! 打牌候補の評価結果を表現するモジュール
//!
//! このモジュールは、シングルプレイヤー麻雀における
//! 打牌候補の評価結果（期待値、勝率、聴牌率など）を管理する構造体を提供します。
//! 主な機能：
//! - 打牌候補の評価データの保持
//! - 候補間の比較機能
//! - CSV形式での出力機能

// 最大ツモ回数定数
use super::MAX_TSUMOS_LEFT;
// 必要牌の型
use super::tile::RequiredTile;
// 基本的な牌の型
use crate::tile::Tile;
// 比較演算用
use std::cmp::Ordering;

// 固定サイズの動的配列
use tinyvec::ArrayVec;

/// 打牌候補とその評価結果を表す構造体
/// 
/// この構造体は外部APIとして公開され、
/// 各打牌候補の詳細な評価データを提供する
#[derive(Debug)]
pub struct Candidate {
    /// 打牌候補の牌
    pub tile: Tile,
    /// 各巡目における聴牌確率（0.0〜1.0）
    /// インデックス0が次巡、1が2巡後...を表す
    pub tenpai_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 各巡目における和了確率（0.0〜1.0）
    pub win_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 各巡目における期待値（点数）
    pub exp_values: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 有効牌（この牌を切った後に必要な牌）のリスト
    pub required_tiles: ArrayVec<[RequiredTile; 34]>,
    /// 有効牌の総枚数
    pub num_required_tiles: u8,
    /// 向聴戻し（向聴数が増加する打牌）かどうか
    pub shanten_down: bool,
}

/// 内部計算用の候補データ構造体
/// 
/// スライス参照を使用してメモリ効率を向上させた内部用構造体
#[derive(Default)]
pub(super) struct RawCandidate<'a> {
    /// 打牌候補の牌
    pub(super) tile: Tile,
    /// 聴牌確率の配列への参照
    pub(super) tenpai_probs: &'a [f32],
    /// 和了確率の配列への参照
    pub(super) win_probs: &'a [f32],
    /// 期待値の配列への参照
    pub(super) exp_values: &'a [f32],
    /// 有効牌のリスト（所有）
    pub(super) required_tiles: ArrayVec<[RequiredTile; 34]>,
    /// 向聴戻しかどうか
    pub(super) shanten_down: bool,
}

/// 打牌候補の比較基準を表す列挙型
/// 
/// 候補をソートする際の優先順位を指定
#[derive(Clone, Copy)]
pub enum CandidateColumn {
    /// 期待値（最も重要な指標）
    EV,
    /// 和了確率
    WinProb,
    /// 聴牌確率
    TenpaiProb,
    /// 向聴戻しでないこと（向聴数維持・改善）
    NotShantenDown,
    /// 有効牌の総枚数
    NumRequiredTiles,
    /// 牌の優先度（字牌・端牌を優先的に切る）
    DiscardPriority,
}

/// RawCandidateからCandidateへの変換実装
/// 
/// 内部計算用データから公開用データへ変換し、
/// 値の範囲を適切にクランプする
impl From<RawCandidate<'_>> for Candidate {
    fn from(
        RawCandidate {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            shanten_down,
        }: RawCandidate<'_>,
    ) -> Self {
        // 有効牌の総枚数を計算
        let num_required_tiles = required_tiles.iter().map(|r| r.count).sum();
        // 確率値を0.0〜1.0の範囲にクランプ
        let tenpai_probs = tenpai_probs.iter().map(|p| p.clamp(0., 1.)).collect();
        let win_probs = win_probs.iter().map(|p| p.clamp(0., 1.)).collect();
        // 期待値を0以上にクランプ（負の値は0に）
        let exp_values = exp_values.iter().map(|v| v.max(0.)).collect();

        Self {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            num_required_tiles,
            shanten_down,
        }
    }
}

impl Candidate {
    /// 指定された基準で他の候補と比較
    /// 
    /// 再帰的に次の優先順位で比較を行う：
    /// EV → 和了率 → 聴牌率 → 向聴維持 → 有効牌枚数 → 牌の優先度
    pub fn cmp(&self, other: &Self, by: CandidateColumn) -> Ordering {
        // 同じ牌なら同値
        if self.tile == other.tile {
            return Ordering::Equal;
        }
        match by {
            // 期待値で比較（次巡の期待値を使用）
            CandidateColumn::EV => match self.exp_values[0].total_cmp(&other.exp_values[0]) {
                Ordering::Equal => self.cmp(other, CandidateColumn::WinProb), // 同値なら和了率で比較
                o => o,
            },
            // 和了確率で比較
            CandidateColumn::WinProb => match self.win_probs[0].total_cmp(&other.win_probs[0]) {
                Ordering::Equal => self.cmp(other, CandidateColumn::TenpaiProb), // 同値なら聴牌率で比較
                o => o,
            },
            // 聴牌確率で比較
            CandidateColumn::TenpaiProb => {
                match self.tenpai_probs[0].total_cmp(&other.tenpai_probs[0]) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::NotShantenDown), // 同値なら向聴維持で比較
                    o => o,
                }
            }
            // 向聴戻しでないことを優先（false > true）
            CandidateColumn::NotShantenDown => match (self.shanten_down, other.shanten_down) {
                (false, true) => Ordering::Greater, // selfが向聴維持、otherが向聴戻し
                (true, false) => Ordering::Less,    // selfが向聴戻し、otherが向聴維持
                _ => self.cmp(other, CandidateColumn::NumRequiredTiles), // 同じなら有効牌枚数で比較
            },
            // 有効牌の総枚数で比較（多い方が良い）
            CandidateColumn::NumRequiredTiles => {
                match self.num_required_tiles.cmp(&other.num_required_tiles) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::DiscardPriority), // 同値なら牌の優先度で比較
                    o => o,
                }
            }
            // 牌の優先度で比較（字牌・端牌を優先的に切る）
            CandidateColumn::DiscardPriority => self.tile.cmp_discard_priority(other.tile),
        }
    }

    /// CSV形式のヘッダー行を生成
    /// 
    /// # Arguments
    /// * `can_discard` - 打牌可能な状態かどうか（テンパイ時はfalse）
    pub const fn csv_header(can_discard: bool) -> &'static [&'static str] {
        if can_discard {
            // 打牌可能な場合のヘッダー
            &[
                "Tile",          // 打牌候補
                "EV",            // 期待値
                "Win prob",      // 和了確率(%)
                "Tenpai prob",   // 聴牌確率(%)
                "Shanten down?", // 向聴戻しか
                "Kinds",         // 有効牌の種類数
                "Sum",           // 有効牌の総枚数
                "Required tiles", // 有効牌リスト
            ]
        } else {
            // テンパイ時のヘッダー（打牌列なし）
            &[
                "EV",
                "Win prob",
                "Tenpai prob",
                "Kinds",
                "Sum",
                "Required tiles",
            ]
        }
    }

    /// CSV形式のデータ行を生成
    /// 
    /// # Arguments
    /// * `can_discard` - 打牌可能な状態かどうか
    pub fn csv_row(&self, can_discard: bool) -> Vec<String> {
        // 有効牌リストを "牌@枚数,牌@枚数..." の形式で文字列化
        let required_tiles = self
            .required_tiles
            .iter()
            .map(|r| format!("{}@{}", r.tile, r.count))
            .collect::<Vec<_>>()
            .join(",");
        
        if can_discard {
            // 打牌可能な場合のデータ行
            vec![
                self.tile.to_string(),                                        // 打牌候補
                format!("{:.03}", self.exp_values[0]),                      // 期待値（小数点以下3桁）
                format!("{:.03}", self.win_probs[0] * 100.),               // 和了確率を%表示
                format!("{:.03}", self.tenpai_probs[0] * 100.),            // 聴牌確率を%表示
                if self.shanten_down { "Yes" } else { "No" }.to_owned(),    // 向聴戻しか
                self.required_tiles.len().to_string(),                       // 有効牌の種類数
                self.num_required_tiles.to_string(),                         // 有効牌の総枚数
                required_tiles,                                              // 有効牌リスト
            ]
        } else {
            // テンパイ時のデータ行（打牌列なし）
            vec![
                format!("{:.03}", self.exp_values[0]),
                format!("{:.03}", self.win_probs[0] * 100.),
                format!("{:.03}", self.tenpai_probs[0] * 100.),
                self.required_tiles.len().to_string(),
                self.num_required_tiles.to_string(),
                required_tiles,
            ]
        }
    }

    /// C++版の動作を再現するための補正処理
    /// 
    /// C++版との互換性のための処理で、本来は不要
    #[cfg(feature = "sp_reproduce_cpp_ver")]
    pub(super) fn calibrate(mut self, real_max_tsumo: usize) -> Self {
        if self.shanten_down {
            // 向聴戻しの場合、C++版の挙動に合わせて1巡ずらす
            // （向聴戻しをしない場合の確率が過小評価される問題への対処）
            // 最初の要素を0にして左に回転
            self.tenpai_probs[0] = 0.;
            self.tenpai_probs.rotate_left(1);
            self.win_probs[0] = 0.;
            self.win_probs.rotate_left(1);
            self.exp_values[0] = 0.;
            self.exp_values.rotate_left(1);
        }
        // 実際の最大ツモ数に合わせて配列を調整
        // 右に回転させて先頭を適切な位置に
        self.tenpai_probs.rotate_right(real_max_tsumo);
        self.tenpai_probs.truncate(real_max_tsumo);
        self.win_probs.rotate_right(real_max_tsumo);
        self.win_probs.truncate(real_max_tsumo);
        self.exp_values.rotate_right(real_max_tsumo);
        self.exp_values.truncate(real_max_tsumo);
        self
    }
}
