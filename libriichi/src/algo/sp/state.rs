//! 手牌と山牌の状態管理
//!
//! このモジュールは、シングルプレイヤー麻雀における
//! ゲーム状態（手牌、山牌、赤牌の位置など）を管理する構造体を提供します。
//! 主に以下の機能を実装：
//! - 手牌の操作（ツモ・打牌）
//! - 山牌の管理
//! - 有効牌の計算
//! - 向聴数に基づく牌の評価

// 向聴数計算関数をインポート
use super::CALC_SHANTEN_FN;
// 牌に関する型定義をインポート
use super::tile::{DiscardTile, DrawTile, RequiredTile};
// 基本的な牌の型をインポート
use crate::tile::Tile;
// 牌生成用のマクロをインポート
use crate::{must_tile, t, tu8};

// 固定サイズの動的配列（スタック割り当て）
use tinyvec::ArrayVec;

/// ゲーム中の可変状態を管理する構造体
/// 
/// 手牌と山牌の両方の状態を保持し、計算中に変更される
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct State {
    // 手牌関連
    /// 手牌の枚数配列（34種類の牌それぞれの枚数）
    pub(super) tehai: [u8; 34],
    /// 手牌中の赤牌の有無（[赤5m, 赤5p, 赤5s]）
    pub(super) akas_in_hand: [bool; 3],

    // 山牌関連
    /// 山に残っている牌の枚数配列（34種類）
    pub(super) tiles_in_wall: [u8; 34],
    /// 山に残っている赤牌の有無（[赤5m, 赤5p, 赤5s]）
    pub(super) akas_in_wall: [bool; 3],
    /// 追加のツモ回数（カンドラなど特殊ケース用）
    pub(super) n_extra_tsumo: u8,
}

/// 計算開始時の初期状態を表す構造体
/// 
/// この構造体は外部APIとして公開され、
/// ユーザーが計算を開始する際の入力として使用される
#[derive(Clone)]
pub struct InitState {
    // 手牌の初期状態
    /// 手牌の枚数配列（34種類の牌それぞれの枚数）
    pub tehai: [u8; 34],
    /// 手牌中の赤牌の有無（[赤5m, 赤5p, 赤5s]）
    pub akas_in_hand: [bool; 3],

    // 場に見えている牌の情報
    /// 場に見えている牌の枚数（捨て牌、副露牌など）
    pub tiles_seen: [u8; 34],
    /// 場に見えている赤牌の有無
    pub akas_seen: [bool; 3],
}

/// InitStateからStateへの変換実装
/// 
/// 見えている牌の情報から、山に残っている牌を計算する
impl From<InitState> for State {
    fn from(
        InitState {
            tehai,
            akas_in_hand,
            tiles_seen,
            akas_seen,
        }: InitState,
    ) -> Self {
        // 見えている牌から山の残り枚数を計算
        let mut tiles_in_wall = tiles_seen;
        let mut akas_in_wall = akas_seen;
        // 各牌は全部で4枚なので、4から見えている枚数を引く
        tiles_in_wall.iter_mut().for_each(|v| *v = 4 - *v);
        // 赤牌の有無を反転（見えている→山に残っていない）
        akas_in_wall.iter_mut().for_each(|v| *v = !*v);
        Self {
            tehai,
            akas_in_hand,
            tiles_in_wall,
            akas_in_wall,
            n_extra_tsumo: 0, // 初期状態では追加ツモなし
        }
    }
}

impl State {
    /// 手牌から牌を切る（打牌）
    /// 
    /// 指定された牌を手牌から1枚減らし、
    /// 赤牌の場合は赤牌フラグも更新する
    pub(super) const fn discard(&mut self, tile: Tile) {
        // 赤牌を通常の5に変換してインデックスを取得し、枚数を減らす
        self.tehai[tile.deaka().as_usize()] -= 1;
        // 赤牌の場合は対応するフラグをfalseに
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_hand[0] = false, // 赤5萬
            tu8!(5pr) => self.akas_in_hand[1] = false, // 赤5筒
            tu8!(5sr) => self.akas_in_hand[2] = false, // 赤5索
            _ => (), // 通常牌の場合は何もしない
        }
    }

    /// 打牌操作を取り消す（手牌に戻す）
    /// 
    /// discardの逆操作で、牌を手牌に戻す
    pub(super) const fn undo_discard(&mut self, tile: Tile) {
        // 手牌の枚数を増やす
        self.tehai[tile.deaka().as_usize()] += 1;
        // 赤牌の場合は対応するフラグをtrueに
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_hand[0] = true, // 赤5萬
            tu8!(5pr) => self.akas_in_hand[1] = true, // 赤5筒
            tu8!(5sr) => self.akas_in_hand[2] = true, // 赤5索
            _ => (), // 通常牌の場合は何もしない
        }
    }

    /// 山から牌をツモる（配牌）
    /// 
    /// 山から指定された牌を1枚減らし、手牌に加える
    pub(super) const fn deal(&mut self, tile: Tile) {
        // 山から牌を1枚減らす
        self.tiles_in_wall[tile.deaka().as_usize()] -= 1;
        // 赤牌の場合は山の赤牌フラグをfalseに
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_wall[0] = false, // 赤5萬
            tu8!(5pr) => self.akas_in_wall[1] = false, // 赤5筒
            tu8!(5sr) => self.akas_in_wall[2] = false, // 赤5索
            _ => (), // 通常牌の場合は何もしない
        }
        // 手牌に牌を加える（undo_discardを利用）
        self.undo_discard(tile);
    }

    /// ツモ操作を取り消す
    /// 
    /// dealの逆操作で、手牌から牌を取り除き山に戻す
    pub(super) const fn undo_deal(&mut self, tile: Tile) {
        // 手牌から牌を取り除く（discardを利用）
        self.discard(tile);
        // 山に牌を戻す
        self.tiles_in_wall[tile.deaka().as_usize()] += 1;
        // 赤牌の場合は山の赤牌フラグをtrueに
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_wall[0] = true, // 赤5萬
            tu8!(5pr) => self.akas_in_wall[1] = true, // 赤5筒
            tu8!(5sr) => self.akas_in_wall[2] = true, // 赤5索
            _ => (), // 通常牌の場合は何もしない
        }
    }

    /// 打牌候補のリストを取得
    /// 
    /// 現在の手牌から打てる牌とその評価（向聴数の変化）を計算
    /// 
    /// # Arguments
    /// * `shanten` - 現在の向聴数
    /// * `tehai_len_div3` - 手牌の長さ÷3（通常は4）
    /// 
    /// # Returns
    /// 打牌候補のリスト（最大14種類）
    pub(super) fn get_discard_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        // 手牌のコピーを作成（計算用）
        let mut tehai = self.tehai;
        // 全34種類の牌について調べる
        for tid in 0..34 {
            // その牌を持っていない場合はスキップ
            if tehai[tid] == 0 {
                continue;
            }

            // その牌を1枚切った場合の向聴数を計算
            tehai[tid] -= 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] += 1; // 元に戻す

            // 向聴数の変化量を計算
            let shanten_diff = shanten_after - shanten;

            // 赤牌の判定：5の牌で赤牌を持っていて、かつ5が1枚しかない場合
            let tile = match tid as u8 {
                tu8!(5m) if self.akas_in_hand[0] && tehai[tid] == 1 => t!(5mr), // 赤5萬
                tu8!(5p) if self.akas_in_hand[1] && tehai[tid] == 1 => t!(5pr), // 赤5筒
                tu8!(5s) if self.akas_in_hand[2] && tehai[tid] == 1 => t!(5sr), // 赤5索
                _ => must_tile!(tid), // 通常牌
            };

            // 打牌候補として追加
            discard_tiles.push(DiscardTile { tile, shanten_diff });
        }

        discard_tiles
    }

    /// ツモ可能な牌のリストを取得
    /// 
    /// 山に残っている牌とその評価（向聴数の変化）を計算
    /// 
    /// # Arguments
    /// * `shanten` - 現在の向聴数
    /// * `tehai_len_div3` - 手牌の長さ÷3（通常は4）
    /// 
    /// # Returns
    /// ツモ可能な牌のリスト（最大37種類：34種類の通常牌+3種類の赤牌）
    pub(super) fn get_draw_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DrawTile; 37]> {
        let mut draw_tiles = ArrayVec::default();

        // 手牌のコピーを作成（計算用）
        let mut tehai = self.tehai;
        // 山に残っている全ての牌について調べる
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            // 山に残っていない牌はスキップ
            if count == 0 {
                continue;
            }

            // その牌をツモった場合の向聴数を計算
            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] -= 1; // 元に戻す

            // 向聴数の変化量を計算
            let shanten_diff = shanten_after - shanten;

            let tile = must_tile!(tid);
            // 赤牌の処理：5の牌で山に赤牌が残っている場合
            match (tid as u8, self.akas_in_wall) {
                (tu8!(5m), [true, _, _]) | (tu8!(5p), [_, true, _]) | (tu8!(5s), [_, _, true]) => {
                    // 通常の5が2枚以上ある場合は、通常の5も追加
                    if count >= 2 {
                        draw_tiles.push(DrawTile {
                            tile,
                            count: count - 1, // 赤牌1枚を除いた枚数
                            shanten_diff,
                        });
                    }
                    // 赤牌を追加（必ず1枚）
                    draw_tiles.push(DrawTile {
                        tile: tile.akaize(), // 赤牌に変換
                        count: 1,
                        shanten_diff,
                    });
                }
                // 通常牌の場合
                _ => draw_tiles.push(DrawTile {
                    tile,
                    count,
                    shanten_diff,
                }),
            }
        }

        draw_tiles
    }

    /// 有効牌（必要牌）のリストを取得
    /// 
    /// ツモると向聴数が減少する牌（有効牌）のリストを返す
    /// 
    /// # Arguments
    /// * `tehai_len_div3` - 手牌の長さ÷3（通常は4）
    /// 
    /// # Returns
    /// 有効牌のリスト（最大34種類）
    pub(super) fn get_required_tiles(&self, tehai_len_div3: u8) -> ArrayVec<[RequiredTile; 34]> {
        // 手牌のコピーを作成（計算用）
        let mut tehai = self.tehai;

        // 現在の向聴数を計算
        let shanten = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
        let mut required_tiles = ArrayVec::default();

        // 山に残っている全ての牌について調べる
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            // 山に残っていない牌はスキップ
            if count == 0 {
                continue;
            }

            // その牌をツモった場合の向聴数を計算
            tehai[tid] += 1;
            let shanten_after = CALC_SHANTEN_FN(&tehai, tehai_len_div3);
            tehai[tid] -= 1; // 元に戻す

            // 向聴数が減少する（改善する）牌は有効牌
            if shanten_after < shanten {
                required_tiles.push(RequiredTile {
                    tile: must_tile!(tid),
                    count, // 山に残っている枚数
                });
            }
        }

        required_tiles
    }

    /// 山に残っている牌の総枚数を計算
    /// 
    /// # Returns
    /// 山に残っている全ての牌の合計枚数
    pub(super) fn sum_left_tiles(&self) -> u8 {
        self.tiles_in_wall.iter().sum()
    }
}
