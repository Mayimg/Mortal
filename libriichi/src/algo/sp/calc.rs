//! シングルプレイヤー麻雀の期待値計算エンジン
//!
//! このモジュールは、一人麻雀における打牌選択の期待値計算を行う
//! コアロジックを実装しています。主な機能：
//! - 各打牌候補の期待値・勝率・聴牌率の計算
//! - 動的計画法による効率的な計算
//! - 向聴戻しや手変わりの考慮
//! - キャッシュによる高速化

// 内部モジュールのインポート
use super::candidate::RawCandidate;
use super::state::{InitState, State};
use super::tile::{DiscardTile, DrawTile};
use super::{Candidate, CandidateColumn, MAX_TSUMOS_LEFT};
// アガリ計算用
use crate::algo::agari::{Agari, AgariCalculator};
// 牌の型
use crate::tile::Tile;
// 牌生成マクロ
use crate::{must_tile, t, tu8};
// 参照カウント（メモリ効率化）
use std::rc::Rc;

// 高速ハッシュマップ
use ahash::AHashMap;
// エラー処理
use anyhow::{Result, ensure};

// 向聴数の閾値（3向聴まで詳細計算、4向聴以上は簡易計算）
const SHANTEN_THRES: i8 = 3;
// 最大残り牌数（全136枚 - 配牌14枚 - 王牌1枚）
const MAX_TILES_LEFT: usize = 34 * 4 - 1 - 13;

/// 裏ドラの乗る確率のテーブル
/// [ドラ表示牌の枚数][手牌の枚数] = 確率
const URADORA_PROB_TABLE: [[f32; 13]; 5] = include!("../data/uradora_prob_table.txt");

// 状態キャッシュの型定義
// 向聴数ごとに計算結果をキャッシュする
type StateCache<const MAX_TSUMO: usize> =
    [AHashMap<State, Rc<Values<MAX_TSUMO>>>; SHANTEN_THRES as usize + 1];

/// 各巡目の確率と期待値を保持する構造体
struct Values<const MAX_TSUMO: usize> {
    /// 各巡目の聴牌確率
    tenpai_probs: [f32; MAX_TSUMO],
    /// 各巡目の和了確率
    win_probs: [f32; MAX_TSUMO],
    /// 各巡目の期待値
    exp_values: [f32; MAX_TSUMO],
}

/// 計算結果を表す列挙型
enum ScoresOrValues<const MAX_TSUMO: usize> {
    /// テンパイかつ役ありの場合：[ツモ和了, ロン和了, リーチツモ, リーチロン]の点数
    Scores([f32; 4]),
    /// 向聴数が1以上の場合：確率と期待値
    Values(Rc<Values<MAX_TSUMO>>),
}

/// シングルプレイヤー麻雀計算機の本体
/// 
/// ゲーム状態と計算設定を保持し、期待値計算を実行する
#[derive(Debug)]
pub struct SPCalculator<'a> {
    // アガリ計算で使用する不変の状態
    /// 手牌の長さ÷3（通常は4、カンした場合は3）
    pub tehai_len_div3: u8,
    /// チーした牌の種類
    pub chis: &'a [u8],
    /// ポンした牌の種類
    pub pons: &'a [u8],
    /// 明カンした牌の種類
    pub minkans: &'a [u8],
    /// 暗カンした牌の種類
    pub ankans: &'a [u8],
    /// 場風（東=0, 南=1, 西=2, 北=3）
    pub bakaze: u8,
    /// 自風（東=0, 南=1, 西=2, 北=3）
    pub jikaze: u8,
    /// 門前かどうか
    pub is_menzen: bool,

    /// 副露（暗カン含む）に含まれるドラの枚数
    pub num_doras_in_fuuro: u8,
    /// ドラ表示牌のリスト
    pub dora_indicators: &'a [Tile],
    /// ダブルリーチを計算するか
    pub calc_double_riichi: bool,
    /// 海底・河底を計算するか
    pub calc_haitei: bool,
    /// リーチを優先するか（ダマテンより）
    pub prefer_riichi: bool,
    /// 結果をソートするか
    pub sort_result: bool,

    /// 和了確率を最大化する（期待値より優先）
    pub maximize_win_prob: bool,
    /// 手変わりを考慮する
    pub calc_tegawari: bool,
    /// 向聴落としを考慮する
    pub calc_shanten_down: bool,
}

/// 計算中の内部状態を管理する構造体
/// 
/// const genericsを使用してコンパイル時最適化を実現
struct SPCalculatorState<'a, const MAX_TSUMO: usize> {
    /// 計算機本体への参照
    sup: &'a SPCalculator<'a>,
    /// 現在のゲーム状態
    state: State,

    /// ツモ確率テーブル（有効牌枚数ごと）
    tsumo_prob_table: &'a [[f32; MAX_TSUMO]; 4],
    /// 非ツモ確率テーブル（有効牌総数ごと）
    not_tsumo_prob_table: &'a [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1],

    /// 打牌後の状態キャッシュ（向聴数ごと）
    discard_cache: StateCache<MAX_TSUMO>,
    /// ツモ後の状態キャッシュ（向聴数ごと）
    draw_cache: StateCache<MAX_TSUMO>,

    /// C++版互換モード用：実際の最大ツモ数
    #[cfg(feature = "sp_reproduce_cpp_ver")]
    real_max_tsumo: usize,
}

impl SPCalculator<'_> {
    /// 期待値計算のメインメソッド
    /// 
    /// 現在の手牌状態から、各打牌候補の期待値・勝率・聴牌率を計算する
    /// 
    /// # Arguments
    /// - `init_state`: 初期状態（手牌と見えている牌）
    /// - `can_discard`: 打牌可能か（手牌が3n+2枚ならtrue）
    /// - `tsumos_left`: 残りツモ回数（1〜17の範囲）
    /// - `cur_shanten`: 現在の向聴数（0以上）
    ///
    /// # Returns
    /// 打牌候補のリスト（ソート済み、index 0が最良の選択）
    pub fn calc(
        &self,
        init_state: InitState,
        can_discard: bool,
        tsumos_left: u8,
        cur_shanten: i8,
    ) -> Result<Vec<Candidate>> {
        // 入力値の検証
        ensure!(cur_shanten >= 0, "can't calculate an agari hand");
        ensure!(tsumos_left >= 1, "need at least one more tsumo");
        ensure!(tsumos_left <= MAX_TSUMOS_LEFT as u8);

        // 最大ツモ数の決定
        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let max_tsumo = if can_discard { 17 } else { 18 }; // C++版互換
        #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
        let max_tsumo = tsumos_left as usize; // 実際の残りツモ数を使用

        // 初期状態から計算用状態への変換
        let state = State::from(init_state);
        let n_left_tiles = state.sum_left_tiles() as usize;

        // const genericsを使用したコンパイル時最適化
        // バイナリサイズは増加するが、分岐削減・バッファ削減・
        // ループアンロール・ベクトル化などの最適化が期待できる
        macro_rules! static_expand {
            ($($n:literal),*) => {
                match max_tsumo {
                    $($n => {
                        // 確率テーブルの構築
                        let tsumo_prob_table = build_tsumo_prob_table(n_left_tiles);
                        let not_tsumo_prob_table = build_not_tsumo_prob_table(n_left_tiles);
                        // 計算状態の初期化
                        let mut calc_state = SPCalculatorState::<$n> {
                            sup: self,
                            state,
                            tsumo_prob_table: &tsumo_prob_table,
                            not_tsumo_prob_table: &not_tsumo_prob_table,
                            discard_cache: Default::default(),
                            draw_cache: Default::default(),
                            #[cfg(feature = "sp_reproduce_cpp_ver")]
                            real_max_tsumo: tsumos_left as usize,
                        };
                        // 計算実行
                        calc_state.calc(can_discard, cur_shanten)
                    },)*
                    _ => unreachable!(),
                }
            }
        }
        // マクロを使用して各ツモ数に対応した関数を生成
        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let candidates = static_expand!(17, 18); // C++版は17,18のみ
        #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
        let candidates = static_expand!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17); // 全パターン
        Ok(candidates)
    }
}

/// ツモ確率テーブルを構築
/// 
/// 有効牌の枚数ごとに、各巡目で有効牌を引ける確率を事前計算
/// 
/// # Returns
/// table[i][j] = 有効牌が(i+1)枚の場合にj巡目に有効牌を引ける確率
fn build_tsumo_prob_table<const MAX_TSUMO: usize>(n_left_tiles: usize) -> [[f32; MAX_TSUMO]; 4] {
    let mut table = [[0.; MAX_TSUMO]; 4];
    // 有効牌の枚数ごとに確率を計算
    for (i, row) in table.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            // j巡目での確率 = 有効牌数 / (残り牌総数 - すでに引いた枚数)
            *v = (i + 1) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

/// 非ツモ確率テーブルを構築
/// 
/// 有効牌の総枚数ごとに、指定巡目までに有効牌を引けない確率を事前計算
/// 
/// # Returns
/// table[i][j] = 有効牌がi枚の場合に(j-1)巡目までに有効牌を引けなかった確率
fn build_not_tsumo_prob_table<const MAX_TSUMO: usize>(
    n_left_tiles: usize,
) -> [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1] {
    let mut table = [[0.; MAX_TSUMO]; MAX_TILES_LEFT + 1];
    
    // 元のC++版では`n_left_tiles`行のみだったが、
    // 9999m6677p88s335zのような全種類の牌で改善する手牌で
    // オーバーフローする可能性があるため、+1して修正
    for (i, row) in table.iter_mut().enumerate().take(n_left_tiles + 1) {
        // 0巡目までに引けない確率は1（まだ引いていない）
        row[0] = 1.;
        // 各巡目での非ツモ確率を累積的に計算
        for j in 0..(MAX_TSUMO - 1).min(n_left_tiles - i) {
            // j巡目で引けない確率 = 前巡までの確率 × (非有効牌数 / 残り牌総数)
            row[j + 1] = row[j] * (n_left_tiles - i - j) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

impl<const MAX_TSUMO: usize> SPCalculatorState<'_, MAX_TSUMO> {
    /// 計算のメインエントリーポイント
    /// 
    /// 向聴数に応じて詳細計算か簡易計算を選択し、結果をソートして返す
    fn calc(&mut self, can_discard: bool, cur_shanten: i8) -> Vec<Candidate> {
        if cur_shanten <= SHANTEN_THRES {
            // 3向聴以下：聴牌確率、和了確率、期待値を詳細計算
            let mut candidates = if can_discard {
                self.analyze_discard(cur_shanten) // 打牌時の分析
            } else {
                self.analyze_draw(cur_shanten)     // ツモ時の分析
            };

            // 結果のソート（設定に応じて期待値順または和了率順）
            if self.sup.sort_result && !candidates.is_empty() {
                let by = if self.sup.maximize_win_prob {
                    CandidateColumn::WinProb // 和了率優先
                } else {
                    CandidateColumn::EV      // 期待値優先（デフォルト）
                };
                candidates.sort_by(|l, r| r.cmp(l, by)); // 降順ソート
            }
            candidates
        } else {
            // 4向聴以上：有効牌枚数のみの簡易計算
            let mut candidates = if can_discard {
                self.analyze_discard_simple(cur_shanten) // 簡易打牌分析
            } else {
                self.analyze_draw_simple()                // 簡易ツモ分析
            };

            // 結果のソート（向聴維持優先）
            if self.sup.sort_result && !candidates.is_empty() {
                candidates.sort_by(|l, r| r.cmp(l, CandidateColumn::NotShantenDown));
            }
            candidates
        }
    }

    /// 打牌時の詳細分析
    /// 
    /// 各打牌候補について期待値・確率を計算
    fn analyze_discard(&mut self, shanten: i8) -> Vec<Candidate> {
        // 現在の手牌から打牌候補を取得
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);

        let mut candidates = Vec::with_capacity(discard_tiles.len());
        for DiscardTile { tile, shanten_diff } in discard_tiles {
            if shanten_diff == 0 {
                // 向聴数維持の場合
                // 一時的に打牌して計算
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                let values = self.draw(shanten); // ツモ後の期待値を計算
                self.state.undo_discard(tile);   // 状態を戻す

                let mut tenpai_probs = values.tenpai_probs;
                if shanten == 0 {
                    // すでに聴牌している場合は聴牌確率100%
                    tenpai_probs.fill(1.);
                }

                // 候補を作成
                let candidate = Candidate::from(RawCandidate {
                    tile,
                    tenpai_probs: &tenpai_probs,
                    win_probs: &values.win_probs,
                    exp_values: &values.exp_values,
                    required_tiles,
                    shanten_down: false, // 向聴維持
                });
                #[cfg(feature = "sp_reproduce_cpp_ver")]
                let candidate = candidate.calibrate(self.real_max_tsumo);
                candidates.push(candidate);
            } else if self.sup.calc_shanten_down && shanten_diff == 1 && shanten < SHANTEN_THRES {
                // 向聴戻しの場合（設定で有効かつ3向聴以下）
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                self.state.n_extra_tsumo += 1;   // 向聴戻しは1巡余分にかかる
                let values = self.draw(shanten + 1); // 向聴数+1で計算
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);

                // 候補を作成
                let candidate = Candidate::from(RawCandidate {
                    tile,
                    tenpai_probs: &values.tenpai_probs,
                    win_probs: &values.win_probs,
                    exp_values: &values.exp_values,
                    required_tiles,
                    shanten_down: true, // 向聴戻し
                });
                #[cfg(feature = "sp_reproduce_cpp_ver")]
                let candidate = candidate.calibrate(self.real_max_tsumo);
                candidates.push(candidate);
            }
        }
        candidates
    }

    fn analyze_draw(&mut self, shanten: i8) -> Vec<Candidate> {
        let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
        let values = self.draw(shanten);

        let mut tenpai_probs = values.tenpai_probs;
        if shanten == 0 {
            // すでに聴牌している場合の例外処理
            tenpai_probs.fill(1.);
        }

        let candidate = Candidate::from(RawCandidate {
            tile: t!(?),
            tenpai_probs: &tenpai_probs,
            win_probs: &values.win_probs,
            exp_values: &values.exp_values,
            required_tiles,
            shanten_down: false,
        });
        #[cfg(feature = "sp_reproduce_cpp_ver")]
        let candidate = candidate.calibrate(self.real_max_tsumo);
        vec![candidate]
    }

    fn analyze_discard_simple(&mut self, shanten: i8) -> Vec<Candidate> {
        // 打牌候補を取得する。
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);
        discard_tiles
            .into_iter()
            .map(|DiscardTile { tile, shanten_diff }| {
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
                self.state.undo_discard(tile);

                Candidate::from(RawCandidate {
                    tile,
                    required_tiles,
                    shanten_down: shanten_diff == 1,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn analyze_draw_simple(&mut self) -> Vec<Candidate> {
        let required_tiles = self.state.get_required_tiles(self.sup.tehai_len_div3);
        let candidate = Candidate::from(RawCandidate {
            tile: t!(?),
            required_tiles,
            shanten_down: false,
            ..Default::default()
        });
        vec![candidate]
    }

    /// ツモ後の期待値計算
    /// 
    /// 手変わり考慮の有無に応じて適切な計算方法を選択
    fn draw(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        if self.sup.calc_tegawari && self.state.n_extra_tsumo == 0 {
            self.draw_with_tegawari(shanten)    // 手変わり考慮あり
        } else {
            self.draw_without_tegawari(shanten) // 手変わり考慮なし
        }
    }

    fn draw_with_tegawari(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.draw_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.draw_with_tegawari_slow(shanten))
    }

    fn draw_with_tegawari_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        let mut tenpai_probs = [0.; MAX_TSUMO];
        let mut win_probs = [0.; MAX_TSUMO];
        let mut exp_values = [0.; MAX_TSUMO];

        // 自摸候補を取得する。
        let draw_tiles = self.state.get_draw_tiles(shanten, self.sup.tehai_len_div3);

        // 有効牌の合計枚数を計算する。【暫定対応】
        let sum_left_tiles = self.state.sum_left_tiles();

        for &DrawTile {
            tile,
            count,
            shanten_diff,
        } in &draw_tiles
        {
            if shanten_diff != -1 {
                // 有効牌以外の場合
                continue;
            }

            self.state.deal(tile);
            let scores_or_values = if shanten > 0 {
                ScoresOrValues::Values(self.discard(shanten - 1))
            } else if let Some(scores) = self.get_score(tile) {
                ScoresOrValues::Scores(scores)
            } else {
                self.state.undo_deal(tile);
                continue;
            };
            self.state.undo_deal(tile);

            // 【暫定対応】 (2021/9/24)
            // FIX_TEGAWARI_PROB について
            // draw_without_tegawari() で有効牌が引けない場合、有効牌以外のどの牌を引いたのかということは考慮していないため、
            // counts で管理している各牌の残りの合計枚数 > 現在の巡目の残り枚数という状況が発生し、結果的に確率値が1を超えてしまう。
            // 実際に正しい確率値を求めるには、draw_without_tegawari() でどの牌を引いたのかをすべてシミュレーションする必要があるが、
            // 計算量的に難しいので、巡目に関係なく、
            //「自摸の確率 = 牌の残り枚数 / 残り枚数の合計」で確率値が1を超えないように暫定対応した。

            for i in 0..MAX_TSUMO {
                // 【暫定対応】 (2021/9/24)
                let tump_prob = count as f32 / sum_left_tiles as f32;
                // let tump_prob = &TSUMO_PROB_TABLE[count as usize - 1][i];

                match &scores_or_values {
                    ScoresOrValues::Scores(scores) => {
                        let assume_riichi = self.sup.is_menzen && self.sup.prefer_riichi;
                        // 聴牌の場合は次で和了
                        // i 巡目で聴牌の場合はダブル立直成立
                        let win_double_riichi =
                            assume_riichi && self.sup.calc_double_riichi && i == 0;
                        // i 巡目で聴牌し、次の巡目で和了の場合は一発成立
                        let win_ippatsu = assume_riichi;
                        // 最後の巡目で和了の場合は海底撈月成立
                        let win_haitei = self.sup.calc_haitei && i == MAX_TSUMO - 1;
                        let han_plus =
                            win_double_riichi as usize + win_ippatsu as usize + win_haitei as usize;

                        win_probs[i] += tump_prob;
                        exp_values[i] += tump_prob * scores[han_plus];
                    }
                    ScoresOrValues::Values(next_values) => {
                        if shanten == 1 {
                            // 1向聴の場合は次で聴牌
                            tenpai_probs[i] += tump_prob;
                        }
                        if i < MAX_TSUMO - 1 {
                            if shanten > 1 {
                                tenpai_probs[i] += tump_prob * next_values.tenpai_probs[i + 1];
                            }
                            win_probs[i] += tump_prob * next_values.win_probs[i + 1];
                            exp_values[i] += tump_prob * next_values.exp_values[i + 1];
                        }
                    }
                }
            }
        }

        for DrawTile {
            tile,
            count,
            shanten_diff,
        } in draw_tiles
        {
            if shanten_diff != 0 {
                // 有効牌の場合
                continue;
            }

            self.state.deal(tile);
            self.state.n_extra_tsumo += 1;
            let next_values = self.discard(shanten);
            self.state.n_extra_tsumo -= 1;
            self.state.undo_deal(tile);

            for i in 0..MAX_TSUMO - 1 {
                // 【暫定対応】 (2021/9/24)
                let tump_prob = count as f32 / sum_left_tiles as f32;
                // let tump_prob = &TSUMO_PROB_TABLE[count as usize - 1][i];

                tenpai_probs[i] += tump_prob * next_values.tenpai_probs[i + 1];
                win_probs[i] += tump_prob * next_values.win_probs[i + 1];
                exp_values[i] += tump_prob * next_values.exp_values[i + 1];
            }
        }

        let values = Rc::new(Values {
            tenpai_probs,
            win_probs,
            exp_values,
        });
        self.draw_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    fn draw_without_tegawari(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.draw_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.draw_without_tegawari_slow(shanten))
    }

    fn draw_without_tegawari_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        let mut tenpai_probs = [0.; MAX_TSUMO];
        let mut win_probs = [0.; MAX_TSUMO];
        let mut exp_values = [0.; MAX_TSUMO];

        // 自摸候補を取得する。
        let draw_tiles = self.state.get_draw_tiles(shanten, self.sup.tehai_len_div3);

        // 有効牌の合計枚数を計算する。
        let sum_required_tiles: u8 = draw_tiles
            .iter()
            .filter(|d| d.shanten_diff == -1)
            .map(|d| d.count)
            .sum();
        let not_tsumo_probs = &self.not_tsumo_prob_table[sum_required_tiles as usize];

        for DrawTile {
            tile,
            count,
            shanten_diff,
        } in draw_tiles
        {
            if shanten_diff != -1 {
                // 有効牌以外の場合
                continue;
            }

            self.state.deal(tile);
            let scores_or_values = if shanten > 0 {
                ScoresOrValues::Values(self.discard(shanten - 1))
            } else if let Some(scores) = self.get_score(tile) {
                ScoresOrValues::Scores(scores)
            } else {
                self.state.undo_deal(tile);
                continue;
            };
            self.state.undo_deal(tile);

            let tsumo_probs = &self.tsumo_prob_table[count as usize - 1];
            for i in 0..MAX_TSUMO {
                let m = not_tsumo_probs[i];
                if m == 0. {
                    // We are breaking here because `not_tsumo_probs[i..]` must
                    // all be zero, since `not_tsumo_probs` is monotonically
                    // decreasing.
                    //
                    // This divide-by-zero check is missing in the original
                    // version, which is very problematic.
                    break;
                }

                for j in i..MAX_TSUMO {
                    let n = not_tsumo_probs[j];
                    if n == 0. {
                        // `not_tsumo_probs[j..]` must all be zero, no need to
                        // proceed.
                        break;
                    }
                    // 現在の巡目が i の場合に j 巡目に有効牌を引く確率
                    let prob = tsumo_probs[j] * n / m;

                    match &scores_or_values {
                        ScoresOrValues::Scores(scores) => {
                            let assume_riichi = self.sup.is_menzen && self.sup.prefer_riichi;
                            // 聴牌の場合は次で和了
                            // i 巡目で聴牌の場合はダブル立直成立
                            let win_double_riichi =
                                assume_riichi && self.sup.calc_double_riichi && i == 0;
                            // i 巡目で聴牌し、次の巡目で和了の場合は一発成立
                            let win_ippatsu = assume_riichi && j == i;
                            // 最後の巡目で和了の場合は海底撈月成立
                            let win_haitei = self.sup.calc_haitei && j == MAX_TSUMO - 1;
                            let han_plus = win_double_riichi as usize
                                + win_ippatsu as usize
                                + win_haitei as usize;

                            win_probs[i] += prob;
                            exp_values[i] += prob * scores[han_plus];
                        }
                        ScoresOrValues::Values(next_values) => {
                            if shanten == 1 {
                                // 1向聴の場合は次で聴牌
                                tenpai_probs[i] += prob;
                            }
                            if j < MAX_TSUMO - 1 {
                                if shanten > 1 {
                                    // 2向聴以上で max_tsumo_ - 1 巡目以下の場合
                                    tenpai_probs[i] += prob * next_values.tenpai_probs[j + 1];
                                }
                                // 聴牌以上で max_tsumo_ - 1 巡目以下の場合
                                win_probs[i] += prob * next_values.win_probs[j + 1];
                                exp_values[i] += prob * next_values.exp_values[j + 1];
                            }
                        }
                    }
                }
            }
        }

        let values = Rc::new(Values {
            tenpai_probs,
            win_probs,
            exp_values,
        });
        self.draw_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    /// 打牌後の期待値計算（キャッシュ付き）
    fn discard(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.discard_cache[shanten as usize]
            .get(&self.state)
            .cloned()
            .unwrap_or_else(|| self.discard_slow(shanten))
    }

    /// 打牌後の期待値計算（実際の計算処理）
    /// 
    /// 全ての打牌候補を評価し、最適な打牌を選択
    fn discard_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        // 現在の手牌から打牌候補を取得
        let discard_tiles = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);

        // 各巡目での最大値を保持する配列
        let mut max_tenpai_probs = [f32::MIN; MAX_TSUMO];
        let mut max_win_probs = [f32::MIN; MAX_TSUMO];
        let mut max_exp_values = [f32::MIN; MAX_TSUMO];
        let mut max_tiles = [t!(?); MAX_TSUMO]; // 最適な打牌
        let mut max_values = [i32::MIN; MAX_TSUMO]; // 比較用の値

        for DiscardTile { tile, shanten_diff } in discard_tiles {
            let values;
            if shanten_diff == 0 {
                // 向聴数が変化しない打牌
                self.state.discard(tile);
                values = self.draw(shanten);
                self.state.undo_discard(tile);
            } else if self.sup.calc_shanten_down
                && self.state.n_extra_tsumo == 0
                && shanten_diff == 1
                && shanten < SHANTEN_THRES
            {
                // 向聴戻しになる打牌
                self.state.discard(tile);
                self.state.n_extra_tsumo += 1;
                values = self.draw(shanten + 1);
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);
            } else {
                // 手牌に存在しない牌、または向聴落としが無効な場合に向聴落としとなる牌
                continue;
            };

            for i in 0..MAX_TSUMO {
                // 和了確率は下2桁まで一致していれば同じ、期待値は下0桁まで一致していれば同じとみなす。
                let value = if self.sup.maximize_win_prob {
                    values.win_probs[i] * 1e5
                } else {
                    values.exp_values[i]
                } as i32;
                let max_value = max_values[i];
                let max_tile = max_tiles[i];

                if value > max_value
                    || value == max_value && tile.cmp_discard_priority(max_tile).is_gt()
                {
                    // 値が同等なら、DiscardPriorities が高い牌を優先して選択する。
                    max_tenpai_probs[i] = values.tenpai_probs[i];
                    max_win_probs[i] = values.win_probs[i];
                    max_exp_values[i] = values.exp_values[i];
                    max_values[i] = value;
                    max_tiles[i] = tile;
                }
            }
        }

        let values = Rc::new(Values {
            tenpai_probs: max_tenpai_probs,
            win_probs: max_win_probs,
            exp_values: max_exp_values,
        });
        self.discard_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));

        values
    }

    /// 和了時の点数計算
    /// 
    /// 指定された牌で和了した場合の点数を計算
    /// 裏ドラや追加役（ダブリー、一発、海底）を考慮した4パターンの点数を返す
    /// 
    /// # Returns
    /// Some([基本点, +1翻, +2翻, +3翻]) または None（役なし）
    fn get_score(&self, win_tile: Tile) -> Option<[f32; 4]> {
        // アガリ計算機を初期化
        let calc = AgariCalculator {
            tehai: &self.state.tehai,
            is_menzen: self.sup.is_menzen,
            chis: self.sup.chis,
            pons: self.sup.pons,
            minkans: self.sup.minkans,
            ankans: self.sup.ankans,
            bakaze: self.sup.bakaze,
            jikaze: self.sup.jikaze,
            winning_tile: win_tile.deaka().as_u8(),
            is_ron: false, // シングルプレイヤーは常にツモ
        };
        let is_oya = self.sup.jikaze == tu8!(E); // 親かどうか

        let additional_yakus = match (self.sup.is_menzen, self.sup.prefer_riichi) {
            (true, true) => 2,
            (true, false) => 1,
            (false, _) => 0,
        };
        let num_doras = self
            .sup
            .dora_indicators
            .iter()
            .map(|ind| self.state.tehai[ind.next().as_usize()])
            .sum::<u8>()
            + self.state.akas_in_hand.iter().filter(|&&b| b).count() as u8
            + self.sup.num_doras_in_fuuro;

        // Although you can technically win the base hand with just 海底, the
        // original C++ version didn't take this into account and I also agree
        // with that.
        let (fu, han) = match calc.agari(additional_yakus, num_doras)? {
            Agari::Normal { fu, han } => (fu, han),
            a @ Agari::Yakuman(_) => {
                return Some([a.point(is_oya).tsumo_total(is_oya) as f32; 4]);
            }
        };

        // 役ありの場合

        // ダブル立直、一発、海底撈月で最大3翻まで増加するので、
        // ベースとなる点数、+1翻の点数、+2翻の点数、+3翻の点数も計算しておく。
        let mut scores = [0.; 4];

        let assume_riichi = self.sup.is_menzen && self.sup.prefer_riichi;
        if assume_riichi && self.sup.dora_indicators.len() == 1 {
            // 裏ドラ考慮ありかつ表ドラが1枚以上の場合は、厳密に計算する。
            let mut n_indicators = [0; 5];
            let mut sum_indicators = 0;
            for (tid, &count) in self.state.tehai.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                // ドラ表示牌の枚数を数える。
                let tile = must_tile!(tid);
                let ind_count = self.state.tiles_in_wall[tile.prev().as_usize()];
                n_indicators[count as usize] += ind_count;
                sum_indicators += ind_count;
            }

            // 裏ドラの乗る確率を枚数ごとに計算する。
            let mut uradora_probs = [0.; 5];

            #[cfg(feature = "sp_reproduce_cpp_ver")]
            // 厳密に計算するなら残り枚数は数えるべきだが、あまり影響がないので
            // 121枚で固定
            let n_left_tiles = 121;
            #[cfg(not(feature = "sp_reproduce_cpp_ver"))]
            let n_left_tiles = self.state.sum_left_tiles();

            uradora_probs[0] = (n_left_tiles - sum_indicators) as f32 / n_left_tiles as f32;
            for i in 1..5 {
                uradora_probs[i] = n_indicators[i] as f32 / n_left_tiles as f32;
            }

            for (i, s) in scores.iter_mut().enumerate() {
                // 裏ドラ1枚の場合、最大4翻まで乗る可能性がある
                for (j, &p) in uradora_probs.iter().enumerate() {
                    if p == 0. {
                        continue;
                    }
                    let agari = Agari::Normal {
                        fu,
                        han: han + i as u8 + j as u8,
                    };
                    *s += agari.point(is_oya).tsumo_total(is_oya) as f32 * p;
                }
            }
        } else if assume_riichi && self.sup.dora_indicators.len() > 1 {
            // 裏ドラ考慮ありかつ表ドラが2枚以上の場合、統計データを利用する。
            for (i, s) in scores.iter_mut().enumerate() {
                for (j, &p) in URADORA_PROB_TABLE[self.sup.dora_indicators.len() - 1]
                    .iter()
                    .enumerate()
                {
                    if p == 0. {
                        continue;
                    }
                    let agari = Agari::Normal {
                        fu,
                        han: han + i as u8 + j as u8,
                    };
                    *s += agari.point(is_oya).tsumo_total(is_oya) as f32 * p;
                }
            }
        } else {
            // 裏ドラ考慮なしまたは表ドラが0枚の場合
            for (i, s) in scores.iter_mut().enumerate() {
                let agari = Agari::Normal {
                    fu,
                    han: han + i as u8,
                };
                *s = agari.point(is_oya).tsumo_total(is_oya) as f32;
            }
        }

        Some(scores)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algo::sp::CALC_SHANTEN_FN;
    use crate::hand::hand;
    use crate::tuz;

    fn feq(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::EPSILON
    }

    #[test]
    fn nanikiru() {
        let mut calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(N),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &t![P,],
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        let tehai = hand("45678m 34789p 3344z").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 8;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates[0].tile, t!(N));
        assert_eq!(candidates[1].tile, t!(W));
        assert!(candidates[0].exp_values > candidates[1].exp_values);

        // ---

        let tehai = hand("3667m 23489p 34688s").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 15;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state.clone(), can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates[0].tile, t!(9p));
        assert!(candidates[0].shanten_down);

        calc.maximize_win_prob = true;
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates[0].tile, t!(3m));
        assert!(!candidates[0].shanten_down);

        // ---

        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &t![6m,],
            calc_double_riichi: true,
            calc_haitei: true,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        let tehai = hand("45677m 456778p 248s").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let can_discard = true;
        let tsumos_left = 15;
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        let c = if cfg!(feature = "sp_reproduce_cpp_ver") {
            &candidates[2]
        } else {
            &candidates[0]
        };
        assert_eq!(c.tile, t!(2s));
        assert_eq!(c.required_tiles.len(), 17);
        assert_eq!(c.num_required_tiles, 57);
        assert!(c.shanten_down);
        if cfg!(feature = "sp_reproduce_cpp_ver") {
            assert!(feq(c.tenpai_probs[0], 0.88994724));
            assert!(feq(c.win_probs[0], 0.32714003));
            assert!(feq(c.exp_values[0], 5557.188));
        } else {
            assert!(feq(c.tenpai_probs[0], 0.90023905));
            assert!(feq(c.win_probs[0], 0.34794784));
            assert!(feq(c.exp_values[0], 5894.7617));
        }

        // ---

        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(W),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &t![1m,],
            calc_double_riichi: false,
            calc_haitei: false,
            sort_result: true,
            maximize_win_prob: false,
            calc_tegawari: true,
            calc_shanten_down: true,
        };
        let tehai = hand("9999m 6677p 88s 335z 1m").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false; 3],
        };
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let can_discard = true;
        let tsumos_left = 5;
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates.len(), 7);

        // feature = "sp_reproduce_cpp_ver" does not use chitoi shanten
        if !cfg!(feature = "sp_reproduce_cpp_ver") {
            let c = &candidates[1];
            assert_eq!(c.tile, t!(1m));
            assert!(c.shanten_down);
            assert_eq!(c.required_tiles.len(), 33); // literally all kinds of tiles
            assert_eq!(c.num_required_tiles, 34 * 4 - tiles_seen.iter().sum::<u8>());
        }
    }

    #[test]
    fn tsumo_only() {
        let calc = SPCalculator {
            tehai_len_div3: 4,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(W),
            prefer_riichi: true,
            is_menzen: true,
            num_doras_in_fuuro: 0,
            dora_indicators: &t![6m,],
            calc_double_riichi: true,
            calc_haitei: true,
            sort_result: true,
            maximize_win_prob: true,
            calc_tegawari: true,
            calc_shanten_down: true,
        };

        let tehai = hand("45677m 456778p 48s").unwrap();
        let mut tiles_seen = tehai;
        for ind in calc.dora_indicators {
            tiles_seen[ind.deaka().as_usize()] += 1;
        }
        tiles_seen[tuz!(5s)] += 4;

        let state = InitState {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen,
            akas_seen: [false, false, true],
        };
        let cur_shanten = CALC_SHANTEN_FN(&tehai, calc.tehai_len_div3);
        let can_discard = false;
        let tsumos_left = 5;
        let candidates = calc
            .calc(state, can_discard, tsumos_left, cur_shanten)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        let c = &candidates[0];
        assert_eq!(c.tile, t!(?));
        assert_eq!(c.required_tiles.len(), 16);
        assert_eq!(c.num_required_tiles, 54);
        if cfg!(feature = "sp_reproduce_cpp_ver") {
            assert!(feq(c.tenpai_probs[0], 0.4992795));
            assert!(feq(c.win_probs[0], 0.042052355));
            assert!(feq(c.exp_values[0], 527.17926));
        } else {
            assert!(feq(c.tenpai_probs[0], 0.45017204));
            assert!(feq(c.win_probs[0], 0.03441279));
            assert!(feq(c.exp_values[0], 432.26678));
        }
    }
}
