// ================================================================================
// PlayerState ヘルパーメソッド実装
// ================================================================================
// このファイルは、PlayerState構造体に対する麻雀ゲームのAIやエージェントが使用する
// ヘルパーメソッドを実装しています。主な機能：
// 
// 1. 打牌候補の取得 (discard_candidates)
//    - 通常の打牌候補
//    - 聴牌確定打牌の判定
// 
// 2. 流局判定 (rule_based_ryukyoku)
//    - 九種九牌流局の判定
//    - ゲーム状況に基づく流局判断
// 
// 3. 和了判定 (rule_based_agari, agari_points)
//    - 和了可否の判定
//    - 得点計算
// 
// 4. シャンテン数計算 (real_time_shanten)
//    - リアルタイムなシャンテン数の計算
// 
// 5. 期待値計算 (single_player_tables)
//    - 打牌選択のための期待値テーブル作成
// ================================================================================

use super::{PlayerState, SinglePlayerTables};
use crate::algo::agari::AgariCalculator;
use crate::algo::point::Point;
use crate::algo::shanten;
use crate::algo::sp::{InitState, SPCalculator};
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::{must_tile, t, tu8, tuz};

use anyhow::{Context, Result, ensure};
use tinyvec::array_vec;

impl PlayerState {
    /// Used by `BoardState` to check if a player is making 4 kans on his own.
    /// 槓の合計数を返す。四槓子の成立判定に使用される。
    #[inline]
    #[must_use]
    pub fn kans_count(&self) -> usize {
        // 明槓（他家から鳴いた槓）の数 + 暗槓（自分で4枚集めた槓）の数を返す
        self.minkans.len() + self.ankans.len()
    }

    /// Used by `Agent` impls, must be called at 3n+2.
    /// AIエージェントが使用する打牌候補を返す。手牌が3n+2の状態で呼び出す必要がある。
    /// 戻り値は34要素の配列で、各牌が打牌可能ならtrue、不可ならfalse。
    #[must_use]
    pub fn discard_candidates(&self) -> [bool; 34] {
        // 赤牌を含む37要素の打牌候補配列を取得
        let full = self.discard_candidates_aka();
        // 赤牌を除いた34要素の配列を初期化
        let mut ret = [false; 34];
        // 最初の34要素（赤牌を除く通常牌）をコピー
        ret.copy_from_slice(&full[..34]);
        // 赤5萬（5mr）が打牌可能なら、通常の5萬（5m）も打牌可能とする
        ret[tuz!(5m)] |= full[tuz!(5mr)];
        // 赤5索（5sr）が打牌可能なら、通常の5索（5s）も打牌可能とする
        ret[tuz!(5s)] |= full[tuz!(5sr)];
        // 赤5筒（5pr）が打牌可能なら、通常の5筒（5p）も打牌可能とする
        ret[tuz!(5p)] |= full[tuz!(5pr)];
        ret
    }

    /// Aka dora covered version of `discard_candidates`.
    /// 赤牌を含む打牌候補を返す。37要素の配列（通常牌34 + 赤牌3）。
    #[must_use]
    pub fn discard_candidates_aka(&self) -> [bool; 37] {
        // 手牌が3n+2（打牌可能状態）であることを確認
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        // 37要素の打牌候補配列を初期化（全てfalse）
        let mut ret = [false; 37];

        // リーチが成立している場合の処理
        if self.riichi_accepted[0] {
            // リーチ後は最後に自摸した牌しか打牌できない（ツモ切りのみ）
            let last_self_tsumo = self
                .last_self_tsumo
                .expect("riichi accepted without last self tsumo");
            ret[last_self_tsumo.as_usize()] = true;
            return ret;
        }

        // 手牌の各牌について打牌可能かチェック
        for (i, count) in self.tehai.iter().copied().enumerate() {
            // その牌を持っていない場合はスキップ
            if count == 0 {
                continue;
            }

            // リーチ宣言中かどうかで処理を分岐
            ret[i] = if self.riichi_declared[0] {
                // リーチ宣言中の場合
                if self.shanten == 1 {
                    // 1シャンテンの場合：聴牌に向かう打牌のみ可能
                    self.next_shanten_discards[i]
                } else {
                    // 0シャンテン（聴牌）の場合：待ちを変えない打牌のみ可能
                    // shanten must be 0 here according to the rule
                    self.keep_shanten_discards[i]
                }
            } else {
                // 通常時：フリテン牌でなければ打牌可能
                !self.forbidden_tiles[i]
            };
        }

        // 赤牌の処理（5萬、5筒、5索）
        // 赤5萬を持っている場合
        if ret[tuz!(5m)] && self.akas_in_hand[0] {
            ret[tuz!(5mr)] = true;  // 赤5萬も打牌可能
            // 通常の5萬は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5m)] = self.tehai[tuz!(5m)] > 1;
        }
        // 赤5筒を持っている場合
        if ret[tuz!(5p)] && self.akas_in_hand[1] {
            ret[tuz!(5pr)] = true;  // 赤5筒も打牌可能
            // 通常の5筒は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5p)] = self.tehai[tuz!(5p)] > 1;
        }
        // 赤5索を持っている場合
        if ret[tuz!(5s)] && self.akas_in_hand[2] {
            ret[tuz!(5sr)] = true;  // 赤5索も打牌可能
            // 通常の5索は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5s)] = self.tehai[tuz!(5s)] > 1;
        }

        ret
    }

    /// Must be called at 3n+2.
    ///
    /// The return value indicates the tiles which can make the hand tenpai for
    /// real after being discarded, with the number of future tenpai tiles left
    /// and furiten considered, without depending on any incidental yaku, and is
    /// not affected by the riichi status of the player.
    /// 手牌が3n+2の状態で呼び出す必要がある。
    /// 打牌後に確実に聴牌となり、役があることが保証される牌を返す。
    /// フリテンや残り牌数を考慮し、偶発役に依存しない打牌を判定する。
    #[must_use]
    pub fn discard_candidates_with_unconditional_tenpai(&self) -> [bool; 34] {
        // 赤牌を含む37要素の結果を取得
        let full = self.discard_candidates_with_unconditional_tenpai_aka();
        // 赤牌を除いた34要素の配列を初期化
        let mut ret = [false; 34];
        // 最初の34要素をコピー
        ret.copy_from_slice(&full[..34]);
        // 赤牌の結果を通常牌に反映
        ret[tuz!(5m)] |= full[tuz!(5mr)];
        ret[tuz!(5s)] |= full[tuz!(5sr)];
        ret[tuz!(5p)] |= full[tuz!(5pr)];
        ret
    }

    /// Aka dora covered version of `discard_candidates_with_unconditional_tenpai`.
    /// 赤牌を含む無条件聴牌の打牌候補を返す。
    #[must_use]
    pub fn discard_candidates_with_unconditional_tenpai_aka(&self) -> [bool; 37] {
        // 手牌が3n+2であることを確認
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        // 37要素の結果配列を初期化（全てfalse）
        let mut ret = [false; 37];

        // 聴牌不可能な条件をチェック
        if self.tiles_left == 0 // 海底（残り牌なし）
            || self.shanten > 1 // 2シャンテン以上は打牌で聴牌にならない
            || self.shanten == 1 && !self.has_next_shanten_discard // 1シャンテンだが聴牌に向かう打牌がない
        {
            return ret;
        }

        // 自摸牌がある場合の処理
        if let Some(last_self_tsumo) = self.last_self_tsumo {
            // 自摸牌が待ち牌の場合（既に和了形）
            if self.waits[last_self_tsumo.deaka().as_usize()] {
                // すでに和了形なので、どの牌を打ってもフリテンになる
                return ret;
            }
            // リーチが成立している場合
            if self.riichi_accepted[0] {
                if !self.at_furiten {
                    // リーチ中でフリテンでない場合、ツモ切りのみ可能
                    ret[last_self_tsumo.as_usize()] = true;
                }
                return ret;
            }
        } else if shanten::calc_all(&self.tehai, self.tehai_len_div3) == -1 {
            // チー・ポン後で既に和了形の場合も同様
            return ret;
        }

        // 聴牌に向かう打牌を取得
        let tenpai_discards = if self.shanten == 1 {
            // 1シャンテンの場合：聴牌に向かう打牌
            self.next_shanten_discards
        } else {
            // 0シャンテン（聴牌）の場合：待ちを維持する打牌
            self.keep_shanten_discards
        };

        // 各打牌候補について聴牌と役の有無をチェック
        tenpai_discards
            .iter()
            .copied()
            .enumerate()
            .filter(|&(tid, b)| b && !self.forbidden_tiles[tid])  // 打牌可能かつフリテン牌でない
            .for_each(|(discard, _)| {
                // 打牌後の手牌（3n+1）を作成
                let mut tehai_3n1 = self.tehai;
                tehai_3n1[discard] -= 1;

                // 全ての可能な待ち牌をチェック
                for (tsumo, seen) in self.tiles_seen.iter().copied().enumerate() {
                    // 打牌と同じ牌、または既に4枚見えている牌はスキップ
                    if tsumo == discard || tehai_3n1[tsumo] == 4 {
                        continue;
                    }

                    // ツモ後の手牌（3n+2）を作成
                    let mut tehai_3n2 = tehai_3n1;
                    tehai_3n2[tsumo] += 1;
                    // 和了形でない場合はスキップ
                    if shanten::calc_all(&tehai_3n2, self.tehai_len_div3) > -1 {
                        continue;
                    }

                    // フリテンチェック：待ち牌を既に捨てている場合
                    if self.discarded_tiles[tsumo] {
                        ret[discard] = false;
                        break;  // この打牌は無条件聴牌にならない
                    }

                    // フリテンチェック後の処理
                    // 4枚全て見えている、または既に無条件聴牌と判定済みの場合はスキップ
                    if seen == 4 || ret[discard] {
                        continue;
                    }

                    // 役の有無をチェック（偶発役に依存しない）
                    let agari_calc = AgariCalculator {
                        tehai: &tehai_3n2,              // 和了形の手牌
                        is_menzen: self.is_menzen,      // 門前かどうか
                        chis: &self.chis,               // チーした牌
                        pons: &self.pons,               // ポンした牌
                        minkans: &self.minkans,         // 明槓
                        ankans: &self.ankans,           // 暗槓
                        bakaze: self.bakaze.as_u8(),    // 場風
                        jikaze: self.jikaze.as_u8(),    // 自風
                        winning_tile: tsumo as u8,      // 和了牌
                        is_ron: true,                   // ロン和了として計算
                    };
                    // 役があれば、この打牌は無条件聴牌となる
                    ret[discard] = agari_calc.has_yaku();
                }
            });

        // 赤牌の処理
        // 赤5萬を持っている場合
        if ret[tuz!(5m)] && self.akas_in_hand[0] {
            ret[tuz!(5mr)] = true;  // 赤5萬も打牌可能
            // 通常の5萬は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5m)] = self.tehai[tuz!(5m)] > 1;
        }
        // 赤5筒を持っている場合
        if ret[tuz!(5p)] && self.akas_in_hand[1] {
            ret[tuz!(5pr)] = true;  // 赤5筒も打牌可能
            // 通常の5筒は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5p)] = self.tehai[tuz!(5p)] > 1;
        }
        // 赤5索を持っている場合
        if ret[tuz!(5s)] && self.akas_in_hand[2] {
            ret[tuz!(5sr)] = true;  // 赤5索も打牌可能
            // 通常の5索は2枚以上持っている場合のみ打牌可能
            ret[tuz!(5s)] = self.tehai[tuz!(5s)] > 1;
        }

        ret
    }

    /// 手牌中の円九牌（ヤオチュウ牌）の種類数を返す。
    /// 九種九牌流局の判定などに使用される。
    #[inline]
    #[must_use]
    pub fn yaokyuu_kind_count(&self) -> u8 {
        // 円九牌の定義：1・9萬、１・９筒、１・９索、東南西北白発中
        tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C]
            .iter()
            // 各円九牌を持っているか（0または1）をカウント
            .map(|&i| self.tehai[i].min(1))
            // 合計して種類数を返す
            .sum()
    }

    /// ルールベースの九種九牌流局判定。
    /// ゲーム状況を考慮して流局するかどうかを判断する。
    #[inline]
    #[must_use]
    pub fn rule_based_ryukyoku(&self) -> bool {
        // 流局可能でない場合はfalse
        if !self.last_cans.can_ryukyoku {
            return false;
        }
        // 詳細な判定処理を呼び出す
        self.rule_based_ryukyoku_slow()
    }

    /// 詳細な流局判定ロジック。
    /// シャンテン数、場況、点数状況などを総合的に判断する。
    fn rule_based_ryukyoku_slow(&self) -> bool {
        // 2シャンテン以下の場合は流局しない（和了に近いため）
        if shanten::calc_all(&self.tehai, self.tehai_len_div3) <= 2 {
            return false;
        }

        // 西場の場合は流局する（大きな手が必要ないため）
        if self.bakaze == t!(W) {
            return true;
        }

        // オーラス（最終局）の場合の判定
        if self.is_all_last {
            // 親の場合、またはラス目でない場合は流局する
            // （流局判断が難しいため）
            if self.oya == 0 || self.rank < 3 {
                return true;
            }

            // オーラスでラス目かつ子の場合の判定
            // 跳満ツモでもラス回避できない場合は流局しない
            // 跳満ツモの点数計算
            let mut scores = [-3000 - self.honba as i32 * 300; 4];  // 子の支払い
            scores[0] = 12000 + self.kyotaku as i32 * 1000 + self.honba as i32 * 300;  // 自分の収入
            scores[self.oya as usize] = -6000 - self.honba as i32 * 300;  // 親の支払い
            vec_add_assign(&mut scores, &self.scores);  // 現在の点数に加算
            // 跳満ツモしてもラス目なら流局しない
            return self.get_rank(scores) < 3;
        }

        // 円九牌が10種類以上ある場合は流局しない（国士無双を狙える）
        if self.yaokyuu_kind_count() >= 10 {
            return false;
        }

        // 字牌を全種類持っている場合は流局しない（大三元や小四喜の可能性）
        if self.tehai[3 * 9..].iter().all(|&c| c > 0) {
            return false;
        }

        // 上記の条件に該当しない場合は流局する
        true
    }

    /// ルールベースの和了判定。
    /// ゲーム状況を考慮して和了するかどうかを判断する。
    #[inline]
    #[must_use]
    pub fn rule_based_agari(&self) -> bool {
        // 和了可能でない場合はfalse
        if !self.last_cans.can_agari() {
            return false;
        }
        // 詳細な判定処理を呼び出す
        self.rule_based_agari_slow(
            self.last_cans.can_ron_agari,  // ロン和了かどうか
            self.rel(self.last_cans.target_actor),  // 放銃者の相対位置
        )
    }

    /// 詳細な和了判定ロジック。
    /// オーラスでの点数状況を考慮して和了判断を行う。
    fn rule_based_agari_slow(&self, is_ron: bool, target_rel: usize) -> bool {
        // オーラスでない、または親、またはラス目でない場合は必ず和了
        if !self.is_all_last || self.oya == 0 || self.rank < 3 {
            return true;
        }

        if self.bakaze == t!(W) {
            // 西場で西4局（本当の最終局）でない場合は和了
            if self.kyoku < 3 {
                return true;
            }
        } else if self.scores.iter().all(|&s| s < 30000) {
            // 西入（延長戦）が可能な場合は和了
            // 注：この条件は完全ではないが安全側の判定
            return true;
        }

        // この和了で獲得可能な最大点数を計算
        let max_win_point = if self.riichi_accepted[0] {
            // リーチ成立時：裏ドラを最大化して計算
            // 暗槓を含む完全な手牌を作成
            let mut tehai_full = self.tehai;
            for t in &self.ankan_overview[0] {
                tehai_full[t.as_usize()] += 4;
            }

            // 手牌を枚数の多い順にソート（裏ドラになりやすい牌を優先）
            let mut tehai_ordered_by_count: Vec<_> = tehai_full
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .collect();
            tehai_ordered_by_count.sort_unstable_by(|(_, l), (_, r)| r.cmp(l));

            // 可能な裏ドラを価値の高い順に試す
            let mut tiles_seen = self.tiles_seen;
            let mut ura_indicators = array_vec!([_; 5]);
            'outer: for (t, _) in tehai_ordered_by_count {
                // ドラ表示牌はドラの前の牌
                let ura_ind = must_tile!(t).prev();
                loop {
                    // 裏ドラ表示牌がドラ表示牌数に達したら終了
                    if ura_indicators.len() >= self.dora_indicators.len() {
                        break 'outer;
                    }
                    // 4枚全て見えている場合は次の候補へ
                    if tiles_seen[ura_ind.as_usize()] >= 4 {
                        continue 'outer;
                    }
                    // 裏ドラ表示牌として追加
                    ura_indicators.push(ura_ind);
                    tiles_seen[ura_ind.as_usize()] += 1;
                }
            }

            // 点数計算（unwrapは安全：can_agari()で事前チェック済み）
            self.agari_points(is_ron, &ura_indicators).unwrap()
        } else {
            // リーチしていない場合：裏ドラなしで計算
            self.agari_points(is_ron, &[]).unwrap()
        };

        // 和了後の点数状況を計算
        let mut exp_scores = self.scores;
        if is_ron {
            // ロン和了の場合
            exp_scores[0] +=
                max_win_point.ron + self.kyotaku as i32 * 1000 + self.honba as i32 * 300;
            exp_scores[target_rel] -= max_win_point.ron + self.honba as i32 * 300;
        } else {
            // ツモ和了の場合（ここでは必ず子）
            exp_scores[0] += max_win_point.tsumo_total(false)  // 子のツモ和了合計
                + self.kyotaku as i32 * 1000  // 供託
                + self.honba as i32 * 300;    // 本場のボーナス
            // 他家からの支払いを計算
            exp_scores
                .iter_mut()
                .enumerate()
                .skip(1)
                .for_each(|(idx, s)| {
                    if idx as u8 == self.oya {
                        // 親の支払い
                        *s -= max_win_point.tsumo_oya + self.honba as i32 * 100;
                    } else {
                        // 子の支払い
                        *s -= max_win_point.tsumo_ko + self.honba as i32 * 100;
                    }
                });
        }

        // 西4局でないことは既にチェック済み
        //
        // 西入または西入継続が可能な場合は和了
        // この条件は完全かつ安全
        if exp_scores.iter().all(|&s| s < 30000) {
            return true;
        }

        // 理論上の最大点でラス回避できる場合のみ和了
        self.get_rank(exp_scores) < 3
    }

    /// Err is returned if the hand cannot agari, or cannot retrieve the winning
    /// tile.
    ///
    /// This function should be called immediately, otherwise the state may
    /// change.
    ///
    /// `ura_indicators` is used only when the actor has an accepted riichi.
    /// 和了点数を計算する。和了できない場合や和了牌が取得できない場合はErrを返す。
    /// この関数は状態が変わる前に即座に呼び出す必要がある。
    /// `ura_indicators`はリーチ成立時のみ使用される。
    pub fn agari_points(&self, is_ron: bool, ura_indicators: &[Tile]) -> Result<Point> {
        // 和了可能かどうかを確認
        ensure!(
            is_ron && self.last_cans.can_ron_agari || self.last_cans.can_tsumo_agari,
            "cannot agari"
        );

        // 天和・地和の特別処理（複数役満は不可）
        // Here, 天和 and 地和 are handled individually as special cases, and
        // there is no multi yakuman for these two.
        if !is_ron && self.can_w_riichi {
            // ダブルリーチ可能な状態でツモ = 天和（親）または地和（子）
            return Ok(Point::yakuman(self.oya == 0, 1));
        }

        // 和了牌を取得
        let winning_tile = if is_ron {
            self.last_kawa_tile  // ロンの場合：最後の河牌
        } else {
            self.last_self_tsumo  // ツモの場合：最後の自摸牌
        }
        .context("cannot find the winning tile")?;

        // 追加翻数を計算（手牌構成以外の役）
        let additional_hans = if is_ron {
            // ロン和了の場合
            [
                self.riichi_accepted[0],       // 立直（リーチ）
                self.is_w_riichi,              // 両立直（ダブルリーチ）
                self.at_ippatsu,               // 一発（イッパツ）
                self.tiles_left == 0,          // 河底撈魚（ホウテイラオユイ）
                self.chankan_chance.is_some(), // 槍槓（チャンカン）
            ]
            .iter()
            .filter(|&&b| b)
            .count() as u8
        } else {
            // ツモ和了の場合
            [
                self.riichi_accepted[0],                  // 立直（リーチ）
                self.is_w_riichi,                         // 両立直（ダブルリーチ）
                self.at_ippatsu,                          // 一発（イッパツ）
                self.is_menzen,                           // 門前清自摸和（メンゼンツモ）
                self.tiles_left == 0 && !self.at_rinshan, // 海底摸月（ハイテイモーユエ）
                self.at_rinshan,                          // 嶺上開花（リンシャンカイホウ）
            ]
            .iter()
            .filter(|&&b| b)
            .count() as u8
        };

        // 和了形の手牌とドラ数を計算
        let mut tehai = self.tehai;
        let mut final_doras_owned = self.doras_owned[0];
        if is_ron {
            // ロンの場合：和了牌を手牌に追加
            let tid = winning_tile.deaka().as_usize();
            tehai[tid] += 1;
            final_doras_owned += self.dora_factor[tid];  // ドラを加算
            if winning_tile.is_aka() {
                final_doras_owned += 1;  // 赤牌自体もドラ
            };
        }
        // リーチ成立時：裏ドラを計算
        if self.riichi_accepted[0] {
            final_doras_owned += ura_indicators
                .iter()
                .map(|&ura| {
                    let next = ura.next();  // 裏ドラ表示牌の次の牌が裏ドラ
                    let mut count = tehai[next.as_usize()];
                    // 暗槓中の牌も含める
                    if self.ankan_overview[0].contains(&next) {
                        count += 4;
                    }
                    count
                })
                .sum::<u8>();
        }

        // 和了計算器を作成
        let agari_calc = AgariCalculator {
            tehai: &tehai,                              // 和了形の手牌
            is_menzen: self.is_menzen,                  // 門前かどうか
            chis: &self.chis,                           // チーした牌
            pons: &self.pons,                           // ポンした牌
            minkans: &self.minkans,                     // 明槓
            ankans: &self.ankans,                       // 暗槓
            bakaze: self.bakaze.as_u8(),                // 場風
            jikaze: self.jikaze.as_u8(),                // 自風
            winning_tile: winning_tile.deaka().as_u8(), // 和了牌（赤牌を通常牌に変換）
            is_ron,                                     // ロン和了かどうか
        };
        // 和了と役を計算
        let agari = agari_calc
            .agari(additional_hans, final_doras_owned)
            .context("not a hora hand")?;

        // 点数を計算して返す
        Ok(agari.point(self.oya == 0))
    }

    /// Calculate the actual shanten at this point. Unlike `self.shanten`, this
    /// function properly calculates the shanten at 3n+2, which follows the
    /// definition of shanten most people acknowledge.
    /// リアルタイムのシャンテン数を計算する。`self.shanten`とは異なり、
    /// 3n+2の状態でも正確なシャンテン数を計算する。
    pub fn real_time_shanten(&self) -> i8 {
        if !self.last_cans.can_discard {
            // 3n+1の場合：`self.shanten`が正確
            return self.shanten;
        }

        if self.shanten > 0 {
            // 3n+2で聴牌でない場合：シャンテン数を減らす打牌がある場合は
            // `self.shanten - 1`、ない場合は`self.shanten`を返す
            return if self.has_next_shanten_discard {
                self.shanten - 1
            } else {
                self.shanten
            };
        }

        if let Some(tile) = self.last_self_tsumo {
            // 3n+2、自摸後の聴牌状態
            return if self.waits[tile.deaka().as_usize()] {
                -1  // 自摸牌が待ち牌なら和了形（-1）
            } else {
                0   // 自摸牌が待ち牌でないなら聴牌（0）
            };
        }

        // 3n+2、チー・ポン後の聴牌状態。`self.shanten`は0だが、
        // 実際のシャンテン数は0または-1の可能性がある。
        //
        // 例1: 223m 55p 45sの場合、`self.shanten`は1。6sをチーすると
        // `update_shanten`が呼ばれて`self.shanten`は0になる。
        // 実際のシャンテン数も0。
        //
        // 例2: 123m 55p 45sの場合、`self.shanten`は0。6sをチーすると
        // `update_shanten`は値を0以上にクランプするため`self.shanten`は0のまま。
        // 実際のシャンテン数は-1。
        shanten::calc_all(&self.tehai, self.tehai_len_div3)
    }

    /// Can be called at both 3n+1 and 3n+2, but `self.real_time_shanten` must
    /// be >= 0 and `self.tiles_left` must be >= 4.
    ///
    /// This function is currently highly internal.
    /// シングルプレイヤー期待値テーブルを作成する。
    /// 3n+1と3n+2両方で呼び出し可能だが、`self.real_time_shanten` >= 0
    /// かつ`self.tiles_left` >= 4である必要がある。
    pub(super) fn single_player_tables(&self) -> Result<SinglePlayerTables> {
        // 最低1回のツモが必要
        ensure!(self.tiles_left >= 4, "need at least one more tsumo");

        let cur_shanten = self.real_time_shanten();
        // 和了形の場合は計算不可
        ensure!(cur_shanten >= 0, "can't calculate an agari hand");

        let mut can_discard = self.last_cans.can_discard;
        // 残りツモ数と海底判定を計算
        let (tsumos_left, calc_haitei) = if can_discard {
            // 打牌可能な場合：現在の残り牌数から計算
            (self.tiles_left / 4, self.tiles_left % 4 == 0)
        } else {
            // 鳴き後などで打牌できない場合
            let target = self.rel(self.last_cans.target_actor) as u8;
            // チャンカンは無視
            let tiles_left_at_next_tsumo = self.tiles_left.saturating_sub(4 - target);
            (
                tiles_left_at_next_tsumo / 4,
                tiles_left_at_next_tsumo % 4 == 0,
            )
        };
        ensure!(tsumos_left >= 1, "need at least one more tsumo");

        // 副露中のドラ数を計算
        let num_doras_in_fuuro = if self.is_menzen && self.ankan_overview[0].is_empty() {
            0  // 門前で暗槓もない場合
        } else {
            // 手牌中のドラ数を計算
            let num_doras_in_tehai: u8 = self
                .dora_indicators
                .iter()
                .map(|ind| self.tehai[ind.next().as_usize()])
                .sum();
            let num_akas = self.akas_in_hand.iter().filter(|&&b| b).count() as u8;
            // 総ドラ数 - 手牌中のドラ数 - 赤牌数 = 副露中のドラ数
            self.doras_owned[0] - num_doras_in_tehai - num_akas
        };
        let prefer_riichi = self.scores[0] >= 1000;  // リーチ可能か
        let calc_double_riichi = can_discard && self.can_w_riichi;  // ダブリー計算するか

        // リーチ成立後のツモ切り処理
        // リーチ後にツモ牌が待ち牌でない場合、既に捨てたものとして扱う
        let mut tehai = self.tehai;
        let mut akas_in_hand = self.akas_in_hand;
        let is_discard_after_riichi = can_discard && self.riichi_accepted[0];
        if is_discard_after_riichi {
            let last_tsumo = self.last_self_tsumo.unwrap();
            tehai[last_tsumo.deaka().as_usize()] -= 1;
            // 赤牌の処理
            match last_tsumo.as_u8() {
                tu8!(5mr) => akas_in_hand[0] = false,  // 赤5萬
                tu8!(5pr) => akas_in_hand[1] = false,  // 赤5筒
                tu8!(5sr) => akas_in_hand[2] = false,  // 赤5索
                _ => (),
            }
            can_discard = false;
        }

        // 初期状態を設定
        let init_state = InitState {
            tehai,
            akas_in_hand,
            tiles_seen: self.tiles_seen,
            akas_seen: self.akas_seen,
        };
        // シングルプレイヤー計算器を設定
        let sp_calc = SPCalculator {
            tehai_len_div3: self.tehai_len_div3,         // 手牌ブロック数
            is_menzen: self.is_menzen,                   // 門前かどうか
            chis: &self.chis,                            // チーした牌
            pons: &self.pons,                            // ポンした牌
            minkans: &self.minkans,                      // 明槓
            ankans: &self.ankans,                        // 暗槓
            bakaze: self.bakaze.as_u8(),                 // 場風
            jikaze: self.jikaze.as_u8(),                 // 自風
            num_doras_in_fuuro,                          // 副露中のドラ数
            prefer_riichi,                               // リーチを優先するか
            dora_indicators: &self.dora_indicators,      // ドラ表示牌
            calc_double_riichi,                          // ダブリー計算するか
            calc_haitei,                                 // 海底を考慮するか
            sort_result: true,                           // 結果をソートするか
            maximize_win_prob: false,                    // 和了確率最大化モード
            calc_tegawari: false,                        // 手変わりを計算するか
            calc_shanten_down: false,                    // シャンテン戻しを計算するか
        };

        // 期待値を計算
        let mut max_ev_table = sp_calc.calc(init_state, can_discard, tsumos_left, cur_shanten)?;
        // リーチ後のツモ切りの場合、最初の要素をツモ牌に設定
        if is_discard_after_riichi {
            max_ev_table[0].tile = self.last_self_tsumo.unwrap();
        }

        Ok(SinglePlayerTables { max_ev_table })
    }
}
