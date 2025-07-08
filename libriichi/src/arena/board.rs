// ゲームボードの状態管理とゲームロジックの中核実装。
// このファイルは麻雀ゲームの盤面管理、プレイヤーの行動処理、
// 勝敗判定、得点計算、特殊ルールの実装などを担当する。

use super::result::KyokuResult;                       // 局（ラウンド）結果構造体
use crate::array::Simple2DArray;                      // 2次元配列のシンプルな実装
use crate::consts::oracle_obs_shape;                  // Oracle観測データの形状定義
use crate::mjai::{Event, EventExt};                   // Mjaiフォーマットのイベント構造体
use crate::state::PlayerState;                        // プレイヤー状態管理構造体
use crate::tile::Tile;                                // 牌の型定義
use crate::vec_ops::vec_add_assign;                   // ベクトル加算ユーティリティ
use crate::{matches_tu8, must_tile, t, tu8};         // 牌関連のマクロ
use std::convert::TryInto;                            // 型変換トレイト
use std::{array, mem};                                // 標準ライブラリの配列とメモリ操作

use anyhow::{Context, Result, bail};                  // エラーハンドリング用ライブラリ
use derivative::Derivative;                            // カスタムDefault実装用マクロ
use ndarray::prelude::*;                               // 多次元配列操作ライブラリ
use rand::prelude::*;                                  // 乱数生成ユーティリティ
use rand_chacha::ChaCha12Rng;                         // ChaCha12乱数生成器（決定的乱数）
use sha3::{Digest, Sha3_256};                         // SHA3-256ハッシュ関数
use tinyvec::ArrayVec;                                 // スタックベースの可変長配列

/// ゲームボードの基本構造体。
/// 各フィールドはpubで公開されており、呼び出し側が山、ドラ、点数などを
/// 直接設定できるようになっている。
///
/// 以下の点を除いて、すべて天鳳のルールに準拠：
///
/// 1. トリプルロン（三家和）の流局はない
/// 2. 天和（役）と地和は他の役と複合せず、常に1倍役満
#[derive(Debug, Default)]
pub struct Board {
    /// 局数（東一局が0から始まる）
    pub kyoku: u8,
    /// 本場数（連続して流局または親の和了の回数）
    pub honba: u8,
    /// 供託棒数（局のシード値には影響しない）
    pub kyotaku: u8,
    /// 各プレイヤーの点数（通常は[25000; 4]から開始）
    pub scores: [i32; 4],

    /// 各プレイヤーの配牌（13枚ずつ）
    pub haipai: [[Tile; 13]; 4],
    /// 山牌（後ろから取るのでpopを使用）
    pub yama: Vec<Tile>,
    /// 嶺上牌（後ろから取るのでpopを使用）
    pub rinshan: Vec<Tile>,
    /// ドラ表示牌（後ろから取るのでpopを使用）
    pub dora_indicators: Vec<Tile>,
    /// 裏ドラ表示牌（前から参照するのでiterを使用）
    pub ura_indicators: Vec<Tile>,
}

/// ゲームの状態を管理する構造体。Boardをラップして状態管理を行う。
#[derive(Derivative)]
#[derivative(Default)]
pub struct BoardState {
    board: Board,                                      // ゲームボード本体
    // 絶対座席位置（東一局の親が常に0）
    oya: u8,                                          // 現在の親プレイヤー
    player_states: [PlayerState; 4],                 // 4人のプレイヤー状態

    can_renchan: bool,                                // 連荘可能フラグ
    has_hora: bool,                                   // 和了発生フラグ
    has_abortive_ryukyoku: bool,                      // 途中流局発生フラグ
    kyoku_deltas: [i32; 4],                          // この局の点数変動

    #[derivative(Default(value = "70"))]             // デフォルト値は70
    tiles_left: u8,                                   // 残り牌数
    tsumo_actor: u8,                                  // 次にツモするプレイヤー
    // 以下はOption<()>をboolの代わりに使用（意味的な区別のため）
    deal_from_rinshan: Option<()>,                   // 嶺上からツモするか
    need_new_dora_at_discard: Option<()>,            // 打牌時に新ドラが必要か
    need_new_dora_at_tsumo: Option<()>,              // ツモ時に新ドラが必要か
    riichi_to_be_accepted: Option<u8>,               // 受理待ちのリーチ宣言者
    #[derivative(Default(value = "[true; 4]"))]      // 全員trueで初期化
    can_nagashi_mangan: [bool; 4],                   // 流し満貫可能フラグ
    #[derivative(Default(value = "true"))]           // trueで初期化  
    can_four_wind: bool,                              // 四風連打可能フラグ
    four_wind_tile: Option<Tile>,                     // 最初に捨てられた風牌
    accepted_riichis: u8,                             // 受理されたリーチ数
    kans: u8,                                         // カンの回数
    check_four_kan: bool,                             // 四槓散了チェックフラグ
    paos: [Option<u8>; 4],                           // 包（パオ）責任者

    log: Vec<EventExt>,                               // ゲームログ

    // Oracle観測データ専用
    dora_indicators_full: Vec<Tile>,                  // 完全なドラ表示牌リスト
}

/// エージェントに渡すコンテキスト情報。
/// プレイヤー状態とゲームログへの不変参照を含む。
pub struct AgentContext<'a> {
    pub player_states: &'a [PlayerState; 4],  // 4人のプレイヤー状態への参照
    pub log: &'a [EventExt],                  // ゲームログへの参照
}

/// ポーリング結果。ゲーム継続中か終了かを示す。
#[derive(Clone, Copy)]
pub enum Poll {
    InGame,  // ゲーム継続中
    End,     // ゲーム終了
}

impl Board {
    /// シード値からボードを初期化する。
    /// 決定的乱数を使用して、同じシード値からは常に同じ配牌を生成する。
    pub fn init_from_seed(&mut self, game_seed: (u64, u64)) {
        let (nonce, key) = game_seed;                            // シード値を分解
        let kyoku_seed = Sha3_256::new()                        // SHA3-256ハッシュを使用
            .chain_update(nonce.to_le_bytes())                  // nonceをリトルエンディアンで追加
            .chain_update(key.to_le_bytes())                    // keyをリトルエンディアンで追加
            .chain_update([self.kyoku, self.honba])             // 局数と本場数を追加
            .finalize()                                          // ハッシュを確定
            .into();                                             // バイト配列に変換
        let mut rng = ChaCha12Rng::from_seed(kyoku_seed);       // ChaCha12乱数生成器を初期化
        let mut seq = UNSHUFFLED;                                // シャッフル前の牌山をコピー
        seq.shuffle(&mut rng);                                   // 牌山をシャッフル

        self.haipai = array::from_fn(|i| seq[i * 13..(i + 1) * 13].try_into().unwrap());  // 各プレイヤーに13枚ずつ配牌
        let mut idx = 13 * 4;                                    // 配牌後のインデックス

        self.rinshan = seq[idx..idx + 4].to_vec();              // 嶺上牌4枚を設定
        idx += 4;
        self.dora_indicators = seq[idx..idx + 5].to_vec();      // ドラ表示牌5枚を設定
        idx += 5;
        self.ura_indicators = seq[idx..idx + 5].to_vec();       // 裏ドラ表示牌5枚を設定
        idx += 5;
        self.yama = seq[idx..idx + 70].to_vec();                // 残り70枚を山牌とする
        idx += 70;
        assert_eq!(idx, seq.len());                              // 全136枚使い切ったことを確認
    }

    /// BoardをBoardStateに変換する。
    /// これによりゲームの状態管理が開始される。
    pub fn into_state(self) -> BoardState {
        let oya = self.kyoku % 4;                                // 親は局数を4で割った余り
        let dora_indicators_full = self.dora_indicators.clone(); // ドラ表示牌の完全コピー

        BoardState {
            board: self,                                          // Board本体を移動
            oya,                                                  // 親プレイヤーを設定
            player_states: array::from_fn(|i| PlayerState::new(i as u8)),  // 4人分のプレイヤー状態を作成
            dora_indicators_full,                                 // ドラ表示牌リストを保存
            ..Default::default()                                  // 他のフィールドはデフォルト値
        }
    }
}

impl BoardState {
    /// ボードの状態をポーリングする。
    /// いずれかのプレイヤーがアクション可能か、局が終了した場合に返る。
    pub fn poll(&mut self, mut reactions: [EventExt; 4]) -> Result<Poll> {
        loop {                                                   // メインループ
            let poll = self.step(&reactions)?;                  // 1ステップ実行
            match poll {
                Poll::InGame => {                                // ゲーム継続中の場合
                    if self.player_states.iter().any(|c| c.last_cans().can_act()) {  // アクション可能なプレイヤーがいるか
                        return Ok(poll);                         // いる場合はポーリング結果を返す
                    }
                }
                Poll::End => {                                   // ゲーム終了の場合
                    self.add_log_no_meta(Event::EndKyoku);      // 局終了イベントをログに追加
                    vec_add_assign(&mut self.board.scores, &self.kyoku_deltas);  // 点数変動を反映
                    if self.has_abortive_ryukyoku {             // 途中流局の場合
                        self.can_renchan = true;                 // 連荘を可能にする
                    }
                    return Ok(poll);                             // 終了状態を返す
                }
            };
            reactions = Default::default();                      // リアクションをクリア
        }
    }

    /// エージェント用のコンテキストを取得する。
    #[inline]
    pub fn agent_context(&self) -> AgentContext<'_> {
        AgentContext {
            player_states: &self.player_states,  // プレイヤー状態への参照
            log: &self.log,                      // ゲームログへの参照
        }
    }

    /// 局の結果を取得する。
    #[inline]
    pub const fn end(&self) -> KyokuResult {
        KyokuResult {
            kyoku: self.board.kyoku,                             // 局数
            // honba: self.board.honba,                          // 本場数（コメントアウト）
            can_renchan: self.can_renchan,                       // 連荘可能か
            has_hora: self.has_hora,                             // 和了があったか
            has_abortive_ryukyoku: self.has_abortive_ryukyoku,  // 途中流局があったか
            kyotaku_left: self.board.kyotaku,                   // 残り供託棒
            scores: self.board.scores,                           // 点数
        }
    }

    /// ログを取り出し、元のフィールドを空にする。
    #[inline]
    pub fn take_log(&mut self) -> Vec<EventExt> {
        mem::take(&mut self.log)  // ムーブして空のVecと置き換え
    }

    /// メタデータ付きイベントをログに追加する。
    #[inline]
    fn add_log(&mut self, ev: EventExt) {
        self.log.push(ev);  // そのままプッシュ
    }

    /// メタデータなしイベントをログに追加する。
    #[inline]
    fn add_log_no_meta(&mut self, ev: Event) {
        self.log.push(EventExt::no_meta(ev));  // メタデータなしでラップ
    }

    /// すべてのプレイヤーにイベントをブロードキャストする。
    #[inline]
    fn broadcast(&mut self, ev: &Event) {
        for s in &mut self.player_states {                      // 各プレイヤーに対して
            s.update(ev).expect("fatal internal bug in BoardState");  // イベントを通知（失敗は致命的バグ）
        }
    }

    /// 配牌処理を実行する。
    /// 局の開始、ドラ表示、プレイヤーへの配牌、親の第一ツモを処理する。
    fn haipai(&mut self) -> Result<()> {
        let bakaze = must_tile!(tu8!(E) + self.board.kyoku / 4);  // 場風を計算（東から始まり、4局ごとに変わる）
        let start_kyoku = Event::StartKyoku {                     // 局開始イベントを作成
            bakaze,                                                // 場風
            dora_marker: self                                      // ドラ表示牌
                .board
                .dora_indicators
                .pop()                                             // 最初のドラ表示牌を取り出す
                .context("insufficient dora indicators")?,        // ドラ表示牌が不足の場合エラー
            kyoku: self.oya + 1,                                   // 局番号（1基準）
            honba: self.board.honba,                               // 本場数
            kyotaku: self.board.kyotaku,                           // 供託棒数
            oya: self.oya,                                         // 親プレイヤー
            scores: self.board.scores,                             // 各プレイヤーの点数
            tehais: self.board.haipai,                             // 各プレイヤーの配牌
        };
        self.broadcast(&start_kyoku);                              // 全プレイヤーに通知
        self.add_log_no_meta(start_kyoku);                        // ログに記録

        let tile = self                                            // 親の第一ツモ牌
            .board
            .yama
            .pop()                                                 // 山の最後から取る
            .context("invalid yama: empty at init")?;             // 山が空の場合エラー
        self.tiles_left -= 1;                                      // 残り牌数を減らす
        let first_tsumo = Event::Tsumo {                           // 第一ツモイベント
            actor: self.oya,                                       // 親がツモる
            pai: tile,                                             // ツモ牌
        };
        self.broadcast(&first_tsumo);                              // 全プレイヤーに通知
        self.add_log_no_meta(first_tsumo);                        // ログに記録

        Ok(())
    }

    /// 通常流局（山牌がなくなった時の流局）処理を実行する。
    /// テンパイ料、流し満貫の処理を含む。
    fn exhaustive_ryukyoku(&mut self) {
        let mut deltas = [0; 4];                                   // 点数変動配列
        self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;  // 親がテンパイなら連荘

        let mut has_nagashi_mangan = false;                       // 流し満貫フラグ
        self.can_nagashi_mangan                                   // 流し満貫可能なプレイヤーを処理
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)                                 // 流し満貫可能なプレイヤーのみ
            .map(|(i, _)| i)
            .for_each(|i| {
                has_nagashi_mangan = true;                        // 流し満貫があった
                if i as u8 == self.oya {                          // 親の流し満貫
                    let mut dod = [-4000; 4];                     // 全員から4000点減点
                    dod[i] = 12000;                               // 親は12000点加点
                    vec_add_assign(&mut deltas, &dod);           // 点数変動に加算
                } else {                                          // 子の流し満貫
                    let mut dod = [-2000; 4];                     // 全員から2000点減点
                    dod[i] = 8000;                                // 流し満貫者は8000点加点
                    dod[self.oya as usize] = -4000;              // 親は追加で-2000点（計-4000点）
                    vec_add_assign(&mut deltas, &dod);           // 点数変動に加算
                };
            });

        if !has_nagashi_mangan {                                  // 流し満貫がない場合はテンパイ料
            let tenpai_actors: ArrayVec<[_; 4]> = self           // テンパイプレイヤーを収集
                .player_states
                .iter()
                .enumerate()
                .filter(|&(_, s)| s.shanten() == 0)              // シャンテン数が0（テンパイ）
                .map(|(i, _)| i)
                .collect();

            let (plus, minus) = match tenpai_actors.len() {      // テンパイ人数による点数
                1 => (3000, -1000),                               // 1人テンパイ：+3000/-1000
                2 => (1500, -1500),                               // 2人テンパイ：+1500/-1500
                3 => (1000, -3000),                               // 3人テンパイ：+1000/-3000
                // 0 | 4                                          // 0人または4人テンパイ：点数移動なし
                _ => (0, 0),
            };
            if plus > 0 {                                         // 点数移動がある場合
                let mut dod = [minus; 4];                         // 全員にマイナス点を設定
                tenpai_actors.into_iter().for_each(|i| dod[i] = plus);  // テンパイ者にプラス点
                vec_add_assign(&mut deltas, &dod);               // 点数変動に加算
            }
        }

        vec_add_assign(&mut self.kyoku_deltas, &deltas);         // 局全体の点数変動に加算
        let ryukyoku = Event::Ryukyoku {                          // 流局イベント
            deltas: Some(deltas),                                 // 点数変動を含む
        };
        self.add_log_no_meta(ryukyoku);                          // ログに記録
        // ブロードキャストは不要（局終了時の処理なので）
    }

    /// 流し満貫と四風連打の可能性を更新する。
    /// 特定のアクションがこれらの特殊役を無効にする。
    const fn update_nagashi_mangan_and_four_wind(&mut self, ev: &Event) {
        match *ev {
            Event::Dahai { actor, pai, .. } if !pai.is_yaokyuu() => {  // 九牌以外を捨てた場合
                self.can_nagashi_mangan[actor as usize] = false;       // そのプレイヤーの流し満貫不可
            }
            Event::Chi { target, .. }                                  // チー
            | Event::Pon { target, .. }                                // ポン
            | Event::Daiminkan { target, .. } => {                     // 大明カンの場合
                self.can_nagashi_mangan[target as usize] = false;      // 鳴かれたプレイヤーの流し満貫不可
                self.can_four_wind = false;                             // 四風連打も不可（第一巡が終わった）
            }
            Event::Ankan { .. } => {                                    // 暗カンの場合
                self.can_four_wind = false;                             // 四風連打不可（第一巡が終わった）
            }
            _ => (),                                                    // その他のイベントは影響なし
        };
    }

    /// 四風連打（第一巡に4人が同じ風牌を捨てる）をチェックする。
    /// 条件を満たした場合trueを返す。
    fn check_four_wind(&mut self, pai: Tile) -> Result<bool> {
        if !matches_tu8!(pai.as_u8(), E | S | W | N) {                 // 風牌以外の場合
            self.can_four_wind = false;                                 // 四風連打不可
        } else if self.player_states[self.tsumo_actor as usize].can_w_riichi() {  // まだ第一巡の場合
            if let Some(tile) = self.four_wind_tile {                  // 既に風牌が記録されている場合
                // 最初の風牌と同じか比較
                self.can_four_wind = tile == pai;                      // 同じならまだ可能性あり
            } else {
                // 最初の捨て牌が風牌の場合、
                // その風牌を記録
                self.four_wind_tile = Some(pai);                       // 捨てられた風牌を記録
            }
        } else if let Some(tile) = self.four_wind_tile {               // 第一巡が終わった直後の場合
            // 第一巡が終わった直後で、最後の捨て牌が
            // 以前と同じ風牌かチェック
            if tile == pai {
                return Ok(true);                                        // 四風連打成立
            }
            // もうチェックする必要はない
            self.can_four_wind = false;
        } else {
            bail!("unexpected state when calculating 四風連打");    // 予期しない状態
        }

        Ok(false)                                                       // 四風連打未成立
    }

    /// リーチの受理処理を実行する。
    /// リーチ棒の支払いと供託の追加を行う。
    fn check_riichi_accepted(&mut self) {
        if let Some(actor) = self.riichi_to_be_accepted.take() {       // 受理待ちのリーチがある場合
            let riichi_accepted = Event::ReachAccepted { actor };       // リーチ受理イベント
            self.broadcast(&riichi_accepted);                           // 全プレイヤーに通知
            self.add_log_no_meta(riichi_accepted);                     // ログに記録
            self.board.scores[actor as usize] -= 1000;                 // リーチ棒（1000点）を支払い
            self.board.kyotaku += 1;                                    // 供託棒を１本追加
            self.accepted_riichis += 1;                                 // 受理されたリーチ数を増やす
        }
    }

    /// 新しいドラ表示牌を追加する。
    /// カンが発生した時に呼ばれる。
    fn add_new_dora(&mut self) -> Result<()> {
        let dora = self
            .board
            .dora_indicators
            .pop()                                                      // ドラ表示牌を１枚取り出す
            .context("illegal kan: already 4 kans and this is the 5th")?;  // 5回目のカンは不可
        let dora_ev = Event::Dora { dora_marker: dora };                // ドラ表示イベント
        self.broadcast(&dora_ev);                                       // 全プレイヤーに通知
        self.add_log_no_meta(dora_ev);                                 // ログに記録

        Ok(())
    }

    /// 和了（ホーラ）処理を実行する。
    /// ツモ、ロン、ダブロン、トリプルロンの処理を含む。
    fn handle_hora(
        &mut self,
        single_actor: u8,                                // 和了者（少なくとも1人）
        single_target: u8,                               // 振り込み者（ツモの場合は和了者と同じ）
        reactions: &[EventExt; 4],                       // 全プレイヤーのリアクション
    ) -> Result<()> {
        self.has_hora = true;                            // 和了発生フラグをセット

        let is_ron = single_actor != single_target;     // ロンかツモか判定
        let mut honba_left = self.board.honba as i32;   // 残り本場数（マルチロン用にmut）
        let mut kyotaku_point = self.board.kyotaku as i32 * 1000;  // 供託点（マルチロン用にmut）
        self.board.kyotaku = 0;                          // 供託棒をクリア（honbaとは違い、必ずクリア）

        // 裏ドラを使って和了点を計算させる。
        let ura_indicators =                             // 使用する裏ドラ表示牌
            self.board.ura_indicators[..5 - self.board.dora_indicators.len()].to_vec();  // カンの数だけ裏ドラを使用
        let points = reactions                           // 各プレイヤーの和了点を計算
            .iter()
            .map(|ev| match ev.event {
                Event::Hora { actor, .. } => {           // 和了イベントの場合
                    self.can_renchan |= actor == self.oya;  // 親の和了なら連荘可能
                    let point =                          // 和了点を計算
                        self.player_states[actor as usize].agari_points(is_ron, &ura_indicators);
                    Some(point).transpose()              // Option<Result>からResult<Option>に変換
                }
                _ => Ok(None),                           // 和了以外はNone
            })
            .collect::<Result<Vec<_>>>()?;               // 結果を収集

        if is_ron {                                      // ロンの場合
            // マルチロン（ダブロン、トリプルロン）を処理
            points
                .into_iter()
                .enumerate()
                .cycle()                                 // 循環イテレータ
                .skip(single_target as usize + 1)       // 振り込み者の次から開始
                .take(3)                                 // 最大3人まで（トリプルロンまで）
                .filter_map(|(actor, v)| v.map(|point| (actor, point)))  // 和了者のみ抽出
                .for_each(|(actor, point)| {            // 各和了者に対して
                    let mut deltas = [0; 4];             // 点数変動配列
                    if let Some(pao_target) = self.paos[actor] {  // 包（パオ）がある場合
                        // 天鳳のルールより：
                        //
                        // > 複合役満を含む得点を、ツモ＝全額・ロン＝折半で支払
                        // > う。積み棒は包。
                        deltas[pao_target as usize] = -point.ron / 2 - honba_left * 300;  // 包責任者が半額+積み棒
                        deltas[single_target as usize] -= point.ron / 2;  // 振り込み者も半額（同一人物の可能性あり）
                    } else {                             // 包がない通常のロン
                        deltas[single_target as usize] = -point.ron - honba_left * 300;  // 振り込み者が全額+積み棒
                    }
                    deltas[actor] = point.ron + kyotaku_point + honba_left * 300;  // 和了者は和了点+供託+積み棒

                    kyotaku_point = 0;                   // 最初の和了者が供託を総取り
                    honba_left = 0;                      // 最初の和了者が積み棒を総取り

                    vec_add_assign(&mut self.kyoku_deltas, &deltas);  // 局の点数変動に加算
                    let ura_markers = self.player_states[actor]       // 裏ドラ表示牌
                        .self_riichi_accepted()                        // リーチしている場合
                        .then(|| ura_indicators.clone())               // 裏ドラをコピー
                        .unwrap_or_default();                          // リーチしていない場合は空

                    let hora = Event::Hora {             // 和了イベント
                        actor: actor as u8,              // 和了者
                        target: single_target,           // 振り込み者
                        deltas: Some(deltas),            // 点数変動
                        ura_markers: Some(ura_markers),  // 裏ドラ表示牌
                    };
                    self.add_log_no_meta(hora);          // ログに記録
                    // ブロードキャストは不要（局終了時の処理なので）
                });
            return Ok(());
        }

        let point = points[single_actor as usize].unwrap();           // ツモ和了者の点数
        let mut deltas = [0; 4];                                       // 点数変動配列
        if let Some(pao_target) = self.paos[single_actor as usize] {  // 包がある場合
            // 包が発生するには少なくとも1役満が必要なので、
            // ロン点数とツモ点数の合計は等しくなる。
            deltas[pao_target as usize] = -point.ron - honba_left * 300;  // 包責任者が全額支払い
        } else {                                                       // 通常のツモ
            deltas.fill(-point.tsumo_ko - honba_left * 100);          // 子の支払いを全員に設定
            if single_actor != self.oya {                             // ツモ者が子の場合
                deltas[self.oya as usize] = -point.tsumo_oya - honba_left * 100;  // 親は別途計算
            }
        };
        deltas[single_actor as usize] =                               // ツモ者の取り分
            point.tsumo_total(single_actor == self.oya) + kyotaku_point + honba_left * 300;  // ツモ点+供託+積み棒

        vec_add_assign(&mut self.kyoku_deltas, &deltas);              // 局の点数変動に加算
        let ura_markers = self.player_states[single_actor as usize]   // 裏ドラ表示牌
            .self_riichi_accepted()                                    // リーチしている場合
            .then_some(ura_indicators)                                // 裏ドラを設定
            .unwrap_or_default();                                      // リーチしていない場合は空

        let hora = Event::Hora {                                       // 和了イベント
            actor: single_actor,                                       // 和了者
            target: single_target,                                     // ツモの場合は同じ
            deltas: Some(deltas),                                      // 点数変動
            ura_markers: Some(ura_markers),                            // 裏ドラ表示牌
        };
        self.add_log_no_meta(hora);                                    // ログに記録
        // ブロードキャストは不要（局終了時の処理なので）

        Ok(())
    }

    /// 包（パオ）責任を更新する。
    /// 大三元、大四喜の確定時に、最後の面子を鳴かせたプレイヤーが責任を負う。
    fn update_paos(&mut self, ev: &Event) {
        match *ev {
            Event::Pon {                                 // ポンの場合
                target, actor, pai, ..
            }
            | Event::Daiminkan {                         // 大明カンの場合
                target, actor, pai, ..
            } if pai.is_jihai() => {                     // 字牌の場合のみ
                let mut jihais = 0_u8;                   // 字牌のビットマスク
                self.player_states[actor as usize]       // 鳴いたプレイヤーの
                    .pons()                              // ポンと
                    .iter()
                    .chain(self.player_states[actor as usize].minkans())  // 明カンを
                    .copied()
                    .filter(|&t| t >= tu8!(E))           // 字牌のみフィルタ
                    .for_each(|t| jihais |= 1 << (t - tu8!(E)));  // ビットを立てる（E=0, S=1, W=2, N=3, P=4, F=5, C=6）
                let daisanein_confirmed = (jihais & 0b1110000) == 0b1110000;  // 白発中（ビット4,5,6）が全て揃っている
                let daisuushi_confirmed = (jihais & 0b0001111) == 0b0001111;  // 東南西北（ビット0,1,2,3）が全て揃っている
                if daisanein_confirmed && matches_tu8!(pai.as_u8(), P | F | C)  // 大三元確定で三元牌を鳴いた
                    || daisuushi_confirmed && matches_tu8!(pai.as_u8(), E | S | W | N)  // 大四喜確定で風牌を鳴いた
                {
                    self.paos[actor as usize] = Some(target);  // 鳴かせたプレイヤーが包責任者
                }
            }
            _ => (),                                     // その他のイベントは無視
        }
    }

    /// 途中流局（九種九牌、四家立直、四風連打、四槓散了）処理を実行する。
    #[inline]
    fn abortive_ryukyoku(&mut self) {
        let ryukyoku = Event::Ryukyoku {                 // 流局イベント
            deltas: Some([0; 4]),                         // 点数変動なし
        };
        self.add_log_no_meta(ryukyoku);                  // ログに記録
        self.has_abortive_ryukyoku = true;                // 途中流局フラグをセット
        // ブロードキャストは不要（局終了時の処理なので）
    }

    /// ゲームの1ステップを実行するメインロジック。
    /// プレイヤーのリアクションを処理し、ゲーム状態を更新する。
    fn step(&mut self, reactions: &[EventExt; 4]) -> Result<Poll> {
        if self.tiles_left == 70 {                       // ゲーム開始時（残り70牌）
            self.haipai()?;                              // 配牌処理
            return Ok(Poll::InGame);                     // ゲーム継続
        }

        if self.accepted_riichis == 4 {                  // 4人全員がリーチした場合
            // 四家立直
            self.abortive_ryukyoku();                    // 途中流局
            return Ok(Poll::End);                        // 局終了
        }

        // リアクションの妥当性を検証
        for (actor, ev) in reactions.iter().enumerate() {                     // 各プレイヤーのリアクションに対して
            self.player_states[actor]
                .validate_reaction(&ev.event)                                  // 状態に対して妥当か検証
                .with_context(|| {                                             // エラー時のコンテキスト情報
                    format!(
                        "invalid action: {ev:?}\nstate:\n{}",
                        self.player_states[actor].brief_info(),                // プレイヤー状態の要約を含む
                    )
                })?;
        }

        let ev = reactions                               // 優先度が最も高いリアクションを選択
            .iter()
            .min_by_key(|ev| match ev.event {           // 優先度順にソート
                Event::Hora { .. } => 0,                 // 和了が最優先
                Event::Daiminkan { .. } | Event::Pon { .. } => 1,  // カン・ポンが次
                Event::None => 3,                        // 何もしないが最低優先
                _ => 2,                                  // その他（チー、リーチ等）
            })
            .unwrap();                                   // unwrapは安全（配列が空でないことが保証されている）

        if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {  // 4カン後で和了以外の場合
            // 四槓散了
            self.abortive_ryukyoku();                    // 途中流局
            return Ok(Poll::End);                        // 局終了
        }

        self.update_nagashi_mangan_and_four_wind(&ev.event);  // 流し満貫と四風連打の状態更新

        match ev.event {                                 // イベントの種類による処理分岐
            Event::None => {                             // 誰もリアクションしない場合（次のツモへ）
                if self.tiles_left == 0 {                // 山牌がなくなった場合
                    self.exhaustive_ryukyoku();          // 通常流局処理
                    return Ok(Poll::End);                // 局終了
                }
                self.check_riichi_accepted();            // リーチ受理処理

                let tile = if self.deal_from_rinshan.take().is_some() {  // 嶺上からツモする場合
                    self.board
                        .rinshan
                        .pop()                           // 嶺上牌を取る
                        .context("illegal kan: already 4 kans and this is the 5th")?  // 5回目のカンは不可
                } else {                                 // 通常のツモ
                    self.board.yama.pop().with_context(|| {  // 山からツモ
                        format!("tiles left > 0 ({}) but yama is empty", self.tiles_left)  // 山が空の場合エラー
                    })?
                };
                self.tiles_left -= 1;                    // 残り牌数を減らす
                let tsumo = Event::Tsumo {               // ツモイベント
                    actor: self.tsumo_actor,             // ツモプレイヤー
                    pai: tile,                           // ツモ牌
                };

                // これは加カン専用。槍カンは実際のツモまで可能なため。
                if self.need_new_dora_at_tsumo.take().is_some() {  // ツモ時に新ドラが必要な場合
                    self.add_new_dora()?;                // 新ドラ追加
                }

                self.broadcast(&tsumo);                  // 全プレイヤーに通知
                self.add_log_no_meta(tsumo);             // ログに記録
            }

            Event::Dahai { actor, pai, .. } => {         // 打牌イベント
                if self.need_new_dora_at_discard.take().is_some() {  // 打牌時に新ドラが必要な場合
                    self.add_new_dora()?;                // 新ドラ追加
                }

                self.broadcast(&ev.event);               // 全プレイヤーに通知
                self.add_log(ev.clone());                // メタデータ付きでログに記録
                self.tsumo_actor = (actor + 1) % 4;      // 次のツモプレイヤーを設定

                // 四風連打チェック
                if self.can_four_wind && self.check_four_wind(pai)? {  // 四風連打成立の場合
                    self.abortive_ryukyoku();            // 途中流局
                    return Ok(Poll::End);                // 局終了
                }

                if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {  // 4カンされていて、誰も4カンしていない場合
                    // 四槓散了
                    self.check_four_kan = true;          // 四槓散了チェックフラグをセット
                }
            }

            Event::Chi { .. } | Event::Pon { .. } => {  // チーまたはポンイベント
                self.check_riichi_accepted();            // リーチ受理処理（鳴きはリーチ宣言をキャンセルする）
                self.broadcast(&ev.event);               // 全プレイヤーに通知
                self.add_log(ev.clone());                // メタデータ付きでログに記録
            }

            Event::Ankan { actor, .. } => {              // 暗カンイベント
                // 連続カン用の処理
                if self.need_new_dora_at_discard.take().is_some() {  // 前のカンのドラ表示待ちの場合
                    self.add_new_dora()?;                // 新ドラ追加
                }

                self.broadcast(&ev.event);               // 全プレイヤーに通知
                self.add_log(ev.clone());                // メタデータ付きでログに記録

                // 暗カンは即座に新ドラを追加
                self.add_new_dora()?;                    // 新ドラ追加

                self.tsumo_actor = actor;                // カンしたプレイヤーが次にツモ
                self.deal_from_rinshan = Some(());       // 嶺上からツモ
                self.kans += 1;                          // カン数を増やす
            }

            Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {  // 大明カンまたは加カンイベント
                // 加カン専用、`.take()`しない
                if self.need_new_dora_at_discard.is_some() {                  // 新ドラ待ちの場合
                    self.need_new_dora_at_tsumo = Some(());                   // ツモ時に新ドラを追加するよう設定
                }

                // 大明カン専用
                self.check_riichi_accepted();            // リーチ受理処理

                self.broadcast(&ev.event);               // 全プレイヤーに通知
                self.add_log(ev.clone());                // メタデータ付きでログに記録

                self.need_new_dora_at_discard = Some(()); // 次の打牌時に新ドラを追加

                self.tsumo_actor = actor;                // カンしたプレイヤーが次にツモ
                self.deal_from_rinshan = Some(());       // 嶺上からツモ
                self.kans += 1;                          // カン数を増やす
            }

            Event::Reach { actor } => {                  // リーチ宣言イベント
                self.broadcast(&ev.event);               // 全プレイヤーに通知
                self.add_log(ev.clone());                // メタデータ付きでログに記録
                self.riichi_to_be_accepted = Some(actor); // リーチ受理待ちに設定
            }

            Event::Hora { actor, target, .. } => {       // 和了イベント
                self.handle_hora(actor, target, reactions)?;  // 和了処理を実行
                return Ok(Poll::End);                    // 局終了
            }

            Event::Ryukyoku { .. } => {                  // 流局イベント
                // 九種九牌
                self.abortive_ryukyoku();                // 途中流局処理
                return Ok(Poll::End);                    // 局終了
            }

            _ => {                                       // 予期しないイベント
                bail!("unexpected event: {:?}", ev.event);  // エラーを返す
            }
        };

        // 包（パオ）チェックは現在のイベント（ポンまたは大明カン）が
        // 処理された後に行う必要がある。`.pons()`と`.minkans()`を
        // 読む必要があるため。
        self.update_paos(&ev.event);                     // 包責任を更新

        Ok(Poll::InGame)                                 // ゲーム継続
    }

    /// Oracle用の観測データをエンコードする。
    /// AIモデルが完全情報を使って学習・推論するためのデータを生成する。
    pub fn encode_oracle_obs(&self, perspective: u8, version: u32) -> Array2<f32> {
        let shape = oracle_obs_shape(version);           // バージョンに応じた観測データの形状を取得
        let mut arr = Simple2DArray::<34, f32>::new(shape.0);  // 34列の2D配列を作成
        let mut idx = 0;                                 // 現在の行インデックス

        self.player_states                               // 他の3人のプレイヤー状態をエンコード
            .iter()
            .cycle()                                     // 循環イテレータ
            .skip(perspective as usize + 1)              // 視点プレイヤーの次から開始
            .take(3)                                     // 3人分
            .for_each(|state| {
                state                                    // 手牌情報
                    .tehai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)    // 持っている牌のみ
                    .for_each(|(tile_id, &count)| {
                        arr.assign_rows(idx, tile_id, count as usize, 1.);  // 牌の枚数分の行を1で埋める
                    });
                idx += 4;                                // 4枚分のスペース

                state                                    // 赤牌情報
                    .akas_in_hand()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &has_it)| has_it)      // 赤牌を持っている場合
                    .for_each(|(i, _)| arr.fill(idx + i, 1.));  // 対応位置を1で埋める
                idx += 3;                                // 3種類の赤牌分

                let n = state.shanten() as usize;        // シャンテン数
                match version {                          // バージョン別のエンコード
                    1 => {                               // バージョン1
                        arr.fill_rows(idx, n, 1.);       // シャンテン数分の行を1で埋める
                        idx += 6;                        // 最大6シャンテン分
                    }
                    2 | 3 | 4 => {                       // バージョン2,3,4
                        arr.fill(idx + n, 1.);           // ワンホットエンコーディング
                        idx += 7;                        // 最大7シャンテン分

                        let v = n as f32 / 6.;           // 正規化されたシャンテン数
                        arr.fill(idx, v);                // 連続値として保存
                        idx += 1;
                    }
                    _ => unreachable!(),                 // 未対応バージョン
                }

                state                                    // 待ち牌情報
                    .waits()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)                // 待ち牌のみ
                    .for_each(|(t, _)| arr.assign(idx, t, 1.));  // 待ち牌を1でマーク
                idx += 1;

                if state.at_furiten() {                  // フリテン状態
                    arr.fill(idx, 1.);                   // フリテンフラグ
                }
                idx += 1;
            });

        let mut encode_tile = |idx: usize, tile: Tile| {  // 牌をエンコードするクロージャ
            let tile_id = tile.deaka().as_usize();        // 赤牌を通常牌に変換してID取得
            arr.assign(idx, tile_id, 1.);                 // 牌種をマーク
            if tile.is_aka() {                            // 赤牌の場合
                arr.fill(idx + 1, 1.);                    // 赤牌フラグをセット
            }
        };

        self.board                                        // 山牌情報
            .yama
            .iter()
            .copied()
            .rev()                                        // 逆順（ツモ順）
            .take(self.tiles_left as usize)              // 残り牌数分
            .for_each(|tile| {
                encode_tile(idx, tile);                   // 各牌をエンコード
                idx += 2;                                 // 牌情報+赤牌フラグで2行
            });
        idx += (69 - self.tiles_left as usize) * 2;      // 使用済み山牌分をスキップ

        self.board.rinshan.iter().copied().rev().for_each(|tile| {  // 嶺上牌情報
            encode_tile(idx, tile);                       // 各牌をエンコード
            idx += 2;
        });
        idx += (4 - self.board.rinshan.len()) * 2;       // 使用済み嶺上牌分をスキップ

        self.dora_indicators_full                         // ドラ表示牌情報（全て）
            .iter()
            .copied()
            .rev()                                        // 逆順
            .for_each(|tile| {
                encode_tile(idx, tile);                   // 各牌をエンコード
                idx += 2;
            });

        self.board.ura_indicators.iter().copied().for_each(|tile| {  // 裏ドラ表示牌情報
            encode_tile(idx, tile);                       // 各牌をエンコード
            idx += 2;
        });

        assert_eq!(idx, shape.0);                         // サイズ検証
        arr.build()                                       // 2D配列を構築して返す
    }
}

/// シャッフル前の麻雀牌136枚の配列。
/// 各種牌が4枚ずつ、赤ドラは各種1枚ずつ含まれる。
#[rustfmt::skip]  // フォーマットをスキップ（見やすさのため）
const UNSHUFFLED: [Tile; 136] = [
    // 萬子（マンズ）
    t!(1m),  t!(1m), t!(1m), t!(1m),      // 1萬 x 4
    t!(2m),  t!(2m), t!(2m), t!(2m),      // 2萬 x 4
    t!(3m),  t!(3m), t!(3m), t!(3m),      // 3萬 x 4
    t!(4m),  t!(4m), t!(4m), t!(4m),      // 4萬 x 4
    t!(5mr), t!(5m), t!(5m), t!(5m),      // 5萬 x 4 (赤ドラ1枚含む)
    t!(6m),  t!(6m), t!(6m), t!(6m),      // 6萬 x 4
    t!(7m),  t!(7m), t!(7m), t!(7m),      // 7萬 x 4
    t!(8m),  t!(8m), t!(8m), t!(8m),      // 8萬 x 4
    t!(9m),  t!(9m), t!(9m), t!(9m),      // 9萬 x 4

    // 筒子（ピンズ）
    t!(1p),  t!(1p), t!(1p), t!(1p),      // 1筒 x 4
    t!(2p),  t!(2p), t!(2p), t!(2p),      // 2筒 x 4
    t!(3p),  t!(3p), t!(3p), t!(3p),      // 3筒 x 4
    t!(4p),  t!(4p), t!(4p), t!(4p),      // 4筒 x 4
    t!(5pr), t!(5p), t!(5p), t!(5p),      // 5筒 x 4 (赤ドラ1枚含む)
    t!(6p),  t!(6p), t!(6p), t!(6p),      // 6筒 x 4
    t!(7p),  t!(7p), t!(7p), t!(7p),      // 7筒 x 4
    t!(8p),  t!(8p), t!(8p), t!(8p),      // 8筒 x 4
    t!(9p),  t!(9p), t!(9p), t!(9p),      // 9筒 x 4

    // 索子（ソウズ）
    t!(1s),  t!(1s), t!(1s), t!(1s),      // 1索 x 4
    t!(2s),  t!(2s), t!(2s), t!(2s),      // 2索 x 4
    t!(3s),  t!(3s), t!(3s), t!(3s),      // 3索 x 4
    t!(4s),  t!(4s), t!(4s), t!(4s),      // 4索 x 4
    t!(5sr), t!(5s), t!(5s), t!(5s),      // 5索 x 4 (赤ドラ1枚含む)
    t!(6s),  t!(6s), t!(6s), t!(6s),      // 6索 x 4
    t!(7s),  t!(7s), t!(7s), t!(7s),      // 7索 x 4
    t!(8s),  t!(8s), t!(8s), t!(8s),      // 8索 x 4
    t!(9s),  t!(9s), t!(9s), t!(9s),      // 9索 x 4

    // 字牌
    t!(E), t!(E), t!(E), t!(E),           // 東 x 4
    t!(S), t!(S), t!(S), t!(S),           // 南 x 4
    t!(W), t!(W), t!(W), t!(W),           // 西 x 4
    t!(N), t!(N), t!(N), t!(N),           // 北 x 4
    t!(P), t!(P), t!(P), t!(P),           // 白 x 4
    t!(F), t!(F), t!(F), t!(F),           // 発 x 4
    t!(C), t!(C), t!(C), t!(C),           // 中 x 4
];
