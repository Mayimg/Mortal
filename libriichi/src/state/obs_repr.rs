//! 観測表現エンコーダー (Observation Representation Encoder)
//!
//! このモジュールは麻雀ゲームの状態を機械学習モデル用の数値表現に変換する機能を提供します。
//! PlayerStateから観測テンソル（2D配列）とアクションマスク（1D配列）を生成し、
//! ニューラルネットワークが処理できる形式にエンコードします。
//!
//! 主な機能：
//! - 手牌、河（捨て牌）、副露、点数などのゲーム状態のエンコード
//! - 利用可能なアクションのマスク生成
//! - 複数のモデルバージョン（1-4）のサポート
//! - 整数値の様々なエンコード方式（one-hot、rescale、RBF）
//! - Python バインディングを通じた機械学習フレームワークとの連携

use super::item::KawaItem;
use super::{PlayerState, SinglePlayerTables};
use crate::algo::sp::{Candidate, CandidateColumn};
use crate::array::Simple2DArray;
use crate::consts::{ACTION_SPACE, MAX_VERSION, obs_shape};
use crate::tile::Tile;
use crate::{tu8, tuz};
use std::num::NonZeroUsize;

use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

// 自分の河（捨て牌）の情報をエンコードするためのチャンネル数
// カンの牌、捨て牌、赤牌フラグ、ドラフラグで4チャンネル
const SELF_KAWA_ITEM_CHANNELS: usize = 4;

// 他家の河（捨て牌）の情報をエンコードするためのチャンネル数
// チー・ポンの消費牌(2)、カンの牌、捨て牌、赤牌フラグ、ドラフラグ、手出しフラグ、リーチ宣言牌フラグで8チャンネル
const KAWA_ITEM_CHANNELS: usize = 8;

// 最大ターン数（実際の残りツモ数の実用的な最大値）
// ゲーム終了までの最大残りターン数を表し、期待値計算などで使用
const MAX_NUM_TURNS: usize = 17; // aka the actual practical `MAX_TSUMOS_LEFT`

/// 観測エンコーディングのコンテキスト
/// エンコード処理中の状態を保持し、効率的なデータ構築を行う
struct ObsEncoderContext<'a> {
    /// エンコード対象のプレイヤー状態への参照
    state: &'a PlayerState,
    /// 観測データを構築するための2D配列（34列 × n行）
    /// 34列は麻雀牌の種類数（萬子・筒子・索子・字牌）に対応
    arr: Simple2DArray<34, f32>,
    /// 利用可能なアクションを示すマスク配列（ACTION_SPACE = 46要素）
    mask: Array1<bool>,
    /// arr配列の現在の行インデックス（エンコード位置）
    idx: usize,
    /// カン選択モードかどうか（カンする牌を選択中）
    at_kan_select: bool,
    /// エンコードバージョン（1-4、モデルによって異なる観測形式）
    version: u32,
}

/// 整数値を様々な形式でエンコードするためのユーティリティ構造体
/// one-hot、正規化、RBF（放射基底関数）など複数のエンコード方式をサポート
#[must_use]
struct IntegerEncoder {
    /// エンコードする整数値
    n: usize,
    /// 値の上限（クリッピングとエンコード範囲の決定に使用）
    cap: usize,
    /// one-hotエンコーディングを使用するかどうか
    one_hot: bool,
    /// 0-1の範囲に正規化（rescale）するかどうか
    rescale: bool,
    /// RBF（放射基底関数）エンコーディングの区間数
    /// Some(n)の場合、n個のガウス関数でエンコード
    rbf_intervals: Option<NonZeroUsize>,
}

impl IntegerEncoder {
    /// 新しいIntegerEncoderを作成
    /// n: エンコードする値、cap: 値の上限
    const fn new(n: usize, cap: usize) -> Self {
        Self {
            n,
            cap,
            one_hot: false,
            rescale: false,
            rbf_intervals: None,
        }
    }
    
    /// one-hotエンコーディングの有効/無効を設定（ビルダーパターン）
    const fn one_hot(mut self, v: bool) -> Self {
        self.one_hot = v;
        self
    }
    
    /// rescale（正規化）の有効/無効を設定（ビルダーパターン）
    const fn rescale(mut self, v: bool) -> Self {
        self.rescale = v;
        self
    }
    
    /// RBFエンコーディングの区間数を設定（ビルダーパターン）
    /// v: ガウス関数の数（3以上が必要）
    const fn rbf_intervals(mut self, v: usize) -> Self {
        self.rbf_intervals = NonZeroUsize::new(v);
        self
    }

    /// 整数値をコンテキストの配列にエンコード
    fn encode(self, ctx: &mut ObsEncoderContext<'_>) {
        // 値を上限でクリップ
        let n = self.n.min(self.cap);
        match ctx.version {
            // バージョン1: 単純なn行の1.0埋め（廃止予定の形式）
            1 => {
                ctx.arr.fill_rows(ctx.idx, n, 1.);
                ctx.idx += self.cap;
            }
            // バージョン2,3: 複数のエンコード方式をサポート
            2 | 3 => {
                // 少なくとも1つのエンコード方式が指定されていることを確認
                debug_assert!(self.one_hot || self.rescale || self.rbf_intervals.is_some());

                // one-hotエンコーディング: n番目の位置に1.0を設定
                if self.one_hot {
                    ctx.arr.fill(ctx.idx + n, 1.);
                    ctx.idx += self.cap + 1;
                }
                
                // 正規化エンコーディング: 0-1の範囲に正規化
                if self.rescale {
                    let v = n as f32 / self.cap as f32;
                    ctx.arr.fill(ctx.idx, v);
                    ctx.idx += 1;
                }

                // RBF（放射基底関数）エンコーディング: 複数のガウス関数で表現
                if let Some(intervals) = self.rbf_intervals.map(|v| v.get()) {
                    debug_assert!(intervals >= 3);
                    // 各ガウス関数の中心間の距離
                    let interval_size = self.cap as f32 / intervals as f32;
                    // 各ガウス関数について値を計算（最初と最後の中心は除く）
                    for i in 1..intervals {
                        let x = self.n as f32; // クリップ前の元の値を使用
                        let mu = i as f32 * interval_size; // ガウス関数の中心
                        let sigma = interval_size; // 標準偏差
                        // ガウス関数: exp(-(x-μ)²/(2σ²))
                        let v = (-(x - mu).powi(2) / (2. * sigma.powi(2))).exp();
                        ctx.arr.fill(ctx.idx + i - 1, v);
                    }
                    ctx.idx += intervals - 1;
                }
            }
            // バージョン4: RBFを除いたシンプルな形式
            4 => {
                debug_assert!(self.one_hot || self.rescale);

                // one-hotエンコーディング
                if self.one_hot {
                    ctx.arr.fill(ctx.idx + n, 1.);
                    ctx.idx += self.cap + 1;
                }
                
                // 正規化エンコーディング
                if self.rescale {
                    let v = n as f32 / self.cap as f32;
                    ctx.arr.fill(ctx.idx, v);
                    ctx.idx += 1;
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> ObsEncoderContext<'a> {
    /// 新しいObsEncoderContextを作成
    /// state: プレイヤーの状態
    /// version: エンコードバージョン（1-4）
    /// at_kan_select: カン選択モードかどうか
    fn new(state: &'a PlayerState, version: u32, at_kan_select: bool) -> Self {
        // バージョンが有効範囲内であることを確認
        assert!(version <= MAX_VERSION);
        // バージョンに応じた観測形状を取得
        let shape = obs_shape(version);
        // 観測データ用の2D配列を初期化（shape.0は行数）
        let arr = Simple2DArray::new(shape.0);
        // アクションマスクを初期化（全てfalse）
        let mask = Array1::default(ACTION_SPACE);
        Self {
            state,
            arr,
            mask,
            idx: 0,
            at_kan_select,
            version,
        }
    }

    /// プレイヤー状態を観測テンソルとアクションマスクにエンコード
    fn encode_obs(mut self) -> (Array2<f32>, Array1<bool>) {
        let state = self.state;
        // 最後に計算された利用可能なアクション
        let cans = state.last_cans;

        // === 手牌のエンコード（4チャンネル） ===
        // 各牌について、持っている枚数分の行に1.0を設定
        state
            .tehai
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .for_each(|(tile_id, &count)| {
                let n = count as usize;
                // tile_id列のidx行からn行分を1.0で埋める
                self.arr.assign_rows(self.idx, tile_id, n, 1.);
            });
        self.idx += 4; // 手牌は最大4枚なので4行進める

        // === 赤牌の有無（3チャンネル） ===
        // 赤五萬、赤五筒、赤五索の保有状況
        state
            .akas_in_hand
            .into_iter()
            .enumerate()
            .filter(|&(_, has_it)| has_it)
            .for_each(|(i, _)| self.arr.fill(self.idx + i, 1.));
        self.idx += 3;

        // === 各プレイヤーの点数 ===
        for &score in &state.scores {
            // 0-100,000点の範囲に正規化（第1チャンネル）
            let v = score.clamp(0, 100_000) as f32 / 100_000.;
            self.arr.fill(self.idx, v);
            self.idx += 1;

            match self.version {
                // バージョン2,3: RBFエンコーディングを追加
                2 | 3 => IntegerEncoder::new(score as usize / 100, 500)
                    .rbf_intervals(10)
                    .encode(&mut self),
                // バージョン4: 0-30,000点の範囲で追加の正規化チャンネル
                4 => {
                    let v = score.clamp(0, 30_000) as f32 / 30_000.;
                    self.arr.fill(self.idx, v);
                    self.idx += 1;
                }
                _ => (),
            }
        }

        // === 順位（4チャンネル、one-hot） ===
        // 0-3の順位をone-hotエンコード
        let n = state.rank as usize;
        self.arr.fill(self.idx + n, 1.);
        self.idx += 4;

        // === 局数（4チャンネル、one-hot） ===
        // 東1局(0)から北4局(15)までの局数
        let n = state.kyoku as usize;
        match self.version {
            // v1はバグがあり、実際には3チャンネルしか使わない
            1 => self.arr.fill_rows(self.idx, n, 1.),
            2 | 3 | 4 => self.arr.fill(self.idx + n, 1.),
            _ => unreachable!(),
        }
        self.idx += 4;

        // === 本場数と供託のエンコード ===
        // バージョンによって上限値が異なる
        let cap = match self.version {
            1 | 4 => 10,
            2 | 3 => 6,
            _ => unreachable!(),
        };
        
        // 本場数（連続和了回数）
        let n = state.honba as usize;
        IntegerEncoder::new(n, cap)
            .rescale(self.version == 4) // v4では正規化も追加
            .rbf_intervals(3) // v2,3ではRBF使用
            .encode(&mut self);
        
        // 供託数（リーチ棒の数）
        let n = state.kyotaku as usize;
        IntegerEncoder::new(n, cap)
            .rescale(self.version == 4)
            .rbf_intervals(3)
            .encode(&mut self);

        // === 場風と自風（2チャンネル） ===
        // 東(0)南(1)西(2)北(3)をエンコード
        self.arr.assign(self.idx, state.bakaze.as_usize(), 1.);
        self.arr.assign(self.idx + 1, state.jikaze.as_usize(), 1.);
        self.idx += 2;

        // === ゲーム進行度（v2,3,4のみ） ===
        // 東場か南場かと局数を組み合わせた進行度指標
        if matches!(self.version, 2 | 3 | 4) {
            // 東場:0-3、南場:4-7の範囲で進行度を計算
            let n = (state.bakaze.as_u8() - tu8!(E)).min(1) * 4 + state.kyoku;
            IntegerEncoder::new(n as usize, 7)
                .rescale(true)
                .encode(&mut self);
        }

        // === ドラ表示牌（7チャンネル） ===
        self.encode_tile_set(state.dora_indicators);

        // === 自分の河（捨て牌）- 最初の6枚 ===
        // ゲーム序盤の捨て牌パターンを捕捉
        state.kawa[0]
            .iter()
            .take(6)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        // 6枚に満たない場合は空のチャンネルで埋める
        self.idx += (6 - state.kawa[0].len().min(6)) * SELF_KAWA_ITEM_CHANNELS;

        // === 自分の河 - 最新18枚（逆順） ===
        // 最近の捨て牌パターンを捕捉
        state.kawa[0]
            .iter()
            .rev()
            .take(18)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        // 18枚に満たない場合は空のチャンネルで埋める
        self.idx += (18 - state.kawa[0].len().min(18)) * SELF_KAWA_ITEM_CHANNELS;

        // 全プレイヤーの河の最大長を取得
        let max_kawa_len = state.kawa.iter().map(|k| k.len()).max().unwrap();
        
        // === 自分の河の時系列エンコード（v3,4のみ） ===
        // 新しい捨て牌ほど高い値を持つ指数減衰エンコード
        if matches!(self.version, 3 | 4) {
            for (turn, kawa_item) in state.kawa[0].iter().enumerate() {
                if let Some(kawa_item) = kawa_item {
                    let sutehai = kawa_item.sutehai;
                    let tid = sutehai.tile.deaka().as_usize();
                    // 最新の牌ほど1.0に近く、古い牌ほど0に近づく
                    let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                    self.arr.assign(self.idx, tid, v);
                }
            }
            self.idx += 1;
        }

        // === 他家の河（3人分） ===
        for player_kawa in &state.kawa[1..] {
            // 最初の6枚
            player_kawa
                .iter()
                .take(6)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            self.idx += (6 - player_kawa.len().min(6)) * KAWA_ITEM_CHANNELS;

            // 最新18枚（逆順）
            player_kawa
                .iter()
                .rev()
                .take(18)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            self.idx += (18 - player_kawa.len().min(18)) * KAWA_ITEM_CHANNELS;

            match self.version {
                // v2: 河を3つの時期に分けてエンコード（序盤・中盤・終盤）
                2 => {
                    for (turn, kawa_item) in player_kawa.iter().flatten().enumerate() {
                        // 6巡ごとに行を分ける（最大3行）
                        let row = (turn / 6).min(2);
                        let tid = kawa_item.sutehai.tile.deaka().as_usize();
                        // 捨て牌の存在
                        self.arr.assign(self.idx + row, tid, 1.);
                        // 手出し牌かどうか（ツモ切りでない）
                        if kawa_item.sutehai.is_tedashi {
                            self.arr.assign(self.idx + 3 + row, tid, 1.);
                        }
                    }
                    self.idx += 6;
                }
                // v3,4: 時系列エンコード（3チャンネル）
                3 | 4 => {
                    for (turn, kawa_item) in player_kawa.iter().enumerate() {
                        if let Some(kawa_item) = kawa_item {
                            let sutehai = kawa_item.sutehai;
                            let tid = sutehai.tile.deaka().as_usize();
                            // 指数減衰による時系列重み
                            let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                            // チャンネル0: 捨て牌
                            self.arr.assign(self.idx, tid, v);
                            // チャンネル1: 手出し牌
                            if sutehai.is_tedashi {
                                self.arr.assign(self.idx + 1, tid, v);
                            }
                            // チャンネル2: リーチ宣言牌
                            if sutehai.is_riichi {
                                self.arr.assign(self.idx + 2, tid, v);
                            }
                        }
                    }
                    self.idx += 3;
                }
                _ => (),
            }
        }

        // === 残り牌数（1チャンネル） ===
        // 最大69枚（壁牌70枚-王牌1枚）に対する比率
        let v = state.tiles_left as f32 / 69.;
        self.arr.fill(self.idx, v);
        self.idx += 1;

        // === 各プレイヤーの保有ドラ数（4プレイヤー分） ===
        for count in state.doras_owned {
            IntegerEncoder::new(count as usize, 12) // 最大12枚（理論値）
                .rescale(true)
                .rbf_intervals(3)
                .encode(&mut self);
        }

        // === 見えていないドラの枚数 ===
        // ドラ表示牌の枚数×4 + 赤ドラ3枚 - 既に見えているドラ数
        let doras_unseen = state.dora_indicators.len() as u8 * 4 + 3 - state.doras_seen;
        IntegerEncoder::new(doras_unseen as usize, 5 * 4 + 3) // 最大23枚
            .rescale(true)
            .rbf_intervals(4)
            .encode(&mut self);

        // === 各プレイヤーの河の概要（見えている全ての捨て牌） ===
        for player_kawa_overview in &state.kawa_overview {
            self.encode_tile_set(player_kawa_overview.iter().copied());
        }

        // === 各プレイヤーの副露（チー・ポン・カン） ===
        for player_fuuro in &state.fuuro_overview {
            // 各副露セット（最大4つ）
            for f in player_fuuro {
                for tile in f {
                    let tile_id = tile.deaka().as_usize();
                    // 同じ牌が複数枚ある場合のために空いているチャンネルを探す
                    let i = (0..4)
                        .find(|&i| self.arr.get(self.idx + i, tile_id) == 0.)
                        .unwrap();
                    self.arr.assign(self.idx + i, tile_id, 1.);
                    // 天鳳ルールでは1つの副露に複数の赤牌は含まれないため、
                    // 赤牌フラグは1チャンネルのみ使用
                    if tile.is_aka() {
                        self.arr.fill(self.idx + 4, 1.);
                    }
                }
                self.idx += 5;
            }
            // 4セットに満たない場合は空のチャンネルで埋める
            self.idx += (4 - player_fuuro.len()) * 5;
        }

        // === 各プレイヤーの暗槓（4プレイヤー分） ===
        // 暗槓された牌の種類をエンコード
        for player_ankan in &state.ankan_overview {
            for tile in player_ankan {
                let tile_id = tile.as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }
            self.idx += 1;
        }

        if matches!(self.version, 2 | 3 | 4) {
            // === 見えている牌の枚数（1チャンネル） ===
            // 各牌種について見えている枚数の比率（0-1）
            for (tid, count) in state.tiles_seen.iter().copied().enumerate() {
                self.arr.assign(self.idx, tid, count as f32 / 4.);
            }
            self.idx += 1;

            // === 他家の最後の手出し牌（3プレイヤー×3チャンネル） ===
            for &player_last_tedashi in &state.last_tedashis[1..] {
                if let Some(sutehai) = player_last_tedashi {
                    let tile = sutehai.tile;
                    let tile_id = tile.deaka().as_usize();

                    // チャンネル0: 牌の種類
                    self.arr.assign(self.idx, tile_id, 1.);
                    // チャンネル1: 赤牌フラグ
                    if tile.is_aka() {
                        self.arr.fill(self.idx + 1, 1.);
                    }
                    // チャンネル2: ドラフラグ
                    if sutehai.is_dora {
                        self.arr.fill(self.idx + 2, 1.);
                    }
                }
                self.idx += 3;
            }
            
            // === 他家のリーチ宣言牌（3プレイヤー×3チャンネル） ===
            for &player_riichi_sutehai in &state.riichi_sutehais[1..] {
                if let Some(sutehai) = player_riichi_sutehai {
                    let tile = sutehai.tile;
                    let tile_id = tile.deaka().as_usize();

                    // チャンネル0: 牌の種類
                    self.arr.assign(self.idx, tile_id, 1.);
                    // チャンネル1: 赤牌フラグ
                    if tile.is_aka() {
                        self.arr.fill(self.idx + 1, 1.);
                    }
                    // チャンネル2: ドラフラグ
                    if sutehai.is_dora {
                        self.arr.fill(self.idx + 2, 1.);
                    }
                }
                self.idx += 3;
            }
        }

        // === 他家のリーチ宣言状態（3チャンネル） ===
        state.riichi_declared[1..]
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .for_each(|(i, _)| self.arr.fill(self.idx + i, 1.));
        self.idx += 3;
        
        // === 他家のリーチ成立状態（3チャンネル） ===
        // リーチ宣言後、1巡経過して成立した状態
        state.riichi_accepted[1..]
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .for_each(|(i, _)| self.arr.fill(self.idx + i, 1.));
        self.idx += 3;

        // === 待ち牌（1チャンネル） ===
        // テンパイ時の待ち牌をエンコード
        state
            .waits
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .for_each(|(t, _)| self.arr.assign(self.idx, t, 1.));
        self.idx += 1;

        // === フリテン状態（1チャンネル） ===
        if state.at_furiten {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // === 向聴数（7チャンネル、one-hot） ===
        // 0（テンパイ）から6向聴までをone-hotエンコード
        let n = state.shanten as usize;
        IntegerEncoder::new(n, 6).one_hot(true).encode(&mut self);

        // === 自分のリーチ成立状態（1チャンネル） ===
        if state.riichi_accepted[0] {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // === カン選択モード（1チャンネル） ===
        // カンする牌を選択する特殊なモード
        if self.at_kan_select {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        // === 反応可能な牌の情報（3チャンネル） ===
        // チー・ポン・カン・ロンの対象となる牌
        if cans.can_pass() {
            let tile = state
                .last_kawa_tile
                .expect("building chi/pon/daiminkan/ron feature without any kawa tile");
            let tile_id = tile.deaka().as_usize();

            // チャンネル0: 牌の種類
            self.arr.assign(self.idx, tile_id, 1.);
            // チャンネル1: 赤牌フラグ
            if tile.is_aka() {
                self.arr.fill(self.idx + 1, 1.);
            }
            // チャンネル2: ドラかどうか
            if state.dora_factor[tile.deaka().as_usize()] > 0 {
                self.arr.fill(self.idx + 2, 1.);
            }

            // パスアクションのマスク設定
            if !self.at_kan_select {
                self.mask[ACTION_SPACE - 1] = true; // 通常のパス
            } else if cans.can_daiminkan {
                self.mask[tile_id] = true; // 大明槓の牌選択
            }
        }
        self.idx += 3;

        // === 打牌候補と戦略情報（5チャンネル） ===
        if cans.can_discard {
            // チャンネル0: 打牌可能な牌
            state
                .discard_candidates_aka()
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| {
                    // 赤牌の場合は通常の5に変換してエンコード
                    let deaka_t = match t as u8 {
                        tu8!(5mr) => tuz!(5m),
                        tu8!(5pr) => tuz!(5p),
                        tu8!(5sr) => tuz!(5s),
                        _ => t,
                    };
                    self.arr.assign(self.idx, deaka_t, 1.);
                    // カン選択モードでない場合はアクションマスクを設定
                    if !self.at_kan_select {
                        self.mask[t] = true;
                    }
                });

            // チャンネル1: 向聴数維持する打牌
            state
                .keep_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 1, t, 1.));
            
            // チャンネル2: 向聴数が進む打牌
            state
                .next_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 2, t, 1.));

            // チャンネル3: 無条件でテンパイする打牌（1向聴以下の場合のみ）
            if state.shanten <= 1 {
                state
                    .discard_candidates_with_unconditional_tenpai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)
                    .for_each(|(t, _)| self.arr.assign(self.idx + 3, t, 1.));
            }

            // チャンネル4: リーチ宣言中フラグ
            if state.riichi_declared[0] {
                self.arr.fill(self.idx + 4, 1.);
            }
        }
        self.idx += 5;

        // === リーチ可能フラグ（1チャンネル） ===
        if cans.can_riichi {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[37] = true; // リーチアクション
            }
        }
        self.idx += 1;

        // === チー可能フラグ（3チャンネル） ===
        // チー下（例：1-2で3を取る）
        if cans.can_chi_low {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[38] = true;
            }
        }
        // チー中（例：1-3で2を取る）
        if cans.can_chi_mid {
            self.arr.fill(self.idx + 1, 1.);
            if !self.at_kan_select {
                self.mask[39] = true;
            }
        }
        // チー上（例：2-3で1を取る）
        if cans.can_chi_high {
            self.arr.fill(self.idx + 2, 1.);
            if !self.at_kan_select {
                self.mask[40] = true;
            }
        }
        self.idx += 3;

        // === ポン可能フラグ（1チャンネル） ===
        if cans.can_pon {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[41] = true;
            }
        }
        self.idx += 1;

        // === 大明槓可能フラグ（1チャンネル） ===
        if cans.can_daiminkan {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[42] = true; // カン決定アクション
            }
        }
        self.idx += 1;

        // === 暗槓候補（1チャンネル） ===
        if cans.can_ankan {
            // 暗槓可能な牌をエンコード
            for tile in state.ankan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true; // カン牌選択
                }
            }
            if !self.at_kan_select {
                self.mask[42] = true; // カン決定アクション
            }
        }
        self.idx += 1;

        // === 加槓候補（1チャンネル） ===
        if cans.can_kakan {
            // 加槓（ポンに追加）可能な牌をエンコード
            for tile in state.kakan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true; // カン牌選択
                }
            }
            if !self.at_kan_select {
                self.mask[42] = true; // カン決定アクション
            }
        }
        self.idx += 1;

        // === 和了可能フラグ（1チャンネル） ===
        // ツモ和了またはロン和了が可能
        if cans.can_agari() {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[43] = true;
            }
        }
        self.idx += 1;

        // === 流局可能フラグ（1チャンネル） ===
        // 九種九牌などの流局
        if cans.can_ryukyoku {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[44] = true;
            }
        }
        self.idx += 1;

        // === バージョン4の期待値関連情報 ===
        if self.version == 4 {
            // シングルプレイヤーテーブル（期待値計算結果）が利用可能な場合
            if let Ok(SinglePlayerTables { max_ev_table }) = state.single_player_tables() {
                // 最大期待値を取得
                // max_ev_tableは既に期待値でソートされているため、
                // 最初の要素が最大期待値を持つ
                let max_ev = max_ev_table
                    .first()
                    .and_then(|c| c.exp_values.first().copied())
                    .unwrap_or_default();
                self.encode_ev(max_ev);

                // === 有効牌（必要牌）のエンコード ===
                if cans.can_discard {
                    // 各打牌候補について、その牌を打った後の有効牌をエンコード
                    for candidate in &max_ev_table {
                        let discard_tid = candidate.tile.deaka().as_usize();
                        for r in &candidate.required_tiles {
                            let required_tid = r.tile.deaka().as_usize();
                            if candidate.shanten_down {
                                // 向聴数が進む場合は34列目以降に記録
                                self.arr
                                    .assign(self.idx + 34 + discard_tid, required_tid, 1.);
                            } else {
                                // 向聴数維持の場合は0-33列に記録
                                self.arr.assign(self.idx + discard_tid, required_tid, 1.);
                            }
                        }
                    }
                    self.idx += 2 * 34;

                    // 有効牌数が最大となる打牌候補を記録
                    let max_required_tiles_tid = max_ev_table
                        .iter()
                        .max_by(|l, r| l.cmp(r, CandidateColumn::NotShantenDown))
                        .unwrap()
                        .tile
                        .deaka()
                        .as_usize();
                    self.arr.assign(self.idx, max_required_tiles_tid, 1.);
                    self.idx += 2;
                } else {
                    // 打牌不可の場合（ツモ切りなど）
                    self.idx += 2 * 34 + 1;
                    // 現在の有効牌をエンコード
                    for r in &max_ev_table[0].required_tiles {
                        let required_tid = r.tile.deaka().as_usize();
                        self.arr.assign(self.idx, required_tid, 1.);
                    }
                    self.idx += 1;
                }

                // 期待値のスケーリング係数を計算
                let ev_scale = if max_ev < 1. { 0. } else { 1. / max_ev };
                // 詳細な期待値テーブルをエンコード
                self.encode_sp_table(max_ev_table, cans.can_discard, ev_scale);
            } else {
                // 期待値テーブルが利用できない場合（テンパイ時など）
                // 最小ツモ和了点数を期待値として使用（裏ドラなしを仮定）
                let min_tsumo_agari = state
                    .agari_points(cans.can_ron_agari, &[])
                    .map(|p| p.tsumo_total(state.is_oya()) as f32)
                    .unwrap_or_default();
                self.encode_ev(min_tsumo_agari);

                // 残りの期待値関連チャンネルをスキップ
                self.idx += 2 * 34 + 2 + 3 * MAX_NUM_TURNS;
            }
        }

        // エンコードが完了したことを確認
        assert_eq!(self.idx, self.arr.rows());
        // 2D配列を構築
        let arr = self.arr.build();
        // 全ての値が0-1の範囲内であることを確認（デバッグビルドのみ）
        debug_assert!(arr.iter().all(|&v| (0. ..=1.).contains(&v)));
        (arr, self.mask)
    }

    /// 期待値を2つの異なるスケールでエンコード
    fn encode_ev(&mut self, value: f32) {
        // 0-100,000点スケール（全体的な期待値）
        let v = value.clamp(0., 100_000.) / 100_000.;
        self.arr.fill(self.idx, v);
        // 0-30,000点スケール（一般的な和了点数範囲）
        let v = value.clamp(0., 30_000.) / 30_000.;
        self.arr.fill(self.idx + 1, v);
        self.idx += 2;
    }

    /// シングルプレイヤーテーブル（期待値・確率情報）をエンコード
    /// 打牌テーブル: 3 * MAX_NUM_TURNS チャンネル
    /// ツモテーブル: 3 * MAX_NUM_TURNS チャンネル
    /// 最高期待値打牌: 1 チャンネル
    /// 最高和了確率打牌: 1 チャンネル
    fn encode_sp_table(&mut self, candidates: Vec<Candidate>, can_discard: bool, ev_scale: f32) {
        // テンパイ確率が計算されている候補を確認
        let Some(first) = candidates
            .first()
            .filter(|c| c.tenpai_probs.first().is_some_and(|&p| p > 0.))
        else {
            // 向聴数が4以上の場合や確率が全て0の場合はスキップ
            self.idx += 3 * MAX_NUM_TURNS;
            return;
        };

        if can_discard {
            // 打牌可能な場合：各候補についてターンごとの確率・期待値をエンコード
            for candidate in candidates {
                let tid = candidate.tile.deaka().as_usize();
                // 各将来ターンについて
                for (turn, ((&tenpai_prob, &win_prob), &ev)) in candidate
                    .tenpai_probs
                    .iter()
                    .take_while(|&&p| p > 0.) // 確率が0になったら終了
                    .zip(&candidate.win_probs)
                    .zip(&candidate.exp_values)
                    .enumerate()
                {
                    let mut idx = self.idx + turn;
                    // テンパイ確率
                    self.arr.assign(idx, tid, tenpai_prob);
                    idx += MAX_NUM_TURNS;
                    // 和了確率
                    self.arr.assign(idx, tid, win_prob);
                    idx += MAX_NUM_TURNS;
                    // 期待値（スケール済み）
                    self.arr.assign(idx, tid, (ev * ev_scale).min(1.));
                }
            }
        } else {
            // 打牌不可の場合：現在の状態の確率・期待値をエンコード
            for (turn, ((&tenpai_prob, &win_prob), &ev)) in first
                .tenpai_probs
                .iter()
                .take_while(|&&p| p > 0.)
                .zip(&first.win_probs)
                .zip(&first.exp_values)
                .enumerate()
            {
                let mut idx = self.idx + turn;
                self.arr.fill(idx, tenpai_prob);
                idx += MAX_NUM_TURNS;
                self.arr.fill(idx, win_prob);
                idx += MAX_NUM_TURNS;
                self.arr.fill(idx, (ev * ev_scale).min(1.));
            }
        }
        self.idx += 3 * MAX_NUM_TURNS;
    }

    /// 牌のセット（複数枚）を7チャンネルでエンコード
    /// チャンネル0-3: 各牌の枚数（最大4枚）
    /// チャンネル4-6: 赤牌フラグ（赤五萬、赤五筒、赤五索）
    fn encode_tile_set<I>(&mut self, tiles: I)
    where
        I: IntoIterator<Item = Tile>,
    {
        let mut counts = [0; 34];
        for tile in tiles {
            let tile_id = tile.deaka().as_usize();

            // 同じ牌の枚数分、異なるチャンネルに記録
            let i = &mut counts[tile_id];
            self.arr.assign(self.idx + *i, tile_id, 1.);
            *i += 1;

            // 赤牌の場合は専用チャンネルにフラグを立てる
            if tile.is_aka() {
                let i = tile.as_usize() - tuz!(5mr); // 0:赤五萬, 1:赤五筒, 2:赤五索
                self.arr.fill(self.idx + 4 + i, 1.);
            }
        }
        self.idx += 7;
    }

    /// 自分の河の1つのアイテムをエンコード（4チャンネル）
    fn encode_self_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            // チャンネル0: カンした牌（大明槓・加槓の場合赤牌の可能性あり）
            for kan in k.kan {
                // deakaが必要：大明槓や加槓では赤牌の可能性があるため
                let tile_id = kan.deaka().as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.deaka().as_usize();
            // チャンネル1: 捨て牌
            self.arr.assign(self.idx + 1, tile_id, 1.);
            // チャンネル2: 赤牌フラグ
            if sutehai.tile.is_aka() {
                self.arr.fill(self.idx + 2, 1.);
            }
            // チャンネル3: ドラフラグ
            if sutehai.is_dora {
                self.arr.fill(self.idx + 3, 1.);
            }
        }
        self.idx += SELF_KAWA_ITEM_CHANNELS;
    }

    /// 他家の河の1つのアイテムをエンコード（8チャンネル）
    fn encode_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            // チャンネル0-1: チー・ポンで消費された牌（最小値・最大値）
            if let Some(cp) = &k.chi_pon {
                // チー・ポンの赤牌情報は河の詳細にはエンコードされず、
                // fuuro_overviewに含まれる
                //
                // one-hot形式でエンコード
                let a = cp.consumed[0].deaka().as_usize();
                let b = cp.consumed[1].deaka().as_usize();
                let min = a.min(b);
                let max = a.max(b);
                self.arr.assign(self.idx, min, 1.);
                self.arr.assign(self.idx + 1, max, 1.);
            }

            // チャンネル2: カンした牌
            for kan in k.kan {
                let tile_id = kan.deaka().as_usize();
                self.arr.assign(self.idx + 2, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.deaka().as_usize();
            // チャンネル3: 捨て牌
            self.arr.assign(self.idx + 3, tile_id, 1.);
            // チャンネル4: 赤牌フラグ
            if sutehai.tile.is_aka() {
                self.arr.fill(self.idx + 4, 1.);
            }
            // チャンネル5: ドラフラグ
            if sutehai.is_dora {
                self.arr.fill(self.idx + 5, 1.);
            }
            // チャンネル6: 手出しフラグ（ツモ切りでない）
            if sutehai.is_tedashi {
                self.arr.fill(self.idx + 6, 1.);
            }
            // チャンネル7: リーチ宣言牌フラグ
            if sutehai.is_riichi {
                self.arr.fill(self.idx + 7, 1.);
            }
        }
        self.idx += KAWA_ITEM_CHANNELS;
    }
}

/// Python バインディング
#[pymethods]
impl PlayerState {
    /// Pythonから呼び出し可能なencode_obsメソッド
    /// 返り値: (観測テンソル, アクションマスク)
    #[pyo3(name = "encode_obs")]
    fn encode_obs_py<'py>(
        &self,
        version: u32,
        at_kan_select: bool,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>) {
        // Rustの配列をエンコード
        let (obs, mask) = self.encode_obs(version, at_kan_select);
        // NumPy配列に変換してPythonに返す
        let obs = PyArray2::from_owned_array(py, obs);
        let mask = PyArray1::from_owned_array(py, mask);
        (obs, mask)
    }
}

impl PlayerState {
    /// プレイヤー状態を観測テンソルとアクションマスクにエンコード
    /// 
    /// # 引数
    /// * `version` - エンコードバージョン（1-4）
    /// * `at_kan_select` - カン選択モードかどうか
    /// 
    /// # 返り値
    /// * `Array2<f32>` - 観測テンソル（shape: (チャンネル数, 34)）
    /// * `Array1<bool>` - アクションマスク（shape: (46,)）
    #[must_use]
    pub fn encode_obs(&self, version: u32, at_kan_select: bool) -> (Array2<f32>, Array1<bool>) {
        ObsEncoderContext::new(self, version, at_kan_select).encode_obs()
    }
}
