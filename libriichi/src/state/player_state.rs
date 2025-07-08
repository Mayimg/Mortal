//! # PlayerState モジュール
//!
//! このファイルは麻雀ライブラリの中核となる`PlayerState`構造体を定義しています。
//! `PlayerState`は特定の座席（プレイヤー）の視点から観測可能な全てのゲーム状態情報を保持し、
//! そのプレイヤーが実行可能な合法手（アクション）を識別する機能を提供します。
//!
//! ## 主な機能
//! - ゲーム状態の管理（手牌、捨て牌、副露、点数など）
//! - 合法手の判定（ツモ、ロン、チー、ポン、カン、リーチなど）
//! - シャンテン数の計算と待ち牌の判定
//! - フリテンの管理
//! - 深層学習モデル用の特徴量エンコーディング
//! - Python バインディング（PyO3経由）
//!
//! ## 設計上の特徴
//! - 牌は0-36の整数で表現（赤ドラは別IDとして管理）
//! - 配列は基本的に34要素（通常牌のみ）で、赤ドラは別フィールドで管理
//! - 相対的な視点：scores[0]は常に自分の点数、oya=0は自分が親
//! - mjaiイベントを受け取って状態を更新し、可能なアクションを返す

// アクション候補を表す構造体をインポート（チー、ポン、カン、ツモ、ロンなどの可否を保持）
use super::action::ActionCandidate;
// 河（捨て牌）に関連する構造体をインポート（副露情報、捨て牌情報など）
use super::item::{ChiPon, KawaItem, Sutehai};
// 単独プレイヤー向けの牌評価候補を表す構造体をインポート
use crate::algo::sp::Candidate;
// 手牌を人間が読める文字列に変換する関数をインポート
use crate::hand::tiles_to_string;
// 牌IDから牌オブジェクトを作成するマクロをインポート
use crate::must_tile;
// 牌を表す基本的な型をインポート
use crate::tile::Tile;
// 標準ライブラリのイテレータをインポート（河の表示用）
use std::iter;

// エラーハンドリング用のResult型をインポート
use anyhow::Result;
// 構造体のデフォルト値を柔軟に設定するためのマクロをインポート
use derivative::Derivative;
// Python バインディング用のマクロと型をインポート
use pyo3::prelude::*;
// JSON パース用のライブラリをインポート（mjaiイベントのパース用）
use serde_json as json;
// 小さな配列用の効率的なベクター型をインポート（スタック割り当て可能）
use tinyvec::{ArrayVec, TinyVec};

/// `PlayerState` is the core of the lib, which holds all the observable game
/// state information from a specific seat's perspective with the ability to
/// identify the legal actions the specified player can make upon an incoming
/// mjai event, along with some helper functions to build an actual agent.
/// Notably, `PlayerState` encodes observation features into numpy arrays which
/// serve as inputs for deep learning model.
#[pyclass]  // PyO3マクロ：この構造体をPythonクラスとして公開
#[derive(Clone, Derivative)]  // Clone トレイトを自動導出、Derivative で高度なデフォルト値設定
#[derivative(Default)]  // Default トレイトを導出（各フィールドのデフォルト値を個別に指定可能）
pub struct PlayerState {
    // プレイヤーID（0-3）。視点となるプレイヤーの座席番号
    pub(super) player_id: u8,

    /// Does not include aka.
    // 手牌の枚数を牌種別に保持する配列（赤ドラは含まない）
    // インデックス: 0-8:萬子(1-9), 9-17:筒子(1-9), 18-26:索子(1-9), 27-33:字牌(東南西北白發中)
    #[derivative(Default(value = "[0; 34]"))]  // デフォルト値：全て0（牌なし）
    pub(super) tehai: [u8; 34],

    /// Does not consider yakunashi, but does consider other kinds of
    /// furiten.
    // 待ち牌を表すフラグ配列（役なしは考慮しないが、フリテンは考慮する）
    // true の位置が現在の手牌で和了可能な牌を示す
    #[derivative(Default(value = "[false; 34]"))]  // デフォルト値：全てfalse（待ちなし）
    pub(super) waits: [bool; 34],

    // 各牌種がドラとして何枚カウントされるかを保持する配列
    // 例：ドラ表示牌が2萬なら、3萬の位置に1が入る。カンドラで複数枚ある場合は加算される
    #[derivative(Default(value = "[0; 34]"))]  // デフォルト値：全て0（ドラなし）
    pub(super) dora_factor: [u8; 34],

    /// For calculating `waits` and `doras_seen`, also for SPCalculator.
    // プレイヤーが見えている牌の枚数（手牌、捨て牌、副露牌、ドラ表示牌など全て含む）
    // 待ち牌計算、見えているドラ枚数計算、単独プレイヤー計算に使用
    #[derivative(Default(value = "[0; 34]"))]  // デフォルト値：全て0
    pub(super) tiles_seen: [u8; 34],

    /// For SPCalculator.
    // 見えている赤ドラのフラグ配列（[赤5萬, 赤5筒, 赤5索]）
    // 単独プレイヤー計算で赤ドラの所在を考慮するために使用
    pub(super) akas_seen: [bool; 3],

    // シャンテン数を維持する打牌候補を示すフラグ配列
    // true の位置の牌を切ってもシャンテン数が変わらない
    #[derivative(Default(value = "[false; 34]"))]  // デフォルト値：全てfalse
    pub(super) keep_shanten_discards: [bool; 34],

    // シャンテン数を進める（減らす）打牌候補を示すフラグ配列
    // true の位置の牌を切るとシャンテン数が1つ減る
    #[derivative(Default(value = "[false; 34]"))]  // デフォルト値：全てfalse
    pub(super) next_shanten_discards: [bool; 34],

    // 鳴くことができない牌を示すフラグ配列（喰い替え禁止牌）
    // チーやポンの後、同巡内に切ってはいけない牌がtrueになる
    #[derivative(Default(value = "[false; 34]"))]  // デフォルト値：全てfalse
    pub(super) forbidden_tiles: [bool; 34],

    /// Used for furiten check.
    // 自分が捨てた牌を記録するフラグ配列（フリテン判定用）
    // 一度でも捨てた牌はtrueになり、その牌でロンできなくなる
    #[derivative(Default(value = "[false; 34]"))]  // デフォルト値：全てfalse
    pub(super) discarded_tiles: [bool; 34],

    // 場風を表す牌（27:東, 28:南, 29:西, 30:北）
    pub(super) bakaze: Tile,
    // 自風を表す牌（27:東, 28:南, 29:西, 30:北）
    pub(super) jikaze: Tile,
    /// Counts from 0 unlike mjai.
    // 局数（0から数える。mjaiは1から数えるが、内部では0ベース）
    // 0:東1局, 1:東2局, ..., 3:東4局, 4:南1局, ...
    pub(super) kyoku: u8,
    // 本場数（連荘回数）
    pub(super) honba: u8,
    // 供託棒の数（リーチ棒の数）
    pub(super) kyotaku: u8,
    /// Rotated to be relative, so `scores[0]` is the score of the player.
    // 各プレイヤーの点数（相対的に回転済み）
    // scores[0]: 自分, scores[1]: 下家, scores[2]: 対面, scores[3]: 上家
    pub(super) scores: [i32; 4],
    // 現在の順位（1-4）
    pub(super) rank: u8,
    /// Relative to `player_id`.
    // 親の相対位置（0:自分が親, 1:下家が親, 2:対面が親, 3:上家が親）
    pub(super) oya: u8,
    /// Including 西入 sudden death.
    // オーラスフラグ（西入りのサドンデスも含む）
    pub(super) is_all_last: bool,
    // ドラ表示牌のリスト（最大5枚：初期ドラ + カンドラ×4）
    pub(super) dora_indicators: ArrayVec<[Tile; 5]>,

    /// 24 is the theoretical max size of kawa, however, since None is included
    /// in the kawa, in some very rare cases (about one in a million hanchans),
    /// the size can exceed 24.
    ///
    /// Reference:
    /// <https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q1020002370>
    // 各プレイヤーの河（捨て牌列）を保持する配列
    // Noneは鳴かれた牌を表す。理論上の最大サイズは24だが、極稀に超えることがある
    // kawa[0]: 自分, kawa[1]: 下家, kawa[2]: 対面, kawa[3]: 上家
    pub(super) kawa: [TinyVec<[Option<KawaItem>; 24]>; 4],
    // 各プレイヤーの最後の手出し牌を記録（ツモ切りでない牌）
    pub(super) last_tedashis: [Option<Sutehai>; 4],
    // 各プレイヤーのリーチ宣言牌を記録
    pub(super) riichi_sutehais: [Option<Sutehai>; 4],

    /// Using 34-D arrays here may be more efficient, but I don't want to mess up
    /// with aka doras.
    // 各プレイヤーの河の概要（牌のみを抽出したシンプルな配列）
    // 赤ドラも含めて個別の牌として保持（34次元配列より効率は劣るが、赤ドラ処理が簡単）
    pub(super) kawa_overview: [ArrayVec<[Tile; 24]>; 4],
    // 各プレイヤーの副露（チー、ポン、大明槓）の概要
    // fuuro_overview[player][meld_index][tile_index]
    pub(super) fuuro_overview: [ArrayVec<[ArrayVec<[Tile; 4]>; 4]>; 4],
    /// In this field all `Tile` are deaka'd.
    // 各プレイヤーの暗槓の概要（全ての牌は赤ドラ情報を除去済み）
    pub(super) ankan_overview: [ArrayVec<[Tile; 4]>; 4],

    // 各プレイヤーがリーチを宣言したかを示すフラグ
    pub(super) riichi_declared: [bool; 4],
    // 各プレイヤーのリーチが成立したかを示すフラグ（リーチ棒を出したか）
    pub(super) riichi_accepted: [bool; 4],

    // 現在の手番プレイヤー（0:自分, 1:下家, 2:対面, 3:上家）
    pub(super) at_turn: u8,
    // 山に残っている牌の枚数
    pub(super) tiles_left: u8,
    // 処理中のカン（加槓、暗槓、大明槓）の牌
    pub(super) intermediate_kan: ArrayVec<[Tile; 4]>,
    // 処理中のチー・ポンの情報
    pub(super) intermediate_chi_pon: Option<ChiPon>,

    // 現在のシャンテン数（-1:聴牌, 0:イーシャンテン, 1:リャンシャンテン...）
    pub(super) shanten: i8,

    // 自分が最後にツモった牌
    pub(super) last_self_tsumo: Option<Tile>,
    // 最後に捨てられた牌（誰が捨てたかは at_turn で判断）
    pub(super) last_kawa_tile: Option<Tile>,
    // 最後に計算された実行可能なアクション候補
    pub(super) last_cans: ActionCandidate,

    /// Both deaka'd
    // 暗槓可能な牌のリスト（赤ドラ情報は除去済み）
    pub(super) ankan_candidates: ArrayVec<[Tile; 3]>,
    // 加槓可能な牌のリスト（赤ドラ情報は除去済み）
    pub(super) kakan_candidates: ArrayVec<[Tile; 3]>,
    // 槍槓のチャンスがあるかを示すフラグ
    pub(super) chankan_chance: Option<()>,

    // ダブルリーチ可能かを示すフラグ
    pub(super) can_w_riichi: bool,
    // ダブルリーチ状態かを示すフラグ
    pub(super) is_w_riichi: bool,
    // 嶺上開花の可能性がある状態かを示すフラグ（カン直後のツモ）
    pub(super) at_rinshan: bool,
    // 一発の可能性がある状態かを示すフラグ（リーチ後1巡目）
    pub(super) at_ippatsu: bool,
    // フリテン状態かを示すフラグ
    pub(super) at_furiten: bool,
    // 同巡フリテンをマークする必要があるかを示すフラグ
    pub(super) to_mark_same_cycle_furiten: Option<()>,

    /// Used for 4-kan check.
    // 場に出ているカンの総数（四槓流れの判定用）
    pub(super) kans_on_board: u8,

    // 門前（鳴いていない）状態かを示すフラグ
    pub(super) is_menzen: bool,
    /// For agari calc, all deaka'd.
    // 自分のチーした牌のリスト（和了計算用、赤ドラ情報は除去済み）
    pub(super) chis: ArrayVec<[u8; 4]>,
    // 自分のポンした牌のリスト（和了計算用、赤ドラ情報は除去済み）
    pub(super) pons: ArrayVec<[u8; 4]>,
    // 自分の大明槓した牌のリスト（和了計算用、赤ドラ情報は除去済み）
    pub(super) minkans: ArrayVec<[u8; 4]>,
    // 自分の暗槓した牌のリスト（和了計算用、赤ドラ情報は除去済み）
    pub(super) ankans: ArrayVec<[u8; 4]>,

    /// Including aka, originally for agari calc usage but also encoded as a
    /// feature to the obs.
    // 所有しているドラの枚数（[ドラ, 裏ドラ, 赤ドラ, 抜きドラ]）
    // 和了計算用だが、観測特徴量としてもエンコードされる
    pub(super) doras_owned: [u8; 4],
    // 見えているドラの総枚数
    pub(super) doras_seen: u8,

    // 手牌内の赤ドラの有無（[赤5萬, 赤5筒, 赤5索]）
    pub(super) akas_in_hand: [bool; 3],

    /// For shanten calc.
    // 手牌の長さを3で割った値（シャンテン計算用）
    // 通常は4（13枚÷3）、チー・ポンすると減る
    pub(super) tehai_len_div3: u8,

    /// Used in can_riichi, also in single-player features to get the shanten
    /// for 3n+2.
    // シャンテン数を進める打牌が存在するかを示すフラグ
    // リーチ可能判定と、3n+2形のシャンテン数取得に使用
    pub(super) has_next_shanten_discard: bool,
}

// Python から呼び出し可能なメソッドを定義するブロック
#[pymethods]
impl PlayerState {
    /// Panics if `player_id` is outside of range [0, 3].
    // コンストラクタ：新しい PlayerState インスタンスを作成
    #[new]  // PyO3 のコンストラクタマクロ
    #[must_use]  // 戻り値を使用しない場合に警告を出す
    pub fn new(player_id: u8) -> Self {
        // player_id が有効範囲（0-3）内かをチェック
        assert!(player_id < 4, "{player_id} is not in range [0, 3]");
        Self {
            player_id,  // 指定されたプレイヤーIDを設定
            ..Default::default()  // その他のフィールドはデフォルト値を使用
        }
    }

    /// Returns an `ActionCandidate`.
    // mjai形式のJSONイベントを受け取って状態を更新し、可能なアクションを返す
    #[pyo3(name = "update")]  // Python側での関数名を指定
    pub(super) fn update_json(&mut self, mjai_json: &str) -> Result<ActionCandidate> {
        // JSON文字列をパースしてイベントオブジェクトに変換
        let event = json::from_str(mjai_json)?;
        // 実際の更新処理を呼び出し（別モジュールで実装）
        self.update(&event)
    }

    /// Raises an exception if the action is not valid.
    // プレイヤーのアクション（反応）が現在の状態で有効かを検証
    #[pyo3(name = "validate_reaction")]  // Python側での関数名を指定
    pub(super) fn validate_reaction_json(&self, mjai_json: &str) -> Result<()> {
        // JSON文字列をパースしてアクションオブジェクトに変換
        let action = json::from_str(mjai_json)?;
        // 実際の検証処理を呼び出し（別モジュールで実装）
        self.validate_reaction(&action)
    }

    /// For debug only.
    ///
    /// Return a human readable description of the current state.
    // デバッグ用：現在の状態を人間が読める形式で返す
    #[must_use]  // 戻り値を使用しない場合に警告
    pub fn brief_info(&self) -> String {
        // 待ち牌のリストを作成
        let waits = self
            .waits  // 待ち牌フラグ配列
            .iter()  // イテレータに変換
            .enumerate()  // インデックス付きイテレータに変換
            .filter(|&(_, &b)| b)  // true（待ち牌）のものだけフィルタ
            .map(|(i, _)| must_tile!(i))  // インデックスを牌オブジェクトに変換
            .collect::<Vec<_>>();  // ベクターに収集

        // 4人分の河を並列表示用にフォーマット
        let zipped_kawa = self.kawa[0]  // 自分の河
            .iter()
            .chain(iter::repeat(&None))  // 長さを揃えるためにNoneで埋める
            .zip(self.kawa[1].iter().chain(iter::repeat(&None)))  // 下家の河と結合
            .zip(self.kawa[2].iter().chain(iter::repeat(&None)))  // 対面の河と結合
            .zip(self.kawa[3].iter().chain(iter::repeat(&None)))  // 上家の河と結合
            .take_while(|row| !matches!(row, &(((None, None), None), None)))  // 全員Noneになったら終了
            .enumerate()  // 行番号を付与
            .map(|(i, (((a, b), c), d))| {
                format!(
                    "{i:2}. {}\t{}\t{}\t{}",  // 行番号と4人分の捨て牌を表示
                    a.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),  // 自分の捨て牌
                    b.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),  // 下家の捨て牌
                    c.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),  // 対面の捨て牌
                    d.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),  // 上家の捨て牌
                )
            })
            .collect::<Vec<_>>()  // ベクターに収集
            .join("\n");  // 改行で結合

        // 単独プレイヤー向けの期待値テーブルを作成
        let can_discard = self.last_cans.can_discard;  // 打牌可能かどうか
        let mut sp_tables = Candidate::csv_header(can_discard).join("\t");  // CSVヘッダーを作成
        if let Ok(tables) = self.single_player_tables() {  // 期待値テーブルを計算
            for candidate in tables.max_ev_table {  // 各候補について
                sp_tables.push('\n');  // 改行を追加
                sp_tables.push_str(&candidate.csv_row(can_discard).join("\t"));  // 行データを追加
            }
        }

        // デバッグ情報を整形して返す
        format!(
            r#"player (abs): {}  // プレイヤーID（絶対位置）
oya (rel): {}  // 親の相対位置
kyoku: {}{}-{}  // 場風、局数、本場
turn: {}  // 現在の手番
jikaze: {}  // 自風
score (rel): {:?}  // 点数（相対位置）
tehai: {}  // 手牌
fuuro: {:?}  // 副露
ankan: {:?}  // 暗槓
tehai len: {}  // 手牌の長さ÷3
shanten: {} (actual: {})  // シャンテン数（キャッシュ値と実計算値）
furiten: {}  // フリテン状態
waits: {waits:?}  // 待ち牌
dora indicators: {:?}  // ドラ表示牌
doras owned: {:?}  // 所有ドラ数
doras seen: {}  // 見えているドラ数
action candidates: {:#?}  // 可能なアクション
last self tsumo: {:?}  // 最後の自摸牌
last kawa tile: {:?}  // 最後の捨て牌
tiles left: {}  // 残り牌数
kawa:  // 河の状態
{zipped_kawa}
single player table (max EV):  // 単独プレイヤー期待値テーブル
{sp_tables}"#,
            self.player_id,  // プレイヤーID
            self.oya,  // 親の相対位置
            self.bakaze,  // 場風
            self.kyoku + 1,  // 局数（表示用に1を加算）
            self.honba,  // 本場数
            self.at_turn,  // 現在の手番
            self.jikaze,  // 自風
            self.scores,  // 点数配列
            tiles_to_string(&self.tehai, self.akas_in_hand),  // 手牌を文字列化（赤ドラ込み）
            self.fuuro_overview[0],  // 自分の副露
            self.ankan_overview[0],  // 自分の暗槓
            self.tehai_len_div3,  // 手牌長÷3
            self.shanten,  // キャッシュされたシャンテン数
            self.real_time_shanten(),  // リアルタイムで計算したシャンテン数
            self.at_furiten,  // フリテン状態
            self.dora_indicators,  // ドラ表示牌
            self.doras_owned,  // 所有ドラ数
            self.doras_seen,  // 見えているドラ数
            self.last_cans,  // 可能なアクション
            self.last_self_tsumo,  // 最後の自摸牌
            self.last_kawa_tile,  // 最後の捨て牌
            self.tiles_left,  // 残り牌数
        )
    }
}
