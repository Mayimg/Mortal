// ================================================================================
// action.rs - 麻雀ゲームにおけるプレイヤーのアクション管理
// ================================================================================
// このファイルは麻雀ゲームにおいて、プレイヤーが取ることができるアクション（行動）を
// 管理するための構造体と、アクションの妥当性を検証するロジックを提供します。
//
// 主要な機能：
// 1. ActionCandidate構造体：プレイヤーが現在取ることができる全てのアクションを表現
// 2. validate_reaction関数：プレイヤーが選択したアクションが現在の状態で有効かを検証
//
// 麻雀のアクションには以下のようなものがあります：
// - 打牌（Dahai）：手牌から牌を捨てる
// - チー（Chi）：上家の捨て牌で順子を作る
// - ポン（Pon）：他家の捨て牌で刻子を作る  
// - カン（Kan）：槓子を作る（暗槓、明槓、加槓の3種類）
// - リーチ（Riichi）：聴牌を宣言する
// - アガリ（Agari）：和了する（ツモ和了、ロン和了）
// - 流局（Ryukyoku）：ゲームを引き分けにする
// ================================================================================

use super::PlayerState;  // プレイヤーの状態を管理する構造体
use crate::chi_type::ChiType;  // チーの種類（上チー、中チー、下チー）を表す列挙型
use crate::mjai::Event;  // mjaiプロトコルのイベント（アクション）を表す列挙型
use crate::tile::Tile;  // 麻雀牌を表す構造体
use crate::tuz;  // タイル定数を生成するマクロ（例：tuz!(5mr) は赤5萬を表す）

use anyhow::{Result, bail, ensure};  // エラーハンドリング用のマクロと型
use pyo3::prelude::*;  // Python バインディング用のマクロ
use serde::Serialize;  // シリアライズ用のトレイト

// ActionCandidate構造体：プレイヤーが現在取ることができる全てのアクションを表現
// #[pyclass] - Python側からアクセス可能なクラスとして公開
// #[derive(...)] - 自動的にトレイトを実装：
//   - Debug: デバッグ出力用
//   - Default: デフォルト値（全てfalse、target_actorは0）を提供
//   - Clone: 値のコピーを可能にする
//   - Copy: 軽量なコピー（メモリ上でのコピー）を可能にする
//   - Serialize: JSONなどへのシリアライズを可能にする
#[pyclass]
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct ActionCandidate {
    // 打牌可能フラグ：手牌から牌を捨てることができるか
    #[pyo3(get)]  // Python側からの読み取りを許可
    pub can_discard: bool,
    
    // 下チー可能フラグ：上家の捨て牌で下側の順子（例：1-2で3を鳴く）を作れるか
    #[pyo3(get)]
    pub can_chi_low: bool,
    
    // 中チー可能フラグ：上家の捨て牌で中間の順子（例：1-3で2を鳴く）を作れるか
    #[pyo3(get)]
    pub can_chi_mid: bool,
    
    // 上チー可能フラグ：上家の捨て牌で上側の順子（例：2-3で1を鳴く）を作れるか
    #[pyo3(get)]
    pub can_chi_high: bool,
    
    // ポン可能フラグ：他家の捨て牌で刻子（同じ牌3枚）を作れるか
    #[pyo3(get)]
    pub can_pon: bool,
    
    // 大明槓（ダイミンカン）可能フラグ：他家の捨て牌で明槓（4枚組）を作れるか
    #[pyo3(get)]
    pub can_daiminkan: bool,
    
    // 加槓（カカン）可能フラグ：既にポンした刻子に4枚目を加えて槓にできるか
    #[pyo3(get)]
    pub can_kakan: bool,
    
    // 暗槓（アンカン）可能フラグ：手牌の同じ牌4枚で暗槓を作れるか
    #[pyo3(get)]
    pub can_ankan: bool,
    
    // リーチ可能フラグ：聴牌を宣言できるか
    #[pyo3(get)]
    pub can_riichi: bool,
    
    // ツモ和了可能フラグ：自摸した牌で和了できるか
    #[pyo3(get)]
    pub can_tsumo_agari: bool,
    
    // ロン和了可能フラグ：他家の捨て牌で和了できるか
    #[pyo3(get)]
    pub can_ron_agari: bool,
    
    // 流局可能フラグ：ゲームを引き分けにできるか（九種九牌など）
    #[pyo3(get)]
    pub can_ryukyoku: bool,

    // ターゲットプレイヤー：鳴きやロンの対象となるプレイヤーのID（0-3）
    #[pyo3(get)]
    pub target_actor: u8,
}

// ActionCandidateの実装ブロック
// #[pymethods] - Python側から呼び出し可能なメソッドを定義
#[pymethods]
impl ActionCandidate {
    // チー可能判定：3種類のチー（上・中・下）のいずれかが可能かを返す
    // #[getter] - Pythonでプロパティとしてアクセス可能（例：candidate.can_chi）
    // #[inline] - コンパイラに関数のインライン展開を推奨（パフォーマンス最適化）
    // #[must_use] - 戻り値を使用しない場合に警告を出す
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_chi(&self) -> bool {
        self.can_chi_low || self.can_chi_mid || self.can_chi_high
    }

    // カン可能判定：3種類のカン（大明槓・加槓・暗槓）のいずれかが可能かを返す
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_kan(&self) -> bool {
        self.can_daiminkan || self.can_kakan || self.can_ankan
    }

    // 和了可能判定：ツモ和了またはロン和了のいずれかが可能かを返す
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_agari(&self) -> bool {
        self.can_tsumo_agari || self.can_ron_agari
    }

    // パス（スキップ）可能判定：他家のアクションに対して反応可能な状況かを返す
    // チー、ポン、大明槓、ロン和了のいずれかが可能な場合はパスという選択肢がある
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_pass(&self) -> bool {
        self.can_chi() || self.can_pon || self.can_daiminkan || self.can_ron_agari
    }

    // アクション可能判定：何らかのアクションが取れる状況かを返す
    // プレイヤーがゲームに対して何らかの入力が必要な状況かを判定
    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_act(&self) -> bool {
        self.can_discard      // 打牌可能
            || self.can_chi() // チー可能
            || self.can_pon   // ポン可能
            || self.can_kan() // カン可能
            || self.can_riichi // リーチ可能
            || self.can_agari() // 和了可能
            || self.can_ryukyoku // 流局可能
    }

    // Python用の文字列表現メソッド
    // repr()関数やデバッグ時に使用される
    // 例：ActionCandidate { can_discard: true, can_chi_low: false, ... }
    fn __repr__(&self) -> String {
        format!("{self:?}")  // Debug トレイトの実装を使用してフォーマット
    }
}

// PlayerStateに対するvalidate_reaction メソッドの実装
// このメソッドはプレイヤーが選択したアクションが現在のゲーム状態で有効かを検証する
impl PlayerState {
    /// アクションの妥当性を検証する
    /// 
    /// # 引数
    /// - `action`: 検証対象のアクション（Event型）
    /// 
    /// # 戻り値
    /// - `Result<()>`: 有効なアクションの場合はOk(())、無効な場合はエラー
    pub fn validate_reaction(&self, action: &Event) -> Result<()> {
        // 現在取ることができるアクションの候補を取得
        let cans = self.last_cans;

        // 最初に特殊なケース（流局とパス）を処理
        match action {
            // 流局の場合：流局可能フラグをチェック
            Event::Ryukyoku { .. } => {
                ensure!(cans.can_ryukyoku, "cannot ryukyoku");  // 流局不可の場合はエラー
                return Ok(());  // 早期リターン
            }
            // パス（何もしない）の場合：常に有効
            Event::None => {
                return Ok(());  // 早期リターン
            }
            // その他のアクション：後続の処理で検証
            _ => (),
        };

        // アクションの実行者（actor）が自分自身であることを確認
        if let Some(actor) = action.actor() {
            // アクターが自分のプレイヤーIDと一致するかチェック
            ensure!(
                actor == self.player_id,
                "actor is {actor}, not self ({})",  // 不一致の場合のエラーメッセージ
                self.player_id,
            );
        } else {
            // アクターが設定されていない（かつ流局でもない）場合はエラー
            bail!("action does not have actor and is not ryukyoku");
        }

        // 各種アクションの詳細な検証
        match *action {
            // 打牌（Dahai）アクションの検証
            Event::Dahai { pai, tsumogiri, .. } => {
                // 打牌可能フラグをチェック
                ensure!(cans.can_discard, "cannot discard");
                // 捨てようとしている牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&[pai])?;
                // ツモ切り（自摸した牌をそのまま捨てる）の場合の追加検証
                if tsumogiri {
                    if let Some(tile) = self.last_self_tsumo {
                        // 最後に自摸した牌と捨てる牌が一致することを確認
                        ensure!(tile == pai, "cannot tsumogiri");
                    } else {
                        // まだ自摸していないのにツモ切りしようとしている場合はエラー
                        bail!("tsumogiri but the player has not dealt any tile yet");
                    }
                }
            }

            // リーチアクションの検証
            Event::Reach { .. } => {
                // リーチ可能フラグをチェック
                ensure!(cans.can_riichi, "cannot riichi");
            }

            // チー（順子を作る）アクションの検証
            Event::Chi {
                actor,      // チーを行うプレイヤー
                target,     // チーの対象プレイヤー（捨て牌の持ち主）
                pai,        // チー対象の牌
                consumed,   // 手牌から使用する2枚の牌
            } => {
                // チーは上家（左隣のプレイヤー）からのみ可能
                // (target + 1) % 4 == actor という式で上家関係を確認
                ensure!((target + 1) % 4 == actor, "chi from non-kamicha");
                // チー対象の牌が最新の捨て牌であることを確認
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "chi target is not the last kawa tile",
                );
                // 使用する2枚の牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&consumed)?;

                // チーの種類（上・中・下）に応じて適切なフラグをチェック
                match ChiType::new(consumed, pai) {
                    ChiType::Low => ensure!(cans.can_chi_low, "cannot chi low"),
                    ChiType::Mid => ensure!(cans.can_chi_mid, "cannot chi mid"),
                    ChiType::High => ensure!(cans.can_chi_high, "cannot chi high"),
                }
            }
            // ポン（刻子を作る）アクションの検証
            Event::Pon {
                actor,      // ポンを行うプレイヤー
                target,     // ポンの対象プレイヤー（捨て牌の持ち主）
                pai,        // ポン対象の牌
                consumed,   // 手牌から使用する2枚の牌（ポン対象と合わせて3枚になる）
            } => {
                // 自分自身からポンはできない
                ensure!(target != actor, "pon from itself");
                // ポン対象の牌が最新の捨て牌であることを確認
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "pon target is not the last kawa tile",
                );
                // ポン可能フラグをチェック
                ensure!(cans.can_pon, "cannot pon");
                // 使用する2枚の牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&consumed)?;
            }

            // 大明槓（他家の捨て牌で槓を作る）アクションの検証
            Event::Daiminkan {
                actor,      // 大明槓を行うプレイヤー
                target,     // 大明槓の対象プレイヤー
                pai,        // 大明槓対象の牌
                consumed,   // 手牌から使用する3枚の牌（対象と合わせて4枚になる）
            } => {
                // 自分自身から大明槓はできない
                ensure!(target != actor, "daiminkan from itself");
                // 大明槓対象の牌が最新の捨て牌であることを確認
                ensure!(
                    matches!(self.last_kawa_tile, Some(tile) if tile == pai),
                    "daiminkan target is not the last kawa tile",
                );
                // 大明槓可能フラグをチェック
                ensure!(cans.can_daiminkan, "cannot daiminkan");
                // 使用する3枚の牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&consumed)?;
            }
            
            // 加槓（既存のポンに1枚加えて槓にする）アクションの検証
            Event::Kakan { pai, .. } => {
                // 加槓可能フラグをチェック
                ensure!(cans.can_kakan, "cannot kakan");
                // 加槓する牌が候補リストに含まれていることを確認
                // deaka()で赤ドラを通常の牌に変換してから確認
                ensure!(
                    self.kakan_candidates.contains(&pai.deaka()),
                    "cannot kakan {pai}",
                );
                // 加槓する牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&[pai])?;
            }
            
            // 暗槓（手牌の同じ牌4枚で槓を作る）アクションの検証
            Event::Ankan { consumed, .. } => {
                // 暗槓可能フラグをチェック
                ensure!(cans.can_ankan, "cannot ankan");
                // 暗槓する牌の種類を取得（4枚とも同じ牌のはず）
                let tile = consumed[0].deaka();
                // その牌が暗槓候補リストに含まれていることを確認
                ensure!(self.ankan_candidates.contains(&tile), "cannot ankan {tile}");
                // 使用する4枚の牌が手牌にあることを確認
                self.ensure_tiles_in_hand(&consumed)?;
            }

            // 和了（アガリ）アクションの検証
            Event::Hora { target, .. } => {
                if target == self.player_id {
                    // 自分自身が対象 = ツモ和了
                    ensure!(cans.can_tsumo_agari, "cannot tsumo agari");
                } else {
                    // 他家が対象 = ロン和了
                    ensure!(cans.can_ron_agari, "cannot ron agari");
                }
            }

            // パスアクション（再度の確認、通常はここに到達しない）
            Event::None => return Ok(()),

            // 予期しないアクションタイプの場合はエラー
            _ => bail!("unexpected action {action:?}"),
        };

        Ok(())  // 全ての検証を通過した場合は成功
    }

    // 指定された牌が手牌に存在することを確認するヘルパーメソッド
    // 
    // # 引数
    // - `tiles`: 確認対象の牌のスライス
    // 
    // # 戻り値
    // - `Result<()>`: 全ての牌が手牌にある場合はOk(())、ない場合はエラー
    fn ensure_tiles_in_hand(&self, tiles: &[Tile]) -> Result<()> {
        // 各牌について手牌に存在するかチェック
        for &tile in tiles {
            // 通常の牌（赤ドラを通常牌に変換）として手牌配列をチェック
            // tehai配列は牌の種類ごとの枚数を保持している
            ensure!(
                self.tehai[tile.deaka().as_usize()] > 0,
                "{tile} is not in hand",  // 手牌にない場合のエラーメッセージ
            );
            // 赤ドラの場合は追加の確認が必要
            if tile.is_aka() {
                // akas_in_hand配列で赤ドラの有無を確認
                // 赤ドラは5萬、5筒、5索の3種類のみ
                // tuz!(5mr)は赤5萬の定数値を表す
                ensure!(
                    self.akas_in_hand[tile.as_usize() - tuz!(5mr)],
                    "{tile} is not in hand",  // 赤ドラが手牌にない場合のエラー
                );
            }
        }
        Ok(())  // 全ての牌が手牌に存在する場合は成功
    }
}
