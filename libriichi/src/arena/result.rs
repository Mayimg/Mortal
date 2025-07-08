// ゲーム結果の構造体と関連機能の実装。
// 局結果、ゲーム結果の保存、ランキング計算、JSONログ出力機能を提供する。

use crate::mjai::{Event, EventExt};                    // Mjaiフォーマットのイベント
use crate::rankings::Rankings;                         // ランキング計算用構造体

use anyhow::Result;                                    // エラーハンドリング
use serde_json as json;                                // JSONシリアライズ用

/// 局（ラウンド）の結果を保持する構造体。
/// 局終了時の状態、点数、次局への情報を含む。
#[derive(Debug, Clone)]
pub struct KyokuResult {
    pub kyoku: u8,                             // 局数（東一局が0）
    // pub honba: u8,                          // 本場数（コメントアウト）
    pub can_renchan: bool,                     // 連荘可能か
    pub has_hora: bool,                        // 和了があったか
    pub has_abortive_ryukyoku: bool,           // 途中流局があったか
    pub kyotaku_left: u8,                      // 残り供託棒数
    pub scores: [i32; 4],                      // 各プレイヤーの点数
}

/// ゲーム全体の結果を保持する構造体。
/// 最終結果、プレイヤー情報、ゲームログを含む。
#[derive(Debug, Clone, Default)]
pub struct GameResult {
    pub names: [String; 4],                    // 各プレイヤーの名前
    pub scores: [i32; 4],                      // 最終点数
    pub seed: (u64, u64),                      // ゲームのシード値
    pub game_log: Vec<Vec<EventExt>>,          // ゲームログ（局ごとのイベントリスト）
}

impl GameResult {
    /// ゲーム結果からランキング情報を計算する。
    /// 同点も考慮した正確な順位を返す。
    #[inline]
    pub fn rankings(&self) -> Rankings {
        Rankings::new(self.scores)             // 点数からランキングを計算
    }

    /// ゲームログをJSON形式でダンプする。
    /// Mjaiフォーマットに準拠した改行区切りJSONを生成する。
    pub fn dump_json_log(&self) -> Result<String> {
        let mut v = vec![];                    // 出力用バッファ

        let start_game = Event::StartGame {    // ゲーム開始イベント
            names: self.names.clone(),         // プレイヤー名
            seed: Some(self.seed),             // シード値
        };
        json::to_writer(&mut v, &start_game)?; // JSONとして書き込み
        v.push(b'\n');                        // 改行を追加

        for ev in self.game_log.iter().flatten() {  // 全局の全イベントに対して
            json::to_writer(&mut v, ev)?;      // JSONとして書き込み
            v.push(b'\n');                     // 改行を追加
        }

        json::to_writer(&mut v, &Event::EndGame)?;  // ゲーム終了イベント
        v.push(b'\n');                        // 改行を追加

        Ok(String::from_utf8(v)?)              // UTF-8文字列に変換して返す
    }
}
