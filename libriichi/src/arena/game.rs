// バッチゲーム実行エンジンの実装。
// 複数のゲームを同時に実行し、エージェントのバッチ処理を最適化する。
// プログレスバー表示、ゲーム結果の収集、エージェント管理を行う。

use super::board::{Board, BoardState, Poll};           // ボード関連の型
use super::result::GameResult;                         // ゲーム結果構造体
use crate::agent::BatchAgent;                          // バッチ処理対応エージェントトレイト
use crate::mjai::EventExt;                             // メタデータ付きイベント
use std::time::Duration;                               // 時間間隔
use std::{array, mem};                                 // 標準ライブラリ

use anyhow::{Result, ensure};                          // エラーハンドリング
use indicatif::{ProgressBar, ProgressStyle};           // プログレスバー表示
use ndarray::prelude::*;                                // 多次元配列（Oracle観測データ用）

/// バッチゲーム実行用の設定構造体。
pub struct BatchGame {
    /// ゲーム長：8が半荘戦、4が東風戦
    pub length: u8,
    /// 初期点数（通常は[25000; 4]）
    pub init_scores: [i32; 4],
    /// プログレスバー表示を無効にするか
    pub disable_progress_bar: bool,
}

/// ゲームとエージェント間のインデックスマッピング。
/// バッチ処理で複数ゲーム・複数エージェントを扱うための索引。
#[derive(Clone, Copy, Default)]
pub struct Index {
    /// Gameが特定のAgentを見つけるためのインデックス（game → agent）
    pub agent_idx: usize,
    /// Agentが特定のプレイヤーIDを見つけるためのインデックス（agent → game）
    pub player_id_idx: usize,
}

/// 単一ゲームの状態管理構造体。
#[derive(Default)]
struct Game {
    length: u8,                                         // ゲーム長（半荘戦なら8、東風戦なら4）
    seed: (u64, u64),                                   // ゲームシード
    indexes: [Index; 4],                                // プレイヤーとエージェントのマッピング

    oracle_obs_versions: [Option<u32>; 4],              // 各プレイヤーのOracle観測バージョン
    invisible_state_cache: [Option<Array2<f32>>; 4],    // Oracle観測データのキャッシュ

    last_reactions: [EventExt; 4],                      // ポーリングフェーズ用のリアクションキャッシュ

    board: BoardState,                                  // ボード状態
    kyoku: u8,                                          // 現在の局数
    honba: u8,                                          // 本場数
    kyotaku: u8,                                        // 供託棒数
    scores: [i32; 4],                                   // 各プレイヤーの点数
    game_log: Vec<Vec<EventExt>>,                       // ゲームログ（局ごと）

    kyoku_started: bool,                                // 局が開始されたか
    ended: bool,                                        // ゲームが終了したか
    /// 西入（サドンデス）時に使用。親と他プレイヤーが同時に30000点に達したが、
    /// 親がトップでないためゲームが続行する場合に使用。
    ///
    /// 天鳳のルールより：
    ///
    /// > サドンデスルールは、30000点(供託未収)以上になった時点で終了、ただし親の
    /// > 連荘がある場合は連荘を優先する
    in_renchan: bool,                                   // 連荘中か
}

impl Game {
    /// ゲームの状態をポーリングし、必要に応じてエージェントにアクションを要求する。
    /// いずれかのプレイヤーがアクション可能か、ゲームが終了した場合に返る。
    fn poll(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<()> {
        if self.ended {                                          // 既にゲーム終了済み
            return Ok(());
        }

        if !self.kyoku_started {                                 // 局がまだ開始されていない場合
            // ゲーム終了条件のチェック：
            // 1. 西4局終了後
            // 2. または、オーラス終了後で
            //    親が連荘中でなく（連荘中なら既に連荘終了チェックで終了しているはず）
            //    誰かが30000点以上持っている
            if self.kyoku >= self.length + 4                    // 西4局後
                || self.kyoku >= self.length                     // オーラス後
                    && !self.in_renchan                          // 連荘中ではない
                    && self.scores.iter().any(|&s| s >= 30000)  // 誰か30000点以上
            {
                self.ended = true;                               // ゲーム終了
                return Ok(());
            }

            let mut next_board = Board {                         // 新しい局のボードを作成
                kyoku: self.kyoku,                               // 局数
                honba: self.honba,                               // 本場数
                kyotaku: self.kyotaku,                           // 供託棒
                scores: self.scores,                             // 点数
                ..Default::default()                             // その他はデフォルト
            };
            next_board.init_from_seed(self.seed);               // シードから初期化
            self.board = next_board.into_state();               // ボード状態に変換
            self.kyoku_started = true;                           // 局開始フラグをセット
        }

        let reactions = mem::take(&mut self.last_reactions);    // キャッシュされたリアクションを取り出す
        let poll = self.board.poll(reactions)?;                 // ボードをポーリング
        match poll {
            Poll::InGame => {                                    // ゲーム継続中の場合
                let ctx = self.board.agent_context();            // エージェント用コンテキストを取得
                for (player_id, state) in ctx.player_states.iter().enumerate() {  // 各プレイヤーに対して
                    if !state.last_cans().can_act() {           // アクション不可の場合
                        continue;                                // スキップ
                    }

                    let invisible_state = self.oracle_obs_versions[player_id]     // Oracle観測データ
                        .map(|ver| self.board.encode_oracle_obs(player_id as u8, ver));  // バージョンに応じてエンコード
                    self.invisible_state_cache[player_id].clone_from(&invisible_state);  // キャッシュに保存

                    let idx = self.indexes[player_id];           // プレイヤーのインデックス情報
                    agents[idx.agent_idx].set_scene(             // エージェントにシーンを設定
                        idx.player_id_idx,                       // エージェント内でのプレイヤーID
                        ctx.log,                                 // ゲームログ
                        state,                                   // プレイヤー状態
                        invisible_state,                         // Oracle観測データ
                    )?;
                }
            }

            Poll::End => {                                       // 局終了の場合
                self.kyoku_started = false;                      // 局開始フラグをクリア
                self.in_renchan = false;                         // 連荘フラグをクリア

                for idx in &self.indexes {                       // 全エージェントに局終了を通知
                    agents[idx.agent_idx].end_kyoku(idx.player_id_idx)?;
                }

                let kyoku_result = self.board.end();             // 局の結果を取得
                self.kyotaku = kyoku_result.kyotaku_left;        // 残り供託棒を更新
                self.scores = kyoku_result.scores;               // 点数を更新

                let logs = self.board.take_log();                // 局のログを取得
                self.game_log.push(logs);                        // ゲームログに追加

                let has_tobi = self.scores.iter().any(|&s| s < 0);  // トビ（マイナス点）チェック
                if has_tobi {                                    // トビがあった場合
                    self.ended = true;                           // ゲーム終了
                    return Ok(());
                }

                if kyoku_result.has_abortive_ryukyoku {          // 途中流局の場合
                    self.honba += 1;                             // 本場を増やす
                    return self.poll(agents);                    // 次の局へ
                }

                if !kyoku_result.can_renchan {                   // 連荘できない場合
                    self.kyoku += 1;                             // 次の局へ
                    if kyoku_result.has_hora {                   // 和了があった場合
                        self.honba = 0;                          // 本場をリセット
                    } else {                                     // 流局の場合
                        self.honba += 1;                         // 本場を増やす
                    }
                    return self.poll(agents);                    // 次の局へ
                }

                // 連荘終了条件：
                // 1. 連荘可能
                // 2. オーラス
                // 3. 親が30000点以上
                // 4. 親がトップ
                let oya = kyoku_result.kyoku as usize % 4;              // 親の座席番号
                if kyoku_result.kyoku >= self.length - 1 && self.scores[oya] >= 30000 {  // オーラスで親が30000点以上
                    let top = kyoku_result                               // トッププレイヤーを探す
                        .scores
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, &s)| -s)                       // 点数の逆順でソート（最高点が先頭）
                        .map(|(i, _)| i)                                 // インデックスを取得
                        .unwrap();                                       // unwrapは安全（必ず1人はいる）
                    if top == oya {                                      // 親がトップの場合
                        self.ended = true;                               // ゲーム終了
                        return Ok(());
                    }
                }

                // 連荘
                self.in_renchan = true;                                  // 連荘フラグをセット
                self.honba += 1;                                         // 本場を増やす
                return self.poll(agents);                                // 次の局へ
            }
        };

        Ok(())
    }

    /// エージェントのリアクションをコミットし、ゲーム終了時は結果を返す。
    fn commit(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<Option<GameResult>> {
        if self.ended {                                                  // ゲーム終了時
            if self.kyotaku > 0 {                                        // 供託棒が残っている場合
                *self.scores.iter_mut().min_by_key(|s| -**s).unwrap() += self.kyotaku as i32 * 1000;  // トッププレイヤーが総取り
            }

            let names = array::from_fn(|i| agents[self.indexes[i].agent_idx].name());  // エージェント名を収集
            let game_result = GameResult {                               // ゲーム結果を作成
                names,                                                   // プレイヤー名
                scores: self.scores,                                     // 最終点数
                seed: self.seed,                                         // ゲームシード
                game_log: mem::take(&mut self.game_log),                // ゲームログ（ムーブ）
            };

            for idx in &self.indexes {                                   // 全エージェントにゲーム終了を通知
                agents[idx.agent_idx].end_game(idx.player_id_idx, &game_result)?;
            }
            return Ok(Some(game_result));                                // ゲーム結果を返す
        }

        let ctx = self.board.agent_context();                            // エージェント用コンテキストを取得
        for (player_id, state) in ctx.player_states.iter().enumerate() { // 各プレイヤーに対して
            if !state.last_cans().can_act() {                            // アクション不可の場合
                continue;                                                 // スキップ
            }

            let invisible_state = self.invisible_state_cache[player_id].take();  // キャッシュからOracleデータを取り出す

            let idx = self.indexes[player_id];                            // プレイヤーのインデックス情報
            self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(  // エージェントからリアクションを取得
                idx.player_id_idx,                                        // エージェント内でのプレイヤーID
                ctx.log,                                                  // ゲームログ
                state,                                                    // プレイヤー状態
                invisible_state,                                          // Oracle観測データ
            )?;
        }

        Ok(None)                                                          // ゲーム継続中
    }
}

impl BatchGame {
    /// 天鳳ルールの半荘戦設定を作成する。
    /// 初期点25000点、8局（東南戦）の設定。
    pub const fn tenhou_hanchan(disable_progress_bar: bool) -> Self {
        Self {
            length: 8,                       // 半荘戦（東南戦）
            init_scores: [25000; 4],         // 各プレイヤー25000点スタート
            disable_progress_bar,            // プログレスバー表示フラグ
        }
    }

    /// バッチゲームを実行するメインメソッド。
    /// 複数のゲームを同時に実行し、エージェントのバッチ処理を最適化する。
    pub fn run(
        &self,
        agents: &mut [Box<dyn BatchAgent>],     // バッチエージェントの配列
        indexes: &[[Index; 4]],                 // 各ゲームのプレイヤーインデックスマッピング
        seeds: &[(u64, u64)],                   // 各ゲームのシード値
    ) -> Result<Vec<GameResult>> {
        ensure!(!agents.is_empty());            // エージェントが存在することを確認
        ensure!(!indexes.is_empty());           // インデックスが存在することを確認
        ensure!(                                // インデックスとシードの数が一致することを確認
            indexes.len() == seeds.len(),
            "expected `indexes.len() == seeds.len()`, got {} and {}",
            indexes.len(),
            seeds.len(),
        );

        let mut games = indexes                 // ゲームの初期化
            .iter()
            .zip(seeds)                         // インデックスとシードをペアに
            .enumerate()                        // ゲーム番号を追加
            .map(|(game_idx, (idxs, &seed))| {
                let mut oracle_obs_versions = [None; 4];              // Oracleバージョン配列
                for (i, idx) in idxs.iter().enumerate() {             // 各プレイヤーに対して
                    agents[idx.agent_idx].start_game(idx.player_id_idx)?;  // ゲーム開始を通知
                    oracle_obs_versions[i] = agents[idx.agent_idx].oracle_obs_version();  // Oracleバージョンを取得
                }

                let game = Box::new(Game {      // ゲーム構造体を作成
                    length: self.length,        // ゲーム長
                    seed,                       // シード値
                    indexes: *idxs,             // インデックスマッピング
                    scores: self.init_scores,   // 初期点数
                    oracle_obs_versions,        // Oracleバージョン
                    ..Default::default()        // その他はデフォルト
                });
                Ok((game_idx, game))            // ゲーム番号とゲームを返す
            })
            .collect::<Result<Vec<_>>>()?;      // 結果を収集

        let mut game_results = vec![GameResult::default(); games.len()];  // 結果格納用ベクタ
        let mut to_remove = vec![];             // 削除対象ゲームのインデックス
        let mut cycles = 0;                     // サイクル数（ポーリング回数）
        let mut actions = 0;                    // アクション数（ゲーム毎の処理回数）

        let bar = if self.disable_progress_bar {                          // プログレスバーの設定
            ProgressBar::hidden()                                          // 非表示
        } else {
            ProgressBar::new(games.len() as u64)                          // ゲーム数で初期化
        };
        const TEMPLATE: &str =                                             // プログレスバーのテンプレート
            "{spinner:.cyan} {msg}\n[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
        let style = ProgressStyle::with_template(TEMPLATE)?               // スタイル設定
            .tick_chars(".oO°Oo*")                                        // スピナー文字
            .progress_chars("#-");                                        // プログレスバー文字
        bar.set_style(style);
        bar.enable_steady_tick(Duration::from_millis(150));               // 150msごとに更新

        while !games.is_empty() {               // 全ゲームが終了するまで
            for (_, game) in &mut games {       // 各ゲームに対して
                game.poll(agents)?;             // ポーリング（シーン設定）
            }

            for (idx_for_rm, (game_idx, game)) in games.iter_mut().enumerate() {  // 各ゲームに対して
                if let Some(game_result) = game.commit(agents)? {                 // コミット（リアクション取得）
                    game_results[*game_idx] = game_result;                        // 結果を保存
                    to_remove.push(idx_for_rm);                                   // 削除対象に追加
                }
            }

            for idx_for_rm in to_remove.drain(..).rev() {                         // 終了したゲームを削除
                games.swap_remove(idx_for_rm);                                    // 順序を保つ必要がないのでswap_remove
                bar.inc(1);                                                        // プログレスバーを進める
            }

            cycles += 1;                        // サイクル数を増やす
            actions += games.len();             // アクション数を追加

            let secs = bar.elapsed().as_secs_f64();                               // 経過時間
            bar.set_message(format!(                                               // プログレスメッセージを更新
                "cycles: {cycles} ({:.3} cycle/s), actions: {actions} ({:.3} action/s)",
                cycles as f64 / secs,           // サイクル/秒
                actions as f64 / secs,          // アクション/秒
            ));
        }
        bar.abandon();                          // プログレスバーを破棄（最終状態を保持）

        Ok(game_results)                        // ゲーム結果を返す
    }
}

/// テストモジュール
#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::Tsumogiri;               // ツモ切りエージェント（テスト用）

    /// ツモ切りエージェントでのバッチゲーム実行テスト。
    /// 2つのバッチエージェントが合計2ゲームを同時に処理する。
    #[test]
    fn tsumogiri() {
        let g = BatchGame::tenhou_hanchan(true);                                  // プログレスバーなしの半荘戦
        let mut agents = [                                                         // 2つのバッチエージェント
            Box::new(Tsumogiri::new_batched(&[0, 1, 2, 3]).unwrap()) as _,        // エージェント0：4人分担当
            Box::new(Tsumogiri::new_batched(&[3, 2, 1, 0]).unwrap()) as _,        // エージェント1：4人分担当
        ];
        let indexes = &[                        // 2ゲーム分のインデックスマッピング
            [                                   // ゲーム0
                Index {                         // プレイヤー0
                    agent_idx: 0,               // エージェント0が担当
                    player_id_idx: 0,           // エージェント内でID0
                },
                Index {                         // プレイヤー1
                    agent_idx: 0,               // エージェント0が担当
                    player_id_idx: 1,           // エージェント内でID1
                },
                Index {                         // プレイヤー2
                    agent_idx: 1,               // エージェント1が担当
                    player_id_idx: 1,           // エージェント内でID1
                },
                Index {                         // プレイヤー3
                    agent_idx: 1,               // エージェント1が担当
                    player_id_idx: 0,           // エージェント内でID0
                },
            ],
            [                                   // ゲーム1
                Index {                         // プレイヤー0
                    agent_idx: 1,               // エージェント1が担当
                    player_id_idx: 3,           // エージェント内でID3
                },
                Index {                         // プレイヤー1
                    agent_idx: 1,               // エージェント1が担当
                    player_id_idx: 2,           // エージェント内でID2
                },
                Index {                         // プレイヤー2
                    agent_idx: 0,               // エージェント0が担当
                    player_id_idx: 2,           // エージェント内でID2
                },
                Index {                         // プレイヤー3
                    agent_idx: 0,               // エージェント0が担当
                    player_id_idx: 3,           // エージェント内でID3
                },
            ],
        ];

        g.run(&mut agents, indexes, &[(1009, 0), (1021, 0)])                      // 2ゲーム実行
            .unwrap();                                                             // エラーなく終了することを確認
    }
}
