// 2対2対戦モードの実装。
// 2人のチャレンジャーが2人のチャンピオンと対戦する。
// チーム対戦のような形式で、対角の座席配置で評価する。

use super::game::{BatchGame, Index};                         // バッチゲームとインデックス型
use super::result::GameResult;                               // ゲーム結果構造体
use crate::agent::{AkochanAgent, BatchAgent, new_py_agent};  // エージェント関連
use std::fs::{self, File};                                   // ファイルシステム操作
use std::io;                                                 // 入出力操作
use std::iter;                                               // イテレータユーティリティ
use std::path::PathBuf;                                      // パス操作
use std::time::Duration;                                     // 時間間隔

use anyhow::Result;                                          // エラーハンドリング
use flate2::Compression;                                     // Gzip圧縮レベル
use flate2::read::GzEncoder;                                 // Gzipエンコーダ
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};  // プログレスバー
use pyo3::prelude::*;                                        // Pythonバインディング
use rayon::prelude::*;                                       // 並列処理

/// 2対2対戦モードを管理する構造体。
/// Pythonからも使用可能。
#[pyclass]
#[derive(Clone, Default)]
pub struct TwoVsTwo {
    pub disable_progress_bar: bool,              // プログレスバーを無効にするか
    pub log_dir: Option<String>,                 // ログファイルの出力ディレクトリ
}

#[pymethods]
impl TwoVsTwo {
    /// Pythonからのコンストラクタ。
    /// キーワード引数のみを受け付ける。
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,                // プログレスバー表示設定
            log_dir,                             // ログディレクトリ設定
        }
    }

    /// Pythonエージェント同士の2対2対戦。
    /// 結果はログファイルに出力される。
    pub fn py_vs_py(
        &self,
        challenger: PyObject,            // チャレンジャーのPythonエージェント
        champion: PyObject,              // チャンピオンのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数（2ゲームが1セット）
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<()> {
        // `allow_threads`が必要。これがないとPythonGCがブロックされ、
        // 長時間タスクでメモリリークが発生する。
        py.allow_threads(move || {
            self.run_batch(                                        // バッチ実行
                |player_ids| new_py_agent(challenger, player_ids), // チャレンジャーエージェント作成
                |player_ids| new_py_agent(champion, player_ids),   // チャンピオンエージェント作成
                seed_start,
                seed_count,
            )?;
            Ok(())                                                 // 結果はログに出力される
        })
    }

    /// Akochan（チャレンジャー）対Pythonエージェント（チャンピオン）の2対2対戦。
    pub fn ako_vs_py(
        &self,
        engine: PyObject,                // チャンピオンのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<()> {
        py.allow_threads(move || {       // GILを解放して並列処理を可能に
            self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),  // Akochanエージェント作成
                |player_ids| new_py_agent(engine, player_ids),                                 // Pythonエージェント作成
                seed_start,
                seed_count,
            )?;
            Ok(())
        })
    }

    /// Pythonエージェント（チャレンジャー）対Akochan（チャンピオン）の2対2対戦。
    pub fn py_vs_ako(
        &self,
        engine: PyObject,                // チャレンジャーのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<()> {
        py.allow_threads(move || {       // GILを解放して並列処理を可能に
            self.run_batch(
                |player_ids| new_py_agent(engine, player_ids),                                 // Pythonエージェント作成
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),  // Akochanエージェント作成
                seed_start,
                seed_count,
            )?;
            Ok(())
        })
    }

    /// Pythonエージェント（チャレンジャー）対Akochan（チャンピオン）の単一ゲーム対戦。
    /// 特定のスプリット（座席配置）で実行する。
    pub fn py_vs_ako_one(
        &self,
        engine: PyObject,                // チャレンジャーのPythonエージェント
        seed: (u64, u64),                // シード値
        split: usize,                    // スプリット番号（0または1）
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<()> {
        py.allow_threads(move || {       // GILを解放して並列処理を可能に
            self.run_one(                // 単一ゲーム実行
                |player_ids| new_py_agent(engine, player_ids),                                 // Pythonエージェント作成
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),  // Akochanエージェント作成
                seed,
                split,
            )?;
            Ok(())
        })
    }
}

impl TwoVsTwo {
    /// 2対2対戦のバッチ実行。
    /// チャレンジャーチームとチャンピオンチームが対角の座席で対戦する。
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,         // チャレンジャーエージェントのファクトリ関数
        new_champion_agent: M,           // チャンピオンエージェントのファクトリ関数
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数（1セット=2ゲーム）
    ) -> Result<Vec<GameResult>>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,  // チャレンジャーファクトリの型制約
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,  // チャンピオンファクトリの型制約
    {
        if let Some(dir) = &self.log_dir {              // ログディレクトリが指定されている場合
            fs::create_dir_all(dir)?;                    // ディレクトリを作成
        }

        log::info!(                                      // 実行情報をログ出力
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} hanchans",
            seed_start.0,                                // 開始シード
            seed_start.0 + seed_count,                   // 終了シード（排他）
            seed_start.1,                                // キー値
            seed_count,                                  // セット数
            seed_count * 2,                              // 総ゲーム数（半荘数）
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)  // シード列の生成
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), 2))  // 同じシードを2回繰り返す
            .collect();

        let challenger_player_ids_per_seed = [   // チャレンジャーの座席配置パターン
            0, 2, // スプリットA：座席0と2（対角）
            1, 3, // スプリットB：座席1と3（対角）
        ];
        let challenger_player_ids: Vec<_> = challenger_player_ids_per_seed  // チャレンジャーのプレイヤーID列
            .into_iter()
            .cycle()                                                        // パターンを繰り返す
            .take(seed_count as usize * challenger_player_ids_per_seed.len())  // 必要数だけ取る
            .collect();

        let champion_player_ids_per_seed = [     // チャンピオンの座席配置パターン
            1, 3, // スプリットA：座席1と3
            0, 2, // スプリットB：座席0と2
        ];
        let champion_player_ids: Vec<_> = champion_player_ids_per_seed  // チャンピオンのプレイヤーID列
            .into_iter()
            .cycle()                                                     // パターンを繰り返す
            .take(seed_count as usize * champion_player_ids_per_seed.len())  // 必要数だけ取る
            .collect();

        let mut agents = [                                       // バッチエージェント配列
            new_challenger_agent(&challenger_player_ids)?,       // チャレンジャーエージェントを作成
            new_champion_agent(&champion_player_ids)?,           // チャンピオンエージェントを作成
        ];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);  // 天鳳ルール半荘戦

        let mut challenger_idx = 0;                              // チャレンジャーのインデックス
        let mut champion_idx = 0;                                // チャンピオンのインデックス
        let agent_idxs_per_seed = [                              // 各スプリットのエージェント配置
            [0, 1, 0, 1], // スプリットA：チャレンジャーとチャンピオンが交互
            [1, 0, 1, 0], // スプリットB：チャンピオンとチャレンジャーが交互
        ];
        let indexes: Vec<_> = agent_idxs_per_seed                // インデックスマッピングを生成
            .into_iter()
            .cycle()                                             // パターンを繰り返す
            .take(seed_count as usize * agent_idxs_per_seed.len())  // 必要数だけ取る
            .map(|agent_idxs_per_split| {
                agent_idxs_per_split.map(|agent_idx| {          // 各座席のエージェント情報を生成
                    let player_id_idx = if agent_idx == 0 {     // エージェント0（チャレンジャー）の場合
                        &mut challenger_idx
                    } else {                                     // エージェント1（チャンピオン）の場合
                        &mut champion_idx
                    };
                    let ret = Index {                            // インデックス情報を作成
                        agent_idx,                               // エージェント番号
                        player_id_idx: *player_id_idx,           // エージェント内のプレイヤーID
                    };
                    *player_id_idx += 1;                         // 次のプレイヤーIDへ
                    ret
                })
            })
            .collect();

        let results = batch_game.run(&mut agents, &indexes, &seeds)?;  // バッチゲームを実行

        if let Some(dir) = &self.log_dir {                             // ログ出力が有効な場合
            log::info!("dumping game logs");                           // ログダンプ開始を通知

            let bar = if self.disable_progress_bar {                    // プログレスバーの設定
                ProgressBar::hidden()                                   // 非表示
            } else {
                ProgressBar::new(seed_count * 2)                        // 総ゲーム数で初期化
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));         // 150msごとに更新

            results
                .par_iter()                                             // 並列イテレータ
                .progress_with(bar)                                     // プログレスバーをアタッチ
                .enumerate()
                .try_for_each(|(i, game_result)| {                     // 各ゲーム結果に対して
                    let split_name = ["a", "b"][i % 2];                // スプリット名（aまたはb）
                    let (seed, key) = game_result.seed;                // シード値を分解
                    let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]  // ファイル名
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log()?;             // JSONログを生成
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());  // Gzip圧縮（最高レベル）
                    let mut f = File::create(filename)?;                // ファイルを作成
                    io::copy(&mut comp, &mut f)?;                       // 圧縮データを書き込み

                    anyhow::Ok(())                                      // 成功
                })?;
        }

        Ok(results)                                                     // ゲーム結果を返す
    }

    /// 2対2対戦の単一ゲーム実行。
    /// 指定したスプリット（座席配置）で1ゲームのみ実行する。
    pub fn run_one<C, M>(
        &self,
        new_challenger_agent: C,         // チャレンジャーエージェントのファクトリ関数
        new_champion_agent: M,           // チャンピオンエージェントのファクトリ関数
        seed: (u64, u64),                // シード値
        split: usize,                    // スプリット番号（0..2の範囲内で指定）
    ) -> Result<GameResult>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,  // チャレンジャーファクトリの型制約
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,  // チャンピオンファクトリの型制約
    {
        if let Some(dir) = &self.log_dir {              // ログディレクトリが指定されている場合
            fs::create_dir_all(dir)?;                    // ディレクトリを作成
        }

        log::info!(                                      // 実行情報をログ出力
            "seed: {} w/ {:#x}, split: {}, start 1 hanchan",
            seed.0,                                      // シード値
            seed.1,                                      // キー値
            split                                        // スプリット番号
        );

        let challenger_player_ids = if split == 0 { [0, 2] } else { [1, 3] };  // チャレンジャーの座席（対角）
        let champion_player_ids = if split == 0 { [1, 3] } else { [0, 2] };    // チャンピオンの座席（対角）

        let mut agents = [                                       // バッチエージェント配列
            new_challenger_agent(&challenger_player_ids)?,       // チャレンジャーエージェントを作成
            new_champion_agent(&champion_player_ids)?,           // チャンピオンエージェントを作成
        ];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);  // 天鳳ルール半荘戦

        let indexes = if split == 0 {                            // スプリット0の場合
            [[
                Index {                                          // 座席0：チャレンジャー
                    agent_idx: 0,                                // エージェント0
                    player_id_idx: 0,                            // プレイヤーID 0
                },
                Index {                                          // 座席1：チャンピオン
                    agent_idx: 1,                                // エージェント1
                    player_id_idx: 0,                            // プレイヤーID 0
                },
                Index {                                          // 座席2：チャレンジャー
                    agent_idx: 0,                                // エージェント0
                    player_id_idx: 1,                            // プレイヤーID 1
                },
                Index {                                          // 座席3：チャンピオン
                    agent_idx: 1,                                // エージェント1
                    player_id_idx: 1,                            // プレイヤーID 1
                },
            ]]
        } else {                                                 // スプリット1の場合
            [[
                Index {                                          // 座席0：チャンピオン
                    agent_idx: 1,                                // エージェント1
                    player_id_idx: 0,                            // プレイヤーID 0
                },
                Index {                                          // 座席1：チャレンジャー
                    agent_idx: 0,                                // エージェント0
                    player_id_idx: 0,                            // プレイヤーID 0
                },
                Index {                                          // 座席2：チャンピオン
                    agent_idx: 1,                                // エージェント1
                    player_id_idx: 1,                            // プレイヤーID 1
                },
                Index {                                          // 座席3：チャレンジャー
                    agent_idx: 0,                                // エージェント80
                    player_id_idx: 1,                            // プレイヤーID 1
                },
            ]]
        };

        let results = batch_game.run(&mut agents, &indexes, &[seed])?;  // シングルゲームを実行

        if let Some(dir) = &self.log_dir {                             // ログ出力が有効な場合
            log::info!("dumping game logs");                           // ログダンプ開始を通知

            let split_name = ["a", "b"][split];                        // スプリット名
            let (seed, key) = seed;                                    // シード値を分解
            let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]  // ファイル名
                .iter()
                .collect();

            let log = results[0].dump_json_log()?;                     // JSONログを生成
            let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());  // Gzip圧縮（最高レベル）
            let mut f = File::create(filename)?;                       // ファイルを作成
            io::copy(&mut comp, &mut f)?;                              // 圧縮データを書き込み
        }

        Ok(results.into_iter().next().unwrap())                        // 最初（唯一）の結果を返す
    }
}
