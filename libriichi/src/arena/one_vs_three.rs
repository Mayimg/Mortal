// 1対3対戦モードの実装。
// 1人のチャレンジャーが3人のチャンピオンと対戦する。
// チャレンジャーは4つの異なる座席位置で対戦し、実力を公平に評価する。

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

/// 1対3対戦モードを管理する構造体。
/// Pythonからも使用可能。
#[pyclass]
#[derive(Clone, Default)]
pub struct OneVsThree {
    pub disable_progress_bar: bool,              // プログレスバーを無効にするか
    pub log_dir: Option<String>,                 // ログファイルの出力ディレクトリ
}

#[pymethods]
impl OneVsThree {
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

    /// Pythonエージェント同士の対戦。
    /// チャレンジャーの順位分布を返す。
    pub fn py_vs_py(
        &self,
        challenger: PyObject,            // チャレンジャーのPythonエージェント
        champion: PyObject,              // チャンピオンのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数（4ゲームが1セット）
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<[i32; 4]> {
        // `allow_threads`が必要。これがないとPythonGCがブロックされ、
        // 長時間タスクでメモリリークが発生する。
        py.allow_threads(move || {
            let results = self.run_batch(                          // バッチ実行
                |player_ids| new_py_agent(challenger, player_ids), // チャレンジャーエージェント作成
                |player_ids| new_py_agent(champion, player_ids),   // チャンピオンエージェント作成
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];                             // 順位分布配列
            for (i, result) in results.iter().enumerate() {        // 各ゲーム結果に対して
                let rank = result.rankings().rank_by_player[i % 4];  // チャレンジャーの順位を取得
                rankings[rank as usize] += 1;                      // 順位をカウント
            }
            Ok(rankings)                                           // [1位数, 2位数, 3位数, 4位数]
        })
    }

    /// Akochan（チャレンジャー）対Pythonエージェント（チャンピオン）の対戦。
    /// Akochanの順位分布を返す。
    pub fn ako_vs_py(
        &self,
        engine: PyObject,                // チャンピオンのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {       // GILを解放して並列処理を可能に
            let results = self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),  // Akochanエージェント作成
                |player_ids| new_py_agent(engine, player_ids),                                 // Pythonエージェント作成
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];                             // 順位分布配列
            for (i, result) in results.iter().enumerate() {        // 各ゲーム結果に対して
                let rank = result.rankings().rank_by_player[i % 4];  // Akochanの順位を取得
                rankings[rank as usize] += 1;                      // 順位をカウント
            }
            Ok(rankings)
        })
    }

    /// Pythonエージェント（チャレンジャー）対Akochan（チャンピオン）の対戦。
    /// Pythonエージェントの順位分布を返す。
    pub fn py_vs_ako(
        &self,
        engine: PyObject,                // チャレンジャーのPythonエージェント
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数
        py: Python<'_>,                  // Pythonインタープリタ
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {       // GILを解放して並列処理を可能に
            let results = self.run_batch(
                |player_ids| new_py_agent(engine, player_ids),                                 // Pythonエージェント作成
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),  // Akochanエージェント作成
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];                             // 順位分布配列
            for (i, result) in results.iter().enumerate() {        // 各ゲーム結果に対して
                let rank = result.rankings().rank_by_player[i % 4];  // Pythonエージェントの順位を取得
                rankings[rank as usize] += 1;                      // 順位をカウント
            }
            Ok(rankings)
        })
    }
}

impl OneVsThree {
    /// 1対3対戦のバッチ実行。
    /// チャレンジャーが4つの異なる座席位置でチャンピオンと対戦する。
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,         // チャレンジャーエージェントのファクトリ関数
        new_champion_agent: M,           // チャンピオンエージェントのファクトリ関数
        seed_start: (u64, u64),          // 開始シード値
        seed_count: u64,                 // セット数（1セット=4ゲーム）
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
            seed_count * 4,                              // 総ゲーム数（半荘数）
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)  // シード列の生成
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), 4))  // 同じシードを4回繰り返す
            .collect();

        let challenger_player_ids: Vec<_> = (0..4).cycle().take(seed_count as usize * 4).collect();  // チャレンジャーのプレイヤーID列

        let champion_player_ids_per_seed = [     // チャンピオンの座席配置パターン
            1, 2, 3, // スプリットA：チャレンジャーが座席0
            0, 2, 3, // スプリットB：チャレンジャーが座席1
            0, 1, 3, // スプリットC：チャレンジャーが座席2
            0, 1, 2, // スプリットD：チャレンジャーが座席3
        ];
        let champion_player_ids: Vec<_> = champion_player_ids_per_seed  // チャンピオンのプレイヤーID列
            .into_iter()
            .cycle()                                                    // パターンを繰り返す
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
            [0, 1, 1, 1], // スプリットA：座席0がチャレンジャー、他はチャンピオン
            [1, 0, 1, 1], // スプリットB：座席1がチャレンジャー
            [1, 1, 0, 1], // スプリットC：座席2がチャレンジャー
            [1, 1, 1, 0], // スプリットD：座席3がチャレンジャー
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
                ProgressBar::new(seed_count * 4)                        // 総ゲーム数で初期化
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));         // 150msごとに更新

            results
                .par_iter()                                             // 並列イテレータ
                .progress_with(bar)                                     // プログレスバーをアタッチ
                .enumerate()
                .try_for_each(|(i, game_result)| {                     // 各ゲーム結果に対して
                    let split_name = ["a", "b", "c", "d"][i % 4];    // スプリット名（a-d）
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
}
