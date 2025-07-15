use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{AkochanAgent, BatchAgent, new_py_agent};
use std::fs::{self, File};
use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Duration;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use anyhow::Result;
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct OneVsThreeNoDuplicate {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}

#[pymethods]
impl OneVsThreeNoDuplicate {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }

    /// Returns the rankings of the challenger.
    pub fn py_vs_py(
        &self,
        challenger: PyObject,
        champion: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; 4]> {
        // `allow_threads` is required, otherwise it will block python GC to
        // run, leading to memory leaks, since this function is doing long
        // tasks.
        py.allow_threads(move || {
            let (results, challenger_positions) = self.run_batch(
                |player_ids| new_py_agent(challenger, player_ids),
                |player_ids| new_py_agent(champion, player_ids),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];
            for (i, result) in results.iter().enumerate() {
                let chal_pos = challenger_positions[i];
                let rank = result.rankings().rank_by_player[chal_pos as usize];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }

    /// Returns the rankings of the challenger (akochan in this case).
    pub fn ako_vs_py(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {
            let (results, challenger_positions) = self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                |player_ids| new_py_agent(engine, player_ids),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];
            for (i, result) in results.iter().enumerate() {
                let chal_pos = challenger_positions[i];
                let rank = result.rankings().rank_by_player[chal_pos as usize];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }

    /// Returns the rankings of the challenger (python agent in this case).
    pub fn py_vs_ako(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {
            let (results, challenger_positions) = self.run_batch(
                |player_ids| new_py_agent(engine, player_ids),
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];
            for (i, result) in results.iter().enumerate() {
                let chal_pos = challenger_positions[i];
                let rank = result.rankings().rank_by_player[chal_pos as usize];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }
}

impl OneVsThreeNoDuplicate {
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<(Vec<GameResult>, Vec<u8>)>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} hanchans",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
            seed_count,
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)
            .map(|seed| (seed, seed_start.1))
            .collect();

        // Generate random challenger positions for each seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed_start.1);
        let challenger_positions: Vec<u8> = (0..seed_count)
            .map(|_| rng.gen_range(0..4))
            .collect();

        // Create player IDs based on challenger positions
        let challenger_player_ids: Vec<u8> = challenger_positions.clone();
        
        let champion_player_ids: Vec<u8> = challenger_positions
            .iter()
            .flat_map(|&chal_pos| {
                (0..4u8)
                    .filter(|&pos| pos != chal_pos)
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut agents = [
            new_challenger_agent(&challenger_player_ids)?,
            new_champion_agent(&champion_player_ids)?,
        ];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);

        let mut challenger_idx = 0;
        let mut champion_idx = 0;
        
        let indexes: Vec<_> = challenger_positions
            .iter()
            .map(|&chal_pos| {
                let mut idxs = [Index { agent_idx: 0, player_id_idx: 0 }; 4];
                
                // Set challenger index
                idxs[chal_pos as usize] = Index {
                    agent_idx: 0,
                    player_id_idx: challenger_idx,
                };
                challenger_idx += 1;
                
                // Set champion indices for other positions
                for pos in 0..4 {
                    if pos != chal_pos as usize {
                        idxs[pos] = Index {
                            agent_idx: 1,
                            player_id_idx: champion_idx,
                        };
                        champion_idx += 1;
                    }
                }
                idxs
            })
            .collect();

        let results = batch_game.run(&mut agents, &indexes, &seeds)?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(seed_count)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .enumerate()
                .try_for_each(|(i, game_result)| {
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}.json.gz")]
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;

                    anyhow::Ok(())
                })?;
        }

        Ok((results, challenger_positions))
    }
}
