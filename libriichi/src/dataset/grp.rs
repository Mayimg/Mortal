use crate::consts::GRP_SIZE;
use crate::algo::point::Point;
use crate::rankings::Rankings;
use crate::mjai::Event;
use crate::tu8;
use crate::vec_ops::vec_add_assign;
use std::fs::File;
use std::io;
use std::mem;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use serde_json as json;
use tinyvec::array_vec;

#[pyclass]
#[derive(Clone, Default)]
pub struct Grp {
    // [grand_kyoku, honba, kyotaku, [score[i] / 10000]] where i is player_id
    pub feature: Array2<f64>,
    pub rank_by_player: [u8; 4],
    pub final_scores: [i32; 4],
}

#[pymethods]
impl Grp {
    #[staticmethod]
    fn load_log(raw_log: &str) -> Result<Self> {
        let events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        Self::load_events(&events)
    }

    #[staticmethod]
    #[pyo3(name = "load_gz_log_files")]
    fn load_gz_log_files_py(gzip_filenames: Vec<PyBackedStr>) -> Result<Vec<Self>> {
        Self::load_gz_log_files(gzip_filenames)
    }

    /// Returns List[List[np.ndarray]]
    pub fn take_feature<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, mem::take(&mut self.feature))
    }
    pub const fn take_rank_by_player(&self) -> [u8; 4] {
        self.rank_by_player
    }
    pub const fn take_final_scores(&self) -> [i32; 4] {
        self.final_scores
    }
}

impl Grp {
    #[inline]
    pub fn len(&self) -> usize {
        self.feature.len_of(Axis(0))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn load_gz_log_files<V, S>(gzip_filenames: V) -> Result<Vec<Self>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        gzip_filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || {
                    let file = File::open(filename)?;
                    let gz = GzDecoder::new(file);
                    let raw = io::read_to_string(gz)?;
                    Self::load_log(&raw)
                };
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()
    }

    pub fn load_events(events: &[Event]) -> Result<Self> {
        let mut game_info = vec![];
        let mut rank_by_player_opt = None;
        let mut final_deltas = [0; 4];
        let mut final_scores = [0; 4];

        for ev in events.iter().rev() {
            match *ev {
                Event::Hora { deltas, .. } | Event::Ryukyoku { deltas, .. } => {
                    if rank_by_player_opt.is_none() {
                        let ds = deltas.context(
                            "invalid log: field `deltas` is required for Hora and Ryukyoku of AL",
                        )?;
                        vec_add_assign(&mut final_deltas, &ds);
                    }
                }
                Event::ReachAccepted { actor } => {
                    if rank_by_player_opt.is_none() {
                        final_deltas[actor as usize] -= 1000;
                    }
                }
                Event::StartKyoku {
                    bakaze,
                    kyoku,
                    honba,
                    kyotaku,
                    oya,
                    scores,
                    ..
                } => {
                    if rank_by_player_opt.is_none() {
                        final_scores = scores;
                        vec_add_assign(&mut final_scores, &final_deltas);

                        let rk = Rankings::new(final_scores);

                        // assume the sum of scores to be 100k
                        let sum: i32 = final_scores.iter().sum();
                        if sum < 100_000 {
                            final_scores[rk.player_by_rank[0] as usize] += 100_000 - sum;
                        }

                        rank_by_player_opt = Some(rk.rank_by_player);
                    }

                    let mut kyoku_info = array_vec!([_; GRP_SIZE]);
                    let grand_kyoku = match bakaze.as_u8() {
                        tu8!(E) => kyoku - 1,
                        tu8!(S) => 3 + kyoku,
                        _ => 7 + kyoku,
                    };
                    kyoku_info.push(grand_kyoku as f64);
                    kyoku_info.push(honba as f64);
                    kyoku_info.push(kyotaku as f64);
                    // assume player 0 is the oya at E1
                    kyoku_info.extend(scores.iter().map(|&score| score as f64 / 10000.));
                    // Rank one-hot (16 dims: player-major, rank 0..3)
                    let base_len = kyoku_info.len();
                    debug_assert_eq!(base_len, 7);
                    let rk_now = Rankings::new(scores);
                    // fill zeros then set ones
                    kyoku_info.extend((0..16).map(|_| 0f64));
                    for pid in 0..4usize {
                        let r = rk_now.rank_by_player[pid] as usize; // 0..3
                        let idx = base_len + pid * 4 + r;
                        kyoku_info[idx] = 1.0;
                    }

                    // Agari improvement vectors: 4 players x 42 (21 patterns x [tsumo, ron])
                    let mut buf = [0f64; 168];
                    compute_agari_improvements(scores, oya as usize, honba as i32, kyotaku as i32, &mut buf);
                    kyoku_info.extend(buf.into_iter());

                    assert_eq!(kyoku_info.len(), GRP_SIZE);

                    game_info.insert(0, kyoku_info);
                }
                _ => (),
            }
        }

        let rank_by_player =
            rank_by_player_opt.context("invalid log: no Hora or Ryukyoku after a StartKyoku")?;
        let shape = (game_info.len(), GRP_SIZE);
        let feature =
            Array::from_iter(game_info.into_iter().flatten()).into_shape_with_order(shape)?;

        Ok(Self {
            feature,
            rank_by_player,
            final_scores,
        })
    }
}

// 21 agari categories ignoring parent/child distinction.
// Represented by (fu, han) pairs; for mangan and above, han alone determines category.
// Ordering aligns with point.rs match groupings.
const AGARI_PATTERNS: [(u8, u8); 21] = [
    // Non-mangan representatives
    (40, 1), // same as (20,2)
    (40, 2), // same as (20,3) or (80,1)
    (40, 3), // same as (20,4) or (80,2)
    (50, 1), // same as (25,2)
    (50, 2), // same as (25,3) or (100,1)
    (50, 3), // same as (25,4) or (100,2)
    (30, 1),
    (60, 1), // same as (30,2)
    (60, 2), // same as (30,3)
    (60, 3), // same as (30,4)
    (70, 1),
    (70, 2),
    (90, 1),
    (90, 2),
    (110, 1),
    (110, 2),
    // Mangan and above (representatives)
    (30, 5),  // mangan
    (30, 6),  // haneman (6..=7)
    (30, 8),  // baiman (8..=10)
    (30, 11), // sanbaiman (11..=12)
    (30, 13), // yakuman (13..)
];

#[inline]
fn compute_agari_improvements(
    scores: [i32; 4],
    oya: usize,
    honba: i32,
    kyotaku: i32,
    out: &mut [f64; 168],
) {
    // baseline ranks by current scores
    let rk_now = Rankings::new(scores);
    // For each player 0..3, for each pattern index 0..20, push [tsumo, ron]
    for pid in 0..4usize {
        let is_oya = pid == oya;
        let mut write_base = pid * 42; // each player has 42 dims
        for &(fu, han) in &AGARI_PATTERNS {
            let pt = Point::calc(is_oya, fu, han);
            // tsumo improvement
            let tsumo_impr = tsumo_rank_improvement(scores, oya, honba, kyotaku, pid, pt);
            out[write_base] = tsumo_impr;
            write_base += 1;
            // ron improvement (averaged over 3 possible targets)
            let ron_impr = ron_rank_improvement_avg(scores, oya, honba, kyotaku, pid, pt);
            out[write_base] = ron_impr;
            write_base += 1;
        }
        debug_assert_eq!(write_base, (pid + 1) * 42);
        // Explicitly ensure non-negative (max with 0) already handled in helpers
        let _ = rk_now; // silence unused warnings if cfg changes
    }
}

#[inline]
fn tsumo_rank_improvement(
    scores: [i32; 4],
    oya: usize,
    honba: i32,
    kyotaku: i32,
    pid: usize,
    pt: Point,
) -> f64 {
    let mut new_scores = scores;
    // Winner delta: sum of payments + kyotaku + 300 per honba
    let winner_bonus = pt.tsumo_total(pid == oya) + kyotaku * 1000 + honba * 300;
    new_scores[pid] += winner_bonus;

    // Losers' payments; each payment includes +100 per honba
    if pid == oya {
        // oya tsumo: each other pays tsumo_ko + 100*honba
        for j in 0..4usize {
            if j == pid { continue; }
            new_scores[j] -= pt.tsumo_ko + honba * 100;
        }
    } else {
        // ko tsumo: oya pays tsumo_oya; others pay tsumo_ko; all +100*honba
        for j in 0..4usize {
            if j == pid { continue; }
            if j == oya {
                new_scores[j] -= pt.tsumo_oya + honba * 100;
            } else {
                new_scores[j] -= pt.tsumo_ko + honba * 100;
            }
        }
    }

    rank_improvement(scores, new_scores, pid)
}

#[inline]
fn ron_rank_improvement_avg(
    scores: [i32; 4],
    _oya: usize,
    honba: i32,
    kyotaku: i32,
    pid: usize,
    pt: Point,
) -> f64 {
    let mut sum = 0.0f64;
    let mut cnt = 0.0f64;
    for target in 0..4usize {
        if target == pid { continue; }
        let mut new_scores = scores;
        // Winner gets ron + kyotaku + 300*honba
        new_scores[pid] += pt.ron + kyotaku * 1000 + honba * 300;
        // Target loses ron + 300*honba
        new_scores[target] -= pt.ron + honba * 300;

        sum += rank_improvement(scores, new_scores, pid);
        cnt += 1.0;
    }
    if cnt > 0.0 { sum / cnt } else { 0.0 }
}

#[inline]
fn rank_improvement(old_scores: [i32; 4], new_scores: [i32; 4], pid: usize) -> f64 {
    let old = Rankings::new(old_scores).rank_by_player[pid] as i32;
    let new = Rankings::new(new_scores).rank_by_player[pid] as i32;
    let diff = (old - new).max(0) as f64;
    diff
}
