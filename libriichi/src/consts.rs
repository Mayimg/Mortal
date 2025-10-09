use crate::py_helper::add_submodule;

use pyo3::prelude::*;

pub const MAX_VERSION: u32 = 4;

pub const ACTION_SPACE: usize = 37 // discard | kan (choice)
                              + 1  // riichi
                              + 3  // chi
                              + 1  // pon
                              + 1  // kan (decide)
                              + 1  // agari
                              + 1  // ryukyoku
                              + 1; // pass
// = 46
// GRP feature size
//  - 7 base features: [grand_kyoku, honba, kyotaku, score0/1e4, score1/1e4, score2/1e4, score3/1e4]
//  - 16 rank one-hot (4 players x 4 ranks)
//  - 168 agari-improvement features (4 players x 42 patterns)
pub const GRP_SIZE: usize = 7 + 16 + 168; // = 191

#[pyfunction]
#[inline]
pub const fn obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (938, 34),
        2 => (942, 34),
        3 => (934, 34),
        4 => (1012, 34),
        _ => unreachable!(),
    }
}

#[pyfunction]
#[inline]
pub const fn oracle_obs_shape(version: u32) -> (usize, usize) {
    match version {
        1 => (211, 34),
        2 | 3 | 4 => (217, 34),
        _ => unreachable!(),
    }
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "consts")?;
    m.add_function(wrap_pyfunction!(obs_shape, &m)?)?;
    m.add_function(wrap_pyfunction!(oracle_obs_shape, &m)?)?;
    m.add("MAX_VERSION", MAX_VERSION)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("GRP_SIZE", GRP_SIZE)?;
    add_submodule(py, prefix, super_mod, &m)
}
