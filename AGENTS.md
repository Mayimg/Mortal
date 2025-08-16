# Repository Guidelines

## Project Structure & Module Organization
- `libriichi/`: Rust core and Python extension (PyO3) for mahjong logic and datasets.
- `mortal/`: Python training code and runtime (e.g., `run_train.py`, `train.py`, configs, logs).
- `exe-wrapper/`: Small Rust helper binary used by the workspace.
- `docs/`: Static site sources; not required for local dev.
- `log-viewer/`: HTML viewer for logs and samples.
- `data/`, `models/`, `logs/`: Datasets, checkpoints, and training outputs.
- Root configs: `config.toml`, `config_main.toml`, `environment.yml`, `Dockerfile`.

## Build, Test, and Development Commands
- Rust build: `cargo build --release` — builds the workspace (`libriichi`, `exe-wrapper`).
- Lint Rust: `cargo clippy --all-targets --all-features -- -D warnings` — enforce warnings as errors.
- Format Rust: `cargo fmt --all` — standard rustfmt.
- Python env: `conda env create -f environment.yml && conda activate mortal` — set up dependencies.
- Train locally: `python -m mortal.run_train` — starts training using `config.toml`.
- Check data: `python check_train_data.py` — quick dataset sanity checks.
- Docker (alt build): see `BUILDING_LIBRIICHI_FOR_DOCKER.md`.

## Coding Style & Naming Conventions
- Rust: follow rustfmt; keep Clippy clean. Prefer explicit types and small modules. Public APIs live under `libriichi/src/*`.
- Python: PEP 8, 4‑space indent, snake_case for functions/vars, PascalCase for classes; add type hints where practical.
- Files: Python modules under `mortal/` and Rust modules under `libriichi/src/`. Keep scripts executable if they are entrypoints.

## Testing Guidelines
- Rust: add unit tests inline via `#[cfg(test)]` and run with `cargo test`. Benches live under `libriichi/benches` (`cargo bench`).
- Python: prefer `pytest` with tests under `mortal/tests/`. Target critical data loaders and training loops.
- Aim for coverage on core logic (state, dataset, arena) before adding features.

## Commit & Pull Request Guidelines
- Commits: imperative mood, scoped messages (e.g., `libriichi: fix tile scoring`), small and focused.
- PRs: include description, rationale, and how to verify (commands or sample inputs). Link related issues.
- Checklist: run `cargo fmt`, `cargo clippy`, and a local train smoke test. Update docs or examples if behavior changes.

## Security & Configuration Tips
- Do not commit large logs or model files; keep them in `logs/` or `models/` locally.
- Configuration lives in TOML files; prefer adding new keys over repurposing existing ones. 
