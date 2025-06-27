# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview

Mortal (凡夫) is a free and open-source AI for Japanese mahjong, powered by deep reinforcement learning. It uses a hybrid architecture with Rust for performance-critical components and Python for neural network training and inference.

## Architecture

### Rust Components (`libriichi/`)
- Core mahjong emulator with Python bindings via PyO3
- Key modules:
  - `agent/`: AI agents (mortal, akochan, tsumogiri)
  - `algo/`: Core mahjong algorithms (agari, shanten, point calculations)
  - `arena/`: AI vs AI battle simulation
  - `dataset/`: Training data handling
  - `mjai/`: mjai protocol interface
  - `state/`: Game state management

### Python Components (`mortal/`)
- Main entry: `mortal.py`
- Training: `train.py`, `train_grp.py`
- Neural networks: `model.py` (Brain, DQN, AuxNet)
- Inference: `engine.py` (MortalEngine)
- Client-server: `client.py`, `server.py` for distributed training

## Common Development Commands

### Build Commands
```bash
# Build Rust library (required before using Python code)
cargo build -p libriichi --lib --release

# Copy built library to Python module (Linux)
cp target/release/libriichi.so mortal/libriichi.so

# Copy built library to Python module (Windows MSYS2)
cp target/release/riichi.dll mortal/libriichi.pyd

# Build executable utilities
cargo build -p libriichi --bins --no-default-features --release
cargo build -p exe-wrapper --release
```

### Test Commands
```bash
# Run all Rust tests
cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture

# Run benchmarks
cargo test -p libriichi --no-default-features --bench bench

# Run specific test
cargo test -p libriichi test_name -- --nocapture
```

### Lint and Format Commands
```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all --check

# Run clippy
cargo clippy --workspace --all-targets --all-features -- -Dwarnings
```

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate mortal

# Install PyO3 separately based on your hardware
# See https://pytorch.org/get-started/locally/
```

## Key Development Notes

1. **Workspace Structure**: This is a Cargo workspace with `libriichi` and `exe-wrapper` as members
2. **Python Bindings**: libriichi uses PyO3 for Python bindings - the Rust library must be built before Python code can import it
3. **mjai Protocol**: The AI communicates using the standard mjai protocol for mahjong AIs
4. **Performance**: The Rust implementation can achieve up to 40K hanchans/hour on modern hardware
5. **Testing**: Always run tests with `--no-default-features --features flate2/zlib` to avoid dependency issues

## Configuration

- Example config: `config.example.toml` (if it exists)
- Supports training, testing, and online play modes
- Neural network architectures are defined in `mortal/model.py`

## License

- Code: AGPL-3.0-or-later
- Assets: CC BY-SA 4.0