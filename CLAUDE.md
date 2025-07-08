# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mortal is a Japanese mahjong AI powered by deep reinforcement learning. The project consists of:
- **libriichi**: Fast Rust-based mahjong emulator and game logic
- **mortal**: Python package for neural network training and AI implementation

## Architecture

### Core Components

1. **libriichi (Rust)**
   - Game engine and emulator (up to 40K games/hour)
   - Algorithms: shanten calculation, agari detection, point calculation
   - Arena for game simulations (1v3, 2v2)
   - Python bindings via PyO3
   - MJAI protocol implementation

2. **mortal (Python)**
   - Neural network models (Brain, DQN, GRP)
   - Training infrastructure using PyTorch
   - Game client/server implementation
   - Reward calculation system

### Key Design Patterns
- State representation uses observation encoding for neural network input
- Action space includes discard, riichi, chi/pon/kan, and agari decisions
- Uses Conservative Q-Learning (CQL) for offline reinforcement learning
- Supports both stochastic and deterministic play modes

## Development Commands

### Setup
```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Create conda environment
conda env create -f environment.yml
conda activate mortal

# Install PyTorch (adjust for your CUDA version)
pip install torch
```

### Build
```bash
# Build libriichi with Python bindings
cargo build -p libriichi --lib --release

# Copy to Python module (Linux)
cp target/release/libriichi.so mortal/libriichi.so

# Build utilities
cargo build -p libriichi --bins --no-default-features --release
cargo build -p exe-wrapper --release
```

### Test
```bash
# Run all tests
cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture

# Run benchmarks
cargo test -p libriichi --no-default-features --bench bench
```

### Lint
```bash
# Rust formatting
cargo fmt

# Rust linting
cargo clippy
```

### Run
```bash
# Run AI player (from mortal directory)
cd mortal
python mortal.py <PLAYER_ID>  # PLAYER_ID is 0-3

# Training (requires config file)
python train.py
python train_grp.py
python one_vs_three.py
```

## Configuration

Copy `mortal/config.example.toml` and customize:
- Model paths (`state_file`, `best_state_file`)
- Dataset paths (`globs`)
- Training hyperparameters
- Device settings (`cuda:0` or `cpu`)

## Important Files

- `libriichi/src/state/mod.rs` - Game state management
- `libriichi/src/algo/` - Core mahjong algorithms
- `mortal/model.py` - Neural network architecture
- `mortal/train.py` - Main training loop
- `mortal/engine.py` - Game engine interface

## Environment Variables

- `MORTAL_REVIEW_MODE=1` - Enable review mode for analysis
- `TRAIN_PLAY_PROFILE` - Select training play configuration profile

## Testing Guidelines

When adding new features:
1. Add unit tests in the same module (Rust: `#[cfg(test)]`, Python: test files)
2. Test game logic changes with `cargo test` in libriichi
3. Test Python changes by running sample games with the AI

## Common Tasks

### Adding a new mahjong rule
1. Implement logic in `libriichi/src/state/` or `libriichi/src/algo/`
2. Update state representation if needed
3. Add tests to verify rule behavior
4. Update Python bindings if the rule affects the interface

### Modifying the neural network
1. Update architecture in `mortal/model.py`
2. Adjust input/output dimensions if changed
3. Update config example with new hyperparameters
4. Test with small-scale training before full runs

### Debugging game logic
1. Use `validate_logs` binary to check game logs
2. Enable debug logging in game state
3. Use the log-viewer web interface for visual debugging