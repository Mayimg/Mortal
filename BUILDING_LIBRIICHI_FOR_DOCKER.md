# libriichi.so Docker互換ビルドガイド

## 問題の背景

mjai.appなどのDockerコンテナ環境でMortalのlibriichi.soを使用する際、以下の互換性問題が発生することがあります：

1. **GLIBCバージョンの不一致**
   - ホストシステム: GLIBC 2.39 (Ubuntu 24.04など)
   - Dockerコンテナ: GLIBC 2.35 (Ubuntu 22.04) または GLIBC 2.17 (CentOS 7ベース)
   
2. **Pythonバージョンの不一致**
   - ビルド環境: Python 3.11
   - 実行環境: Python 3.10

## 解決方法

manylinux2014 Dockerイメージを使用してビルドすることで、幅広い環境で動作するバイナリを作成できます。

## ビルド前の準備

### 1. Cargo.tomlの修正

プロジェクトのRust edition設定を調整します（古いコンパイラ対応のため）：

```toml
# /home/naoyuki/mahjong01/Mortal/Cargo.toml
[workspace]
resolver = "2"  # "3"から変更
members = [
    "libriichi",
    "exe-wrapper",
]

# libriichi/Cargo.toml と exe-wrapper/Cargo.toml
edition = "2021"  # "2024"から変更
```

### 2. auto-initialize機能の無効化

libriichi/Cargo.tomlのpyo3依存関係から`auto-initialize`機能を削除：

```toml
# 変更前
pyo3 = { version = "0.25", features = ["auto-initialize", "multiple-pymethods", "anyhow"] }

# 変更後
pyo3 = { version = "0.25", features = ["multiple-pymethods", "anyhow"] }
```

## ビルド手順

### 方法1: manylinux2014でPython 3.10用にビルド（推奨）

```bash
# 既存のビルドアーティファクトをクリーンアップ
docker run --rm -v /home/naoyuki/mahjong01/Mortal:/io -w /io quay.io/pypa/manylinux2014_x86_64 rm -rf /io/target

# ビルド実行
docker run --rm -v /home/naoyuki/mahjong01/Mortal:/io -w /io quay.io/pypa/manylinux2014_x86_64 bash -c "
/opt/python/cp310-cp310/bin/pip install maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
cd /io
/opt/python/cp310-cp310/bin/maturin build --release -m libriichi/Cargo.toml --compatibility manylinux2014 --interpreter /opt/python/cp310-cp310/bin/python
"

# ビルドされた.soファイルを抽出
unzip -j target/wheels/libriichi-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl "riichi/riichi.cpython-310-x86_64-linux-gnu.so" -d mortal/

# libriichi.soにリネーム
mv mortal/riichi.cpython-310-x86_64-linux-gnu.so mortal/libriichi.so
```

### 方法2: manylinux_2_28でPython 3.11用にビルド

Python 3.11環境向けには以下のコマンドを使用：

```bash
docker run --rm -v /home/naoyuki/mahjong01/Mortal:/io quay.io/pypa/manylinux_2_28_x86_64 bash -c "
cd /io
/opt/python/cp311-cp311/bin/pip install maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
cd /io/libriichi
/opt/python/cp311-cp311/bin/maturin build --release --compatibility manylinux_2_28
"
```

## ビルド結果の確認

ビルドされたlibriichi.soのGLIBC依存関係を確認：

```bash
# GLIBC依存関係の確認
objdump -T mortal/libriichi.so | grep GLIBC | awk '{print $5}' | sort -u

# 期待される出力（manylinux2014の場合）
# (GLIBC_2.12)
# (GLIBC_2.14)
# (GLIBC_2.15)
# ...最大でGLIBC_2.17まで
```

## トラブルシューティング

### 1. ImportError: undefined symbol: PyType_GetQualName

**原因**: Python 3.11でビルドされたライブラリをPython 3.10で使用しようとしている

**解決策**: 実行環境のPythonバージョンに合わせてビルド

### 2. version `GLIBC_2.XX' not found

**原因**: 新しいGLIBCでビルドされたライブラリを古い環境で実行

**解決策**: manylinuxイメージでビルド

### 3. error: The `auto-initialize` feature is enabled

**原因**: manylinux環境では静的リンクのみサポート

**解決策**: Cargo.tomlから`auto-initialize`機能を削除

## 互換性マトリックス

| Dockerイメージ | 最大GLIBC | Python | 用途 |
|--------------|----------|--------|------|
| manylinux2014 | 2.17 | 3.6-3.10 | CentOS 7ベースのコンテナ |
| manylinux_2_28 | 2.28 | 3.8-3.12 | より新しいLinux環境 |
| ホスト直接ビルド | システム依存 | システム依存 | 開発環境のみ |

## まとめ

Docker環境でMortalを使用する場合は、必ずmanylinuxイメージでビルドすることを推奨します。これにより、様々なLinux環境での互換性が確保されます。