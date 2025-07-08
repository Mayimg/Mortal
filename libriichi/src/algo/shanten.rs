//! 向聴数（シャンテン数）計算モジュール
//!
//! このモジュールは麻雀の手牌が和了まであと何枚必要か（向聴数）を計算します。
//! 通常形（面子手）、七対子、国士無双の3つの和了形に対応しています。
//! -1が返された場合は和了形（テンパイ）を表します。
//!
//! tomohxx氏のC++実装をRustに移植したものです。
//!
//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//!
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>

use crate::tuz;                    // 牌IDをusizeに変換するマクロ
use std::io::prelude::*;           // I/Oトレイト
use std::sync::LazyLock;           // 静的遅延初期化

use flate2::read::GzDecoder;       // gzip圧縮データの展開

// 字牌向聴テーブルのサイズ（字牌の組み合わせパターン数）
const JIHAI_TABLE_SIZE: usize = 78_032;
// 数牌向聴テーブルのサイズ（数牌の組み合わせパターン数）
const SUHAI_TABLE_SIZE: usize = 1_940_777;

// 字牌向聴テーブル（初回アクセス時に圧縮データから読み込み）
static JIHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_jihai.bin.gz"),
        JIHAI_TABLE_SIZE,
    )
});
// 数牌向聴テーブル（初回アクセス時に圧縮データから読み込み）
static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

/// 圧縮された向聴テーブルデータを読み込んで展開
fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    // gzip圧縮データをデコード
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = vec![];
    gz.read_to_end(&mut raw).unwrap();

    // デコードされたデータをテーブル形式に変換
    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];  // 1エントリは10要素の配列
    for (i, b) in raw.into_iter().enumerate() {
        // 1バイトに2つの値（4ビットずつ）がパックされている
        entry[i * 2 % 10] = b & 0b1111;          // 下位4ビット
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;  // 上位4ビット
        // 5バイト（=10値）ごとに1エントリを構成
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);

    ret
}

/// 向聴テーブルの初期化を保証（テスト用）
pub fn ensure_init() {
    assert_eq!(JIHAI_TABLE.len(), JIHAI_TABLE_SIZE);
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

/// 数牌の向聴情報を統合
/// lhs: 現在の向聴テーブルエントリ
/// index: 追加する数牌パターンのインデックス
/// m: 現在の面子数（完成形は3または4）
fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    // テーブルから該当パターンの向聴情報を取得
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();

    // 面子とターツの組み合わせを考慮して最小向聴数を計算
    // lhs[0..5]: ターツ数別の向聴数
    // lhs[5..10]: 面子数別の向聴数
    for j in (5..=(5 + m)).rev() {
        // j個の面子を作るための最小向聴数を計算
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            // 既存のk個の面子と新規(j-k)個の面子の組み合わせ
            sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
        }
        lhs[j] = sht;
    }

    // ターツ数別の向聴数を更新
    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            // 既存のk個のターツと新規(j-k)個のターツの組み合わせ
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

/// 字牌の向聴情報を統合（字牌は刻子または雀頭にしかならない）
fn add_jihai(lhs: &mut [u8; 10], index: usize, m: usize) {
    // テーブルから該当パターンの向聴情報を取得
    let tab = JIHAI_TABLE.get(index).copied().unwrap_or_default();

    // 字牌は順子を作れないので、特定の面子数のみ更新
    let j = m + 5;
    let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
    }
    lhs[j] = sht;
}

/// 牌の枚数配列をテーブルインデックスに変換
/// 各牌種の枚数を0-4枚として5進数で表現
fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

/// 通常形（面子手）の向聴数を計算
///
/// # 引数
/// - `tiles`: 34種の牌の枚数配列
/// - `len_div3`: (手牌枚数 - 2) / 3 の値（通常は4、副露がある場合は減る）
///
/// # 戻り値
/// 向聴数（-1は和了形を表す）
#[must_use]
pub fn calc_normal(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    // 各色の向聴情報を統合
    // まず萬子（マンズ）の向聴情報を取得
    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    // 筒子（ピンズ）の向聴情報を統合
    add_suhai(&mut ret, sum_tiles(&tiles[9..2 * 9]), len_div3);
    // 索子（ソウズ）の向聴情報を統合
    add_suhai(&mut ret, sum_tiles(&tiles[2 * 9..3 * 9]), len_div3);
    // 字牌の向聴情報を統合
    add_jihai(&mut ret, sum_tiles(&tiles[3 * 9..]), len_div3);

    // 指定された面子数での向聴数を取得し、-1して返す
    // （テーブルは0を和了としているため-1で調整）
    (ret[5 + len_div3] as i8) - 1
}

/// 七対子の向聴数を計算
///
/// # 引数
/// - `tiles`: 34種の牌の枚数配列
///
/// # 戻り値
/// 向聴数（-1は和了形を表す）
#[must_use]
pub fn calc_chitoi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;   // 対子の数
    let mut kinds = 0;   // 手牌にある牌の種類数
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        if c >= 2 {
            pairs += 1;  // 2枚以上あれば対子としてカウント
        }
    });

    // 向聴数計算：7対子まであと何対子必要か
    // 手牌の種類が7未満の場合は、新しい牌を集める必要がある
    let redunct = 7_u8.saturating_sub(kinds) as i8;
    7 - pairs + redunct - 1
}

/// 国士無双の向聴数を計算
///
/// # 引数
/// - `tiles`: 34種の牌の枚数配列
///
/// # 戻り値
/// 向聴数（-1は和了形を表す）
#[must_use]
pub fn calc_kokushi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;   // 幺九牌の対子数
    let mut kinds = 0;   // 幺九牌の種類数

    // 幺九牌（13種）の所持状況をチェック
    tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C]
        .iter()
        .map(|&i| tiles[i])
        .filter(|&c| c > 0)
        .for_each(|c| {
            kinds += 1;
            if c >= 2 {
                pairs += 1;  // 幺九牌の対子
            }
        });

    // 向聴数計算：13種全てを集めて、どれか1つを対子にする
    let redunct = (pairs > 0) as i8;  // 既に対子がある場合は1手分短縮
    14 - kinds - redunct - 1
}

/// 全ての和了形を考慮して最小向聴数を計算
///
/// # 引数
/// - `tiles`: 34種の牌の枚数配列
/// - `len_div3`: (手牌枚数 - 2) / 3 の値
///
/// # 戻り値
/// 最小向聴数（-1は和了形を表す）
#[must_use]
pub fn calc_all(tiles: &[u8; 34], len_div3: u8) -> i8 {
    // まず通常形の向聴数を計算
    let mut shanten = calc_normal(tiles, len_div3);
    // 既に和了または副露がある場合は特殊形を考慮しない
    if shanten <= 0 || len_div3 < 4 {
        return shanten;
    }

    // 七対子の向聴数と比較
    shanten = shanten.min(calc_chitoi(tiles));
    if shanten > 0 {
        // まだ和了でない場合は国士無双も考慮
        shanten.min(calc_kokushi(tiles))
    } else {
        shanten
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;  // 手牌文字列から配列への変換関数

    /// 3n+1枚形（ツモって来た状態）のテスト
    #[test]
    fn calc_3n_plus_1() {
        // 1向聴：刻子が3つと一組の同じ牌4枚
        let tehai = hand("1111m 333p 222s 444z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        // 6向聴：全てバラバラ
        let tehai = hand("147m 258p 369s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 6);
        // 2向聴：1副露後
        let tehai = hand("468m 33346p 7s").unwrap();
        assert_eq!(calc_all(&tehai, 3), 2);
        // 4向聴：2副露後
        let tehai = hand("147m 258p 3s").unwrap();
        assert_eq!(calc_all(&tehai, 2), 4);
        // テンパイ：3副露後
        let tehai = hand("4455s").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        // テンパイ：4副露後（裸单騎）
        let tehai = hand("7z").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        // 3向聴：特殊形のテスト
        let tehai = hand("15559m 19p 19s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        // 2向聴：対子が多い形
        let tehai = hand("9999m 6677p 88s 355z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        // 1向聴：国士無双形
        let tehai = hand("19m 19p 159s 123456z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
    }

    /// 3n+2枚形（通常の手牌）のテスト
    #[test]
    fn calc_3n_plus_2() {
        // 3向聴
        let tehai = hand("2344456m 14p 127s 2z 7p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        // 2向聴
        let tehai = hand("2344456m 14p 127s 2z 5p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        // 2向聴
        let tehai = hand("344455667p 1139s 9m").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        // 1向聴
        let tehai = hand("344455667p 1139s 9p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        // テンパイ
        let tehai = hand("122334m 678p 37s 22z 5s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        // テンパイ
        let tehai = hand("122334m 678p 12s 22z 4s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        // 和了形（-1）
        let tehai = hand("12223456m 78889p 2m").unwrap();
        assert_eq!(calc_all(&tehai, 4), -1);
        // テンパイ：3副露後
        let tehai = hand("34778p").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        // テンパイ：4副露後
        let tehai = hand("34s").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        // 和了形：4副露後
        let tehai = hand("55m").unwrap();
        assert_eq!(calc_all(&tehai, 0), -1);
    }
}
