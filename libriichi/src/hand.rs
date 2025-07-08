//! 手牌形式変換モジュール
//! 
//! このモジュールは麻雀の手牌を様々な形式間で変換するためのユーティリティ関数を提供します。
//! 主な機能：
//! - 文字列形式（天鳳形式）から数値配列への変換
//! - 数値配列から文字列形式への変換
//! - 赤ドラ（赤5）の処理
//! - 34種類の牌と37種類（赤ドラ含む）の牌の相互変換
//!
//! 文字列形式について：
//! - 天鳳形式（tenhou.net/2）を使用：例: "123m 456p 789s 11z", "0m"（赤5萬）
//! - mjai形式ではない：例: "5mr", "ESW"は使用しない
//!
//! 配列形式について：
//! - 34要素配列：通常の34種類の牌（萬子9種、筒子9種、索子9種、字牌7種）
//! - 37要素配列：34種類＋赤ドラ3種（赤5萬、赤5筒、赤5索）
//!
//! Hand format conversions, usually only useful for testing and debugging.
//!
//! Note that all functions in this mod that take or produce strings are dealing
//! with tenhou.net/2 format tile description (like 0m 123z) instead of mjai (like
//! 5mr ESW).

// Tile型の定義をインポート（麻雀の牌を表す型）
use crate::tile::Tile;
// ベクトル演算用のヘルパー関数をインポート（配列の要素を別の配列に加算）
use crate::vec_ops::vec_add_assign;
// マクロをインポート
// - must_tile!: 数値からTile型を生成（エラー時はパニック）
// - tuz!: 牌のIDをusize型として取得（例: tuz!(5mr) = 34）
use crate::{must_tile, tuz};

// エラーハンドリング用のライブラリ
// - Result: 成功/失敗を表す型
// - bail!: エラーを返すマクロ
// - ensure!: 条件を満たさない場合にエラーを返すマクロ
use anyhow::{Result, bail, ensure};

/// 赤ドラを含む手牌文字列を37要素の配列に変換する関数
/// 
/// 引数：
/// - `s`: 天鳳形式の手牌文字列（例: "123m 0p 789s 11z"）
///   - 0は赤5を表す（0m=赤5萬、0p=赤5筒、0s=赤5索）
///   - スペース、タブ、改行は許可される
/// 
/// 戻り値：
/// - `Result<[u8; 37]>`: 各牌の枚数を表す37要素の配列
///   - インデックス0-33: 通常の34種類の牌
///   - インデックス34-36: 赤ドラ（赤5萬、赤5筒、赤5索）
/// 
/// Spaces are allowed.
pub fn hand_with_aka(s: &str) -> Result<[u8; 37]> {
    // 文字列をバイト単位で処理するため、ASCII文字のみを許可
    // We will be using bytes instead of chars afterwards.
    ensure!(s.is_ascii(), "hand {s} contains non-ascii content");

    // 結果を格納する37要素の配列を初期化（全て0）
    let mut ret = [0; 37];
    // 数字を一時的に保存するスタック（例: "123m"の場合、[1,2,3]を保存）
    let mut stack = vec![];

    // 文字列をバイト単位で順に処理
    for b in s.as_bytes() {
        match b {
            // 数字（0-9）の場合：スタックに追加
            b'0'..=b'9' => stack.push((b - b'0') as usize),
            // 種類を表す文字（m,p,s,z）の場合：スタックの数字を処理
            b'm' | b'p' | b's' | b'z' => {
                // スタックから数字を取り出して処理（drainで全て取り出し、スタックを空にする）
                for t in stack.drain(..) {
                    // 配列のインデックスを計算
                    let idx = if t == 0 {
                        // 0の場合は赤ドラとして処理
                        match b {
                            b'm' => tuz!(5mr), // 赤5萬のインデックス（34）
                            b'p' => tuz!(5pr), // 赤5筒のインデックス（35）
                            b's' => tuz!(5sr), // 赤5索のインデックス（36）
                            _ => bail!("unexpected byte {b}"),
                        }
                    } else {
                        // 1-9の場合は通常の牌として処理
                        let kind = match b {
                            b'm' => 0, // 萬子：インデックス0-8
                            b'p' => 1, // 筒子：インデックス9-17
                            b's' => 2, // 索子：インデックス18-26
                            b'z' => 3, // 字牌：インデックス27-33
                            _ => unreachable!(),
                        };
                        // インデックス計算：種類×9 + (数字-1)
                        // 例：3m → 0×9 + (3-1) = 2
                        kind * 9 + t - 1
                    };
                    // 該当する牌の枚数を1増やす
                    ret[idx] += 1;
                }
            }
            // 空白文字は無視
            b' ' | b'\t' | b'\n' => (),
            // その他の文字はエラー
            _ => bail!("unexpected byte {b}"),
        };
    }

    Ok(ret)
}

/// 手牌文字列を34要素の配列に変換する関数（赤ドラを通常の5として扱う）
/// 
/// 引数：
/// - `s`: 天鳳形式の手牌文字列（例: "123m 0p 789s 11z"）
///   - 0は赤5として扱われるが、通常の5と同じインデックスに格納される
///   - スペース、タブ、改行は許可される
/// 
/// 戻り値：
/// - `Result<[u8; 34]>`: 各牌の枚数を表す34要素の配列
///   - 赤ドラは通常の5と同じ扱いになる
/// 
/// Spaces are allowed.
pub fn hand(s: &str) -> Result<[u8; 34]> {
    // 結果を格納する34要素の配列を初期化（全て0）
    let mut ret = [0; 34];
    // まず赤ドラを含む37要素の配列として手牌を解析
    let hand = hand_with_aka(s)?;
    // 最初の34要素（通常の牌）をコピー
    vec_add_assign(&mut ret, &hand);
    // 赤ドラを通常の5に加算
    ret[tuz!(5m)] += hand[tuz!(5mr)]; // 赤5萬を通常の5萬に加算
    ret[tuz!(5p)] += hand[tuz!(5pr)]; // 赤5筒を通常の5筒に加算
    ret[tuz!(5s)] += hand[tuz!(5sr)]; // 赤5索を通常の5索に加算

    Ok(ret)
}

/// 37要素の牌配列をTileのVectorに変換する関数
/// 
/// 引数：
/// - `tiles`: 各牌の枚数を表す37要素の配列（赤ドラ含む）
/// 
/// 戻り値：
/// - `Vec<Tile>`: 牌のVector（各牌が枚数分だけ含まれる）
/// 
/// 注意：
/// - 赤ドラ（インデックス34-36）は1枚までしか追加されない
/// 
#[must_use]
pub fn tile37_to_vec(tiles: &[u8; 37]) -> Vec<Tile> {
    // 結果を格納するVector
    let mut ret = vec![];
    tiles
        .iter()
        .enumerate() // (インデックス, 枚数)のタプルを生成
        .filter(|&(_, &count)| count > 0) // 枚数が1以上の牌のみ処理
        .for_each(|(tid, &count)| {
            if tid < 34 {
                // 通常の牌（インデックス0-33）：枚数分だけ追加
                // resizeで現在の長さ+count分まで拡張し、must_tile!(tid)で埋める
                ret.resize(ret.len() + count as usize, must_tile!(tid));
            } else {
                // 赤ドラ（インデックス34-36）：1枚だけ追加
                // 赤ドラは通常1枚しか存在しないため、pushで1枚追加
                ret.push(must_tile!(tid));
            }
        });
    ret
}

/// 34要素の牌配列をTileのVectorに変換する関数
/// 
/// 引数：
/// - `tiles`: 各牌の枚数を表す34要素の配列（赤ドラなし）
/// 
/// 戻り値：
/// - `Vec<Tile>`: 牌のVector（各牌が枚数分だけ含まれる）
/// 
#[must_use]
pub fn tile34_to_vec(tiles: &[u8; 34]) -> Vec<Tile> {
    // 結果を格納するVector
    let mut ret = vec![];
    tiles
        .iter()
        .enumerate() // (インデックス, 枚数)のタプルを生成
        .filter(|&(_, &count)| count > 0) // 枚数が1以上の牌のみ処理
        .for_each(|(tid, &count)| {
            // 枚数分だけ同じ牌を追加
            // resizeで現在の長さ+count分まで拡張し、must_tile!(tid)で埋める
            ret.resize(ret.len() + count as usize, must_tile!(tid));
        });
    ret
}

/// 34要素の牌配列を天鳳形式の文字列に変換する関数
/// 
/// 引数：
/// - `tiles`: 各牌の枚数を表す34要素の配列
/// - `aka`: 赤ドラの有無を表す3要素の配列
///   - aka[0]: 赤5萬の有無
///   - aka[1]: 赤5筒の有無
///   - aka[2]: 赤5索の有無
/// 
/// 戻り値：
/// - `String`: 天鳳形式の手牌文字列（例: "123m 456p 789s 11z"）
///   - 赤5は"0"として表現される
/// 
#[must_use]
pub fn tiles_to_string(tiles: &[u8; 34], aka: [bool; 3]) -> String {
    // 数牌（萬子・筒子・索子）の処理
    let suhai = tiles[..3 * 9] // 最初の27要素（数牌）を取得
        .chunks_exact(9) // 9要素ずつ（各種類ごと）に分割
        .enumerate() // (種類インデックス, 9要素の配列)のタプルを生成
        .map(|(kind, chunk)| {
            // 各種類（萬子/筒子/索子）の文字列を生成
            let mut partial = String::new();
            let mut not_empty = false; // この種類に牌があるかのフラグ
            chunk
                .iter()
                .enumerate() // (数字インデックス, 枚数)のタプルを生成
                .filter(|&(_, &count)| count > 0) // 枚数が1以上の牌のみ処理
                .for_each(|(num, &count)| {
                    let literal_num = num + 1; // インデックスを実際の数字に変換（0→1、1→2、...）
                    if literal_num == 5 && aka[kind] {
                        // 5があり、かつ赤ドラがある場合
                        partial.push('0'); // 最初の1枚を赤5として"0"で表現
                        // 残りの枚数分は通常の5として追加
                        partial += &literal_num.to_string().repeat(count as usize - 1);
                    } else {
                        // 通常の牌：枚数分だけ数字を繰り返す
                        partial += &literal_num.to_string().repeat(count as usize);
                    }
                    not_empty = true;
                });

            if not_empty {
                // 種類を表す文字（m/p/s）を末尾に追加
                let c = match kind {
                    0 => 'm', // 萬子
                    1 => 'p', // 筒子
                    2 => 's', // 索子
                    _ => unreachable!(),
                };
                partial.push(c);
            }
            partial
        })
        .filter(|s| !s.is_empty()) // 空でない文字列のみ残す
        .collect::<Vec<_>>() // Vectorに収集
        .join(" "); // スペースで結合

    // 字牌の処理
    let jihai: String = tiles[3 * 9..] // 最後の7要素（字牌）を取得
        .iter()
        .enumerate() // (インデックス, 枚数)のタプルを生成
        .filter(|&(_, &count)| count > 0) // 枚数が1以上の牌のみ処理
        .map(|(num, &count)| (num + 1).to_string().repeat(count as usize)) // 1-7の数字を枚数分繰り返す
        .collect(); // 文字列に結合

    // 最終的な文字列を生成
    if jihai.is_empty() {
        // 字牌がない場合は数牌のみ
        suhai
    } else {
        // 字牌がある場合は末尾に"z"を付けて結合
        format!("{suhai} {jihai}z")
    }
}

/// テストモジュール
#[cfg(test)]
mod test {
    use super::*;

    /// 文字列から配列への変換をテストする関数
    #[test]
    fn parse() {
        // テストケース1: 基本的な手牌の解析
        // "1111m 333p 222s 444z" → 1萬×4、3筒×3、2索×3、西×3
        assert_eq!(
            hand("1111m 333p 222s 444z").unwrap(),
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, // m: 1萬が4枚
                0, 0, 3, 0, 0, 0, 0, 0, 0, // p: 3筒が3枚
                0, 3, 0, 0, 0, 0, 0, 0, 0, // s: 2索が3枚
                0, 0, 0, 3, 0, 0, 0, // z: 西（4番目の字牌）が3枚
            ]
        );

        // テストケース2: 赤ドラを含む手牌の解析
        // "22334450m234p2s3s4s" → 赤5萬1枚を含む複雑な手牌
        assert_eq!(
            hand_with_aka("22334450m234p2s3s4s").unwrap(),
            [
                0, 2, 2, 2, 1, 0, 0, 0, 0, // m: 2萬×2、3萬×2、4萬×2、5萬×1
                0, 1, 1, 1, 0, 0, 0, 0, 0, // p: 2筒×1、3筒×1、4筒×1
                0, 1, 1, 1, 0, 0, 0, 0, 0, // s: 2索×1、3索×1、4索×1
                0, 0, 0, 0, 0, 0, 0, // z: 字牌なし
                1, 0, 0, // a: 赤5萬×1、赤5筒×0、赤5索×0
            ]
        );

        // テストケース3: スペースを含む複雑な手牌の解析
        // 複数の種類が混在し、同じ数字が複数回出現するケース
        assert_eq!(
            hand("456m 6p 7899p 77z 987s 9p").unwrap(),
            [
                0, 0, 0, 1, 1, 1, 0, 0, 0, // m: 4萬×1、5萬×1、6萬×1
                0, 0, 0, 0, 0, 1, 1, 1, 3, // p: 6筒×1、7筒×1、8筒×1、9筒×3（9pが2回出現）
                0, 0, 0, 0, 0, 0, 1, 1, 1, // s: 7索×1、8索×1、9索×1
                0, 0, 0, 0, 0, 0, 2, // z: 中（7番目の字牌）×2
            ]
        );
    }

    /// 配列から文字列への変換をテストする関数
    #[test]
    fn string() {
        // テストケース: 赤5萬を含む手牌の文字列化
        // 配列 → "33067m 345678p 678s"
        // 赤5萬があるため、5萬の1枚目が"0"として出力される
        assert_eq!(
            tiles_to_string(
                &[
                    0, 0, 2, 0, 1, 1, 1, 0, 0, // m: 3萬×2、5萬×1、6萬×1、7萬×1
                    0, 0, 1, 1, 1, 1, 1, 1, 0, // p: 3筒〜8筒各1枚
                    0, 0, 0, 0, 0, 1, 1, 1, 0, // s: 6索〜8索各1枚
                    0, 0, 0, 0, 0, 0, 0, // z: 字牌なし
                ],
                [true, false, false] // 赤5萬あり、赤5筒なし、赤5索なし
            ),
            "33067m 345678p 678s"
        );
    }
}
