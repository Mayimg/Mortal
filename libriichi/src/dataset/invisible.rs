// ファイル概要：
// このファイルは麻雀ゲームの不可視情報（見えない情報）を管理するモジュールです。
// オラクルモード時に山牌、嶺上牌、ドラ表示牌、裏ドラ表示牌の情報を保持し、
// 他家の手牌情報と合わせて機械学習用の特徴量にエンコードします。

// ゲーム盤面の管理
use crate::arena::Board;
// 2次元配列の簡易実装
use crate::array::Simple2DArray;
// オラクル観測値の形状を取得する関数
use crate::consts::oracle_obs_shape;
// mjaiイベント定義
use crate::mjai::Event;
// プレイヤー状態
use crate::state::PlayerState;
// 牌の定義
use crate::tile::Tile;
// 牌関連のマクロ（must_tile!: usize→Tile変換、tu8!: 牌文字→u8変換、tuz!: 牌文字→usize変換）
use crate::{must_tile, tu8, tuz};
// イテレータユーティリティ
use std::iter;
// メモリ操作（所有権の移動）
use std::mem;

// 多次元配列処理
use ndarray::prelude::*;
// 乱数生成
use rand::prelude::*;
// スレッドローカルな乱数生成器
use rand::rng;

/// All fields are sorted early -> late.
/// 不可視情報を管理する構造体
/// すべてのフィールドは早い順（先にツモする順）でソートされている
#[derive(Default)]
pub struct Invisible {
    // 山牌（通常70枚：配牌後の残り）
    pub yama: Vec<Tile>,
    // 嶺上牌（4枚：カン後にツモする牌）
    pub rinshan: Vec<Tile>,
    // ドラ表示牌（最大5枚）
    pub dora_indicators: Vec<Tile>,
    // 裏ドラ表示牌（最大5枚：リーチ和了時のみ有効）
    pub ura_indicators: Vec<Tile>,
}

impl Invisible {
    // ゲームイベントから各局の不可視情報を構築
    pub fn new(game: &[Event], trust_seed: bool) -> Vec<Self> {
        // 各局の不可視情報を格納するベクター
        let mut ret = vec![];
        // 現在処理中の局の不可視情報
        let mut cur = Self::default();
        // ゲームのシード値（trust_seed=trueの場合のみ使用）
        let mut seed = None;
        // 次のツモが嶺上牌からかどうか
        let mut from_rinshan = false;
        // 裏ドラが記録済みかどうか
        let mut ura_is_recorded = false;
        // まだ見えていない牌の枚数を追跡（各牌種ごと）
        let mut unknown_tiles = new_unknown_tiles();

        // すべてのイベントを処理
        for event in game {
            match event {
                // If the game was emulated by our lib, then use the seed directly
                // ゲームが当ライブラリでエミュレートされた場合、シード値を直接使用
                Event::StartGame {
                    seed: Some(game_seed),
                    ..
                } if trust_seed => {
                    seed = Some(*game_seed);
                }

                // 局開始イベント
                Event::StartKyoku {
                    bakaze,         // 場風
                    kyoku,          // 局数
                    honba,          // 本場数
                    dora_marker,    // ドラ表示牌
                    tehais,         // 各プレイヤーの配牌
                    ..
                } => {
                    // シード値がある場合は、山牌を完全に再現
                    if let Some(seed) = seed {
                        // Boardオブジェクトを作成
                        let mut board = Board {
                            // 通算局数を計算（東1=0、東2=1、...）
                            kyoku: 4 * (bakaze.as_u8() - tu8!(E)) + kyoku - 1,
                            honba: *honba,
                            ..Default::default()
                        };
                        // シード値から山牌を初期化
                        board.init_from_seed(seed);

                        // Boardから不可視情報をコピー
                        cur.yama = board.yama;
                        cur.rinshan = board.rinshan;
                        cur.dora_indicators = board.dora_indicators;
                        cur.ura_indicators = board.ura_indicators;

                        // reverse because of the way Board pops tiles
                        // Boardは末尾から牌を取り出すため、順序を反転
                        cur.yama.reverse();
                        cur.rinshan.reverse();
                        cur.dora_indicators.reverse();

                        // 完成した不可視情報を保存し、次の局へ
                        ret.push(mem::take(&mut cur));
                        continue;
                    }
                    
                    // シード値がない場合：観測可能な情報から推測
                    // ドラ表示牌を記録
                    cur.dora_indicators.push(*dora_marker);
                    unknown_tiles[dora_marker.as_usize()] -= 1;
                    
                    // 配牌された牌を未知の牌から除外
                    tehais
                        .iter()
                        .flatten()
                        .for_each(|tile| unknown_tiles[tile.as_usize()] -= 1);
                }
                _ => (),
            };

            // シード値がある場合は、以降の処理をスキップ
            if seed.is_some() {
                continue;
            }

            // シード値がない場合の処理：イベントから山牌を推測
            match event {
                // ツモイベント：ツモられた牌を記録
                Event::Tsumo { pai, .. } => {
                    if from_rinshan {
                        // 嶺上牌からのツモ（カン後）
                        cur.rinshan.push(*pai);
                        from_rinshan = false;
                    } else {
                        // 通常の山牌からのツモ
                        cur.yama.push(*pai);
                        // 山牌は最大70枚（配牌後の残り）
                        assert!(cur.yama.len() <= 70, "yama size overflow");
                    }
                    // ツモられた牌を未知の牌から除外
                    unknown_tiles[pai.as_usize()] -= 1;
                }
                
                // カンイベント：次のツモは嶺上牌から
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                    from_rinshan = true;
                }
                
                // 新ドライベント：ドラ表示牌を追加
                Event::Dora { dora_marker } => {
                    cur.dora_indicators.push(*dora_marker);
                    unknown_tiles[dora_marker.as_usize()] -= 1;
                }
                
                // 和了イベント：裏ドラ表示牌を記録（リーチ和了時）
                Event::Hora {
                    ura_markers: Some(ura),
                    ..
                } if !ura_is_recorded => {
                    // 裏ドラ表示牌をすべて記録
                    for &tile in ura {
                        cur.ura_indicators.push(tile);
                        unknown_tiles[tile.as_usize()] -= 1;
                    }
                    ura_is_recorded = true;
                }
                
                // 局終了イベント：見えなかった牌をランダムに配置
                Event::EndKyoku => {
                    // まだ見えていない牌のリストを作成
                    let mut filler: Vec<_> = unknown_tiles
                        .into_iter()
                        .enumerate()
                        .filter(|&(_, count)| count > 0)               // 残っている牌種のみ
                        .flat_map(|(tid, count)| {                     // 各牌種について
                            iter::repeat_n(must_tile!(tid), count as usize)  // count枚分生成
                        })
                        .collect();
                    // ランダムにシャッフル
                    filler.shuffle(&mut rng());

                    // 山牌を70枚まで埋める
                    while cur.yama.len() < 70 {
                        cur.yama.push(filler.pop().unwrap());
                    }
                    // 嶺上牌を4枚まで埋める
                    while cur.rinshan.len() < 4 {
                        cur.rinshan.push(filler.pop().unwrap());
                    }
                    // ドラ表示牌を5枚まで埋める
                    while cur.dora_indicators.len() < 5 {
                        cur.dora_indicators.push(filler.pop().unwrap());
                    }
                    // 裏ドラ表示牌を5枚まで埋める
                    while cur.ura_indicators.len() < 5 {
                        cur.ura_indicators.push(filler.pop().unwrap());
                    }
                    // すべての牌が配置されたことを確認
                    assert!(filler.is_empty());

                    // 完成した不可視情報を保存
                    ret.push(mem::take(&mut cur));
                    // 次の局のために状態をリセット
                    from_rinshan = false;
                    ura_is_recorded = false;
                    unknown_tiles = new_unknown_tiles();
                }

                _ => (),  // その他のイベントは無視
            };
        }

        ret  // 全局分の不可視情報を返す
    }

    // TODO: merge this this arena::board::BoardState::encode_oracle_obs; they
    // should be identical.
    // TODO: arena::board::BoardState::encode_oracle_obsと統合すべき（同じ処理）
    // 不可視情報を機械学習用の特徴量にエンコード
    pub fn encode(
        &self,
        opponent_states: &[PlayerState; 3],     // 対戦相手3人の状態
        yama_idx: usize,                        // 山牌の現在位置（何枚目まで使ったか）
        rinshan_idx: usize,                     // 嶺上牌の現在位置
        version: u32,                           // エンコーディングバージョン
    ) -> Array2<f32> {
        // バージョンに応じた特徴量の形状を取得
        let shape = oracle_obs_shape(version);
        // 2次元配列を初期化（34は牌種の数）
        let mut arr = Simple2DArray::<34, f32>::new(shape.0);
        // 現在の行インデックス
        let mut idx = 0;

        // 各対戦相手の情報をエンコード
        for state in opponent_states {
            // 手牌の枚数をエンコード（牌種ごとに4行使用）
            state
                .tehai()
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)   // 持っている牌のみ
                .for_each(|(tile_id, &count)| {
                    // count枚分の行に1.0を設定
                    arr.assign_rows(idx, tile_id, count as usize, 1.)
                });
            idx += 4;

            // 赤牌の有無をエンコード（5m赤、5p赤、5s赤の3行）
            state
                .akas_in_hand()
                .iter()
                .enumerate()
                .filter(|&(_, &has_it)| has_it)    // 持っている赤牌のみ
                .for_each(|(i, _)| arr.fill(idx + i, 1.));
            idx += 3;

            // シャンテン数をエンコード
            let n = state.shanten() as usize;
            match version {
                1 => {
                    // v1: n行分を1.0で埋める
                    arr.fill_rows(idx, n, 1.);
                    idx += 6;
                }
                2 | 3 | 4 => {
                    // v2以降: one-hotエンコーディング
                    arr.fill(idx + n, 1.);
                    idx += 7;

                    // 正規化されたシャンテン数も追加
                    let v = n as f32 / 6.;
                    arr.fill(idx, v);
                    idx += 1;
                }
                _ => unreachable!(),
            }

            // 待ち牌をエンコード（34牌種分の1行）
            state
                .waits()
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)               // 待ちになっている牌
                .for_each(|(t, _)| arr.assign(idx, t, 1.));
            idx += 1;

            // フリテン状態をエンコード
            if state.at_furiten() {
                arr.fill(idx, 1.);
            }
            idx += 1;
        }

        // 牌をエンコードするヘルパークロージャ
        let mut encode_tile = |idx: usize, tile: Tile| {
            // 赤牌を通常牌として扱い、牌種IDを取得
            let tile_id = tile.deaka().as_usize();
            // 牌種の位置に1.0を設定
            arr.assign(idx, tile_id, 1.);
            // 赤牌の場合は次の行にも1.0を設定
            if tile.is_aka() {
                arr.fill(idx + 1, 1.);
            }
        };

        // 山牌の残りをエンコード（各牌2行使用：牌種+赤牌フラグ）
        for &tile in &self.yama[yama_idx..] {
            encode_tile(idx, tile);
            idx += 2;
        }
        // In real life case `self.yama[yama_idx..].len()` is at most 69 since
        // `yama_idx` >= 1 always holds, as the dealer always unconditionally
        // deals the first tile from yama. Therefore we do the minus one here.
        // 実際のケースでは`self.yama[yama_idx..].len()`は最大69枚
        // （親は無条件で最初の1枚を配るため、yama_idx >= 1が常に成立）
        // そのため、ここでマイナス1する
        idx += (yama_idx - 1) * 2;

        // 嶺上牌の残りをエンコード
        for &tile in &self.rinshan[rinshan_idx..] {
            encode_tile(idx, tile);
            idx += 2;
        }
        idx += rinshan_idx * 2;

        // ドラ表示牌をエンコード（各5枚×2行）
        for &tile in &self.dora_indicators {
            encode_tile(idx, tile);
            idx += 2;
        }
        // 裏ドラ表示牌をエンコード（各5枚×2行）
        for &tile in &self.ura_indicators {
            encode_tile(idx, tile);
            idx += 2;
        }

        // 全ての特徴量が正しくエンコードされたことを確認
        assert_eq!(idx, shape.0);
        // 2次元配列を構築して返す
        arr.build()
    }
}

// 未知の牌の初期枚数を設定する関数
const fn new_unknown_tiles() -> [u8; 37] {
    // 基本的に各牌は4枚ずつ存在
    let mut ret = [4; 37];
    // 5万、5筒、5索は通常牌が3枚（赤牌が1枚あるため）
    ret[tuz!(5m)] = 3;
    ret[tuz!(5p)] = 3;
    ret[tuz!(5s)] = 3;
    // 赤5万、赤5筒、赤5索は各1枚
    ret[tuz!(5mr)] = 1;
    ret[tuz!(5pr)] = 1;
    ret[tuz!(5sr)] = 1;
    ret
}
