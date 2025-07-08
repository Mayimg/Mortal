//! 和了（アガリ）判定と役・符・飜数の計算を行うモジュール
//!
//! このモジュールは麻雀の和了形を判定し、成立する役を探索して得点計算に必要な
//! 符（フ）と飜（ハン）を算出します。国士無双や七対子といった特殊形から、
//! 通常の3面子1雀頭形まで、あらゆる和了形に対応しています。
//!
//! 実装は山岡忠夫氏のJava実装を基にしており、EndlessCheng氏によるGo移植版を
//! 経由してRustに移植されています。アルゴリズムの詳細は以下を参照：
//!
//! Rust port of EndlessCheng's Go port of 山岡忠夫's Java implementation of his
//! agari algorithm.
//!
//! Source:
//! * Go: <https://github.com/EndlessCheng/mahjong-helper/blob/master/util/agari.go>
//! * Java: <http://hp.vector.co.jp/authors/VA046927/mjscore/AgariIndex.java>
//! * Algorithm: <http://hp.vector.co.jp/authors/VA046927/mjscore/mjalgorism.html>

// 点数計算モジュール（符と飜から実際の支払い点数を計算）
use super::point::Point;
// 向聴数計算モジュール（国士無双の判定で使用）
use super::shanten;
// 牌の型定義
use crate::tile::Tile;
// 牌ID操作用マクロ（tu8!:u8型、must_tile!:Tile型変換、matches_tu8!:パターンマッチ）
use crate::{matches_tu8, must_tile, tu8};
// 和了結果の比較に使用
use std::cmp::Ordering;
// イテレータ操作（特にchain()で複数のイテレータを連結）
use std::iter;
// 静的変数の遅延初期化（初回アクセス時にのみ初期化）
use std::sync::LazyLock;

// MPHF（Minimal Perfect Hash Function）による高速ハッシュマップ
// 事前に全てのキーが分かっている場合に最適
use boomphf::hashmap::BoomHashMap;
// バイナリデータの読み込み（リトルエンディアン形式）
use byteorder::{LittleEndian, ReadBytesExt};
// gzip圧縮されたデータの展開
use flate2::read::GzDecoder;
// スタック上に確保される可変長配列（ヒープ割り当てを回避）
use tinyvec::ArrayVec;

// 和了パターンテーブルのエントリ数
// 全ての可能な14枚の手牌のうち、和了形となるパターンの総数
const AGARI_TABLE_SIZE: usize = 9_362;

// 和了形パターンテーブル（初回アクセス時に遅延初期化される）
// キー: 手牌のビットパターン表現（u32）
// 値: その手牌で可能な面子分割パターンのリスト（最大4パターン）
static AGARI_TABLE: LazyLock<BoomHashMap<u32, ArrayVec<[Div; 4]>>> = LazyLock::new(|| {
    // コンパイル時に埋め込まれた圧縮バイナリデータを読み込み
    let mut raw = GzDecoder::new(include_bytes!("data/agari.bin.gz").as_slice());

    // テーブルの各エントリを読み込み、キーと値のペアに分解
    let (keys, values): (Vec<_>, Vec<_>) = (0..AGARI_TABLE_SIZE)
        .map(|_| {
            // 手牌パターンのキー（32ビット）を読み込み
            let key = raw.read_u32::<LittleEndian>().unwrap();
            // この手牌で可能な面子分割パターンの数（1〜4）
            let v_size = raw.read_u8().unwrap();
            // 各分割パターンをu32として読み込み、Div構造体に変換
            let value = (0..v_size)
                .map(|_| raw.read_u32::<LittleEndian>().unwrap())
                .map(Div::from) // u32からDiv構造体への変換（ビットフィールドのデコード）
                .collect();
            (key, value)
        })
        .unzip(); // (キーのベクタ, 値のベクタ)に分離

    // テストビルド時のみ実行される検証処理
    if cfg!(test) {
        // キーの重複がないことを確認
        let mut k = keys.clone();
        k.sort_unstable(); // ソート
        k.dedup();         // 重複を除去
        assert_eq!(k.len(), keys.len()); // 重複除去後もサイズが同じ＝重複なし

        // データが完全に読み込まれたことを確認（余りがあればエラー）
        raw.read_u8().unwrap_err();
    }

    // MPHFハッシュマップを構築して返す
    // 全てのキーが事前に分かっているため、完全ハッシュ関数を生成可能
    BoomHashMap::new(keys, values)
});

// 面子分割情報を保持する構造体
// 14枚の手牌をどのように面子（刻子・順子）と雀頭に分割するかを表現
#[derive(Debug, Default)]
struct Div {
    // 雀頭の位置（tile14配列のインデックス、0〜13）
    pair_idx: u8,
    // 刻子の位置リスト（tile14配列のインデックス、最大4つ）
    kotsu_idxs: ArrayVec<[u8; 4]>,
    // 順子の位置リスト（tile14配列のインデックス、最大4つ）
    shuntsu_idxs: ArrayVec<[u8; 4]>,
    // この分割が七対子形かどうか
    has_chitoi: bool,
    // この分割で九蓮宝燈の可能性があるか
    has_chuuren: bool,
    // この分割で一気通貫の可能性があるか
    has_ittsuu: bool,
    // この分割が二盃口形かどうか
    has_ryanpeikou: bool,
    // 注意: 一盃口フラグは暗槓がある場合は不完全（正しく判定されない可能性がある）
    has_ipeikou: bool,
}

// 和了結果を表す列挙型
#[derive(Debug, Clone, Copy, Eq)]
pub enum Agari {
    // 通常の和了（符と飜で表現）
    /// `fu` may be 0 if `han` is greater than 4.
    Normal {
        fu: u8,   // 符数（5飜以上の場合は0でもよい）
        han: u8,  // 飼数
    },
    // 役満（倍数で表現：1=役満、2=ダブル役満など）
    Yakuman(u8),
}

// 和了計算のための入力情報を保持する構造体
#[derive(Debug)]
pub struct AgariCalculator<'a> {
    /// Must include the winning tile (i.e. must be 3n+2)
    // 手牌（34種×枚数、和了牌を含む3n+2枚）
    pub tehai: &'a [u8; 34],
    /// `self.chis.is_empty() && self.pons.is_empty() && self.minkans.is_empty()`
    // 門前かどうか（副露がない場合true）
    pub is_menzen: bool,
    // チーした牌のリスト（順子の最初の牌）
    pub chis: &'a [u8],
    // ポンした牌のリスト
    pub pons: &'a [u8],
    // 明槓した牌のリスト
    pub minkans: &'a [u8],
    // 暗槓した牌のリスト
    pub ankans: &'a [u8],

    // 場風牌（柱=27、南=28、西=29、北=30）
    pub bakaze: u8,
    // 自風牌（柱=27、南=28、西=29、北=30）
    pub jikaze: u8,

    /// Must be deakaized
    // 和了牌（赤牌は除去済み、例：赤5mを5mに変換）
    pub winning_tile: u8,
    /// For consistency reasons, `is_ron` is only used to calculate fu and check
    /// ankou/ankan-related yakus like 三/四暗刻. It will not be used to
    /// determine 門前清自摸和.
    // ロン和了かどうか（符計算と暗刻系役の判定にのみ使用、門前清自摸和の判定には使わない）
    pub is_ron: bool,
}

// 分割パターンを元に役・符計算を行うワーカー構造体
struct DivWorker<'a> {
    // 和了計算器への参照
    sup: &'a AgariCalculator<'a>,
    // 14枚の手牌（実際の牌ID配列）
    tile14: &'a [u8; 14],
    // 面子分割情報
    div: &'a Div,
    // 雀頭の牌ID
    pair_tile: u8,
    // 門前の刻子リスト（牌ID）
    menzen_kotsu: ArrayVec<[u8; 4]>,
    // 門前の順子リスト（最初の牌ID）
    menzen_shuntsu: ArrayVec<[u8; 4]>,

    /// Used in fu calc and sanankou condition, indicating whether or not the
    /// winning tile should build a minkou instead of shuntsu in an ambiguous
    /// pattern.
    ///
    /// The winning tile should try its best to fit into a shuntsu, because that
    /// always gives a higher score than using that winning tile to turn an
    /// existing ankou into a minkou, because a shuntsu can only add at most 2
    /// fu (penchan or kanchan) and does not bring extra yaku (except for pinfu,
    /// but since we have ankou it can never be pinfu), but an ankou adds at
    /// least 2 fu and can bring extra yakus like sanankou.
    ///
    /// An example of this is 45556 + 5, which could be either 456 + (55 + 5) or
    /// 555 + (46 + 5). `menzen_kotsu` will contain 555 while `menzen_shuntsu`
    /// will also contain 456, making it ambiguous whether the winning tile 5
    /// should be a part of either the minkou 55 + 5 or the shuntsu 46 + 5. In
    /// practice, the latter should be preferred because it preserves the ankou.
    /// A test case covers this.
    // 和了牌が明刻を作るかどうか（符計算と三暗刻判定に使用）
    // 例：45556+5の場合、456+555（暗刻を保持）か555+456（ロンで明刻化）か
    // 高点法の原則により、暗刻を保持する方が優先される
    winning_tile_makes_minkou: bool,
}

// u32からDiv構造体への変換（ビットフィールドのデコード）
// ビット配置：
// 0-2: 刻子数（3ビット、0〜4）
// 3-5: 順子数（3ビット、0〜4）
// 6-9: 雀頭位置（4ビット、0〜13）
// 10-25: 面子位置（各面子を4ビット、最大4つ×4ビット=16ビット）
// 26: 七対子フラグ
// 27: 九蓮宝燈フラグ
// 28: 一気通貫フラグ
// 29: 二盃口フラグ
// 30: 一盃口フラグ
impl From<u32> for Div {
    fn from(v: u32) -> Self {
        // 雀頭位置を抽出（6-9ビット目）
        let pair_idx = ((v >> 6) & 0b1111) as u8;

        // 刻子数を抽出（0-2ビット目）
        let kotsu_count = v & 0b111;
        // 刻子位置を抽出（10ビット目から各刻子4ビットずつ）
        let kotsu_idxs = (0..kotsu_count)
            .map(|i| ((v >> (10 + i * 4)) & 0b1111) as u8)
            .collect();

        // 順子数を抽出（3-5ビット目）
        let shuntsu_count = (v >> 3) & 0b111;
        // 順子位置を抽出（刻子の後に続けて格納されている）
        let shuntsu_idxs = (kotsu_count..kotsu_count + shuntsu_count)
            .map(|i| ((v >> (10 + i * 4)) & 0b1111) as u8)
            .collect();

        // 役判定用フラグを抽出
        let has_chitoi = (v >> 26) & 0b1 == 0b1;      // 七対子
        let has_chuuren = (v >> 27) & 0b1 == 0b1;     // 九蓮宝燈
        let has_ittsuu = (v >> 28) & 0b1 == 0b1;      // 一気通貫
        let has_ryanpeikou = (v >> 29) & 0b1 == 0b1;  // 二盃口
        let has_ipeikou = (v >> 30) & 0b1 == 0b1;     // 一盃口

        Self {
            pair_idx,
            kotsu_idxs,
            shuntsu_idxs,
            has_chitoi,
            has_chuuren,
            has_ittsuu,
            has_ryanpeikou,
            has_ipeikou,
        }
    }
}

// Agariの等価性判定
impl PartialEq for Agari {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // 役満同士：倍数が同じか
            (Self::Yakuman(l), Self::Yakuman(r)) => l == r,
            // 通常和了同士：符と飼が両方同じか
            (Self::Normal { fu: lf, han: lh }, Self::Normal { fu: rf, han: rh }) => {
                lf == rf && lh == rh
            }
            // 役満と通常和了は異なる
            _ => false,
        }
    }
}

// Agariの部分順序（全順序が定義されているのでSomeを返す）
impl PartialOrd for Agari {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Agariの完全順序（高点法のための比較）
impl Ord for Agari {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            // 役満同士：倍数が大きい方が上
            (Self::Yakuman(l), Self::Yakuman(r)) => l.cmp(r),
            // 役満は常に通常和了より上
            (Self::Yakuman(_), Self::Normal { .. }) => Ordering::Greater,
            (Self::Normal { .. }, Self::Yakuman(..)) => Ordering::Less,
            // 通常和了同士：まず飼数で比較、同じなら符数で比較
            (Self::Normal { fu: lf, han: lh }, Self::Normal { fu: rf, han: rh }) => {
                match lh.cmp(rh) {
                    Ordering::Equal => lf.cmp(rf), // 飼数が同じ場合は符数で比較
                    v => v,                        // 飼数が異なる場合はその結果を使用
                }
            }
        }
    }
}

impl Agari {
    // 和了結果から実際の点数を計算
    #[must_use]
    pub fn point(self, is_oya: bool) -> Point {
        match self {
            // 通常和了：符と飼から点数を計算
            Self::Normal { fu, han } => Point::calc(is_oya, fu, han),
            // 役満：倍数から点数を計算
            Self::Yakuman(n) => Point::yakuman(is_oya, n as i32),
        }
    }
}

impl AgariCalculator<'_> {
    // 役があるかどうかを判定（高速判定用）
    #[inline]
    #[must_use]
    pub fn has_yaku(&self) -> bool {
        // return_if_any=trueで呼び出し、役が1つでもあれば即座にreturn
        self.search_yakus_impl(true).is_some()
    }

    // 全ての役を検索して最高得点を返す
    #[inline]
    #[must_use]
    pub fn search_yakus(&self) -> Option<Agari> {
        // return_if_any=falseで呼び出し、全ての役を検索
        self.search_yakus_impl(false)
    }

    /// `additional_hans` includes 門前清自摸和, (両)立直, 槍槓, 嶺上開花, 海底
    /// 摸月 and 河底撈魚. 天和 and 地和 are supposed to be checked somewhere
    /// else other than here.
    ///
    /// `None` is returned iff `!self.has_yaku() && additional_hans == 0` holds.
    ///
    /// This function is only supposed to be called by callers who have the
    /// knowledge of the ura doras.
    // 最終的な和了計算（状況役やドラを含む）
    // additional_hans: 門前清自摸和、立直、槍槓、嶺上開花などの状況役
    // doras: ドラ数（表ドラ、裏ドラ、赤ドラの合計）
    #[must_use]
    pub fn agari(&self, additional_hans: u8, doras: u8) -> Option<Agari> {
        // まず手牌役を検索
        if let Some(agari) = self.search_yakus() {
            // 手牌役がある場合、状況役とドラを加算
            Some(match agari {
                Agari::Normal { fu, han } => Agari::Normal {
                    fu,
                    han: han + additional_hans + doras,
                },
                _ => agari, // 役満はドラを加算しない
            })
        } else if additional_hans == 0 {
            // 手牌役も状況役もない場合は和了不可
            None
        } else if additional_hans + doras >= 5 {
            // 状況役とドラだけで5飼以上の場合（符計算不要）
            Some(Agari::Normal {
                fu: 0,
                han: additional_hans + doras,
            })
        } else {
            // 状況役とドラだけで4飼以下の場合（符計算が必要）
            let (tile14, key) = get_tile14_and_key(self.tehai);
            let divs = AGARI_TABLE.get(&key)?;

            // 全ての分割パターンで符を計算し、最大値を採用（高点法）
            let fu = divs
                .iter()
                .map(|div| DivWorker::new(self, &tile14, div))
                .map(|w| w.calc_fu(false))
                .max()?;
            Some(Agari::Normal {
                fu,
                han: additional_hans + doras,
            })
        }
    }

    // 役検索の実装
    // return_if_any: trueの場合は役が1つでもあれば即座にreturn（高速判定用）
    fn search_yakus_impl(&self, return_if_any: bool) -> Option<Agari> {
        // is_menzenフラグと副露配列の整合性を確認
        assert_eq!(
            self.is_menzen,
            self.chis.is_empty() && self.pons.is_empty() && self.minkans.is_empty(),
        );

        // Kokushi has a special pattern and cannot be combined with other
        // pattern-based yakus.
        // 国士無双の特別判定（他の役と複合しない）
        if self.is_menzen && shanten::calc_kokushi(self.tehai) == -1 {
            // 国士無双
            return Some(Agari::Yakuman(1));
        }

        // 手牌をキーに変換し、和了テーブルから分割パターンを取得
        let (tile14, key) = get_tile14_and_key(self.tehai);
        let divs = AGARI_TABLE.get(&key)?;

        if return_if_any {
            // Benchmark result indicates it is too trivial to use rayon here.
            // 役が1つでもあれば即座にreturn（find_mapを使用）
            divs.iter()
                .map(|div| DivWorker::new(self, &tile14, div))
                .find_map(|w| w.search_yakus::<true>())
        } else {
            // 全ての分割パターンで役を検索し、最高得点を返す
            divs.iter()
                .map(|div| DivWorker::new(self, &tile14, div))
                .filter_map(|w| w.search_yakus::<false>())
                .max() // AgariのOrd実装により最高得点が選ばれる
        }
    }
}

impl<'a> DivWorker<'a> {
    // DivWorkerの初期化
    fn new(calc: &'a AgariCalculator<'a>, tile14: &'a [u8; 14], div: &'a Div) -> Self {
        // 雀頭の牌IDを取得
        let pair_tile = tile14[div.pair_idx as usize];
        // 刻子のインデックスから実際の牌IDに変換
        let menzen_kotsu = div
            .kotsu_idxs
            .iter()
            .map(|&idx| tile14[idx as usize])
            .collect();
        // 順子のインデックスから実際の牌IDに変換
        let menzen_shuntsu = div
            .shuntsu_idxs
            .iter()
            .map(|&idx| tile14[idx as usize])
            .collect();

        let mut ret = Self {
            sup: calc,
            tile14,
            div,
            pair_tile,
            menzen_kotsu,
            menzen_shuntsu,
            winning_tile_makes_minkou: false,
        };
        // 和了牌が明刻を作るかどうかを判定
        ret.winning_tile_makes_minkou = ret.winning_tile_makes_minkou();
        ret
    }

    /// For init only.
    // 和了牌が明刻を作るかどうかの判定（初期化時のみ使用）
    fn winning_tile_makes_minkou(&self) -> bool {
        if !self.sup.is_ron {
            // Tsumo agari, no way to make a minkou from the winning tile.
            // ツモ和了の場合、和了牌が明刻になることはない
            return false;
        }
        if !self.menzen_kotsu.contains(&self.sup.winning_tile) {
            // No ankou that contains the winning tile, so no ambiguous pattern
            // at all.
            // 和了牌を含む暗刻がない場合、曖昧なパターンは存在しない
            return false;
        }

        if self.sup.winning_tile >= 3 * 9 {
            // If the ron winning tile is jihai and makes a kotsu, then it must
            // be a minkou.
            // 字牌のロン和了で刻子を作る場合、必ず明刻になる
            return true;
        }
        // 数牌の場合、順子に入れることができるか確認
        let kind = self.sup.winning_tile / 9;    // 色（0:萬、1:筒、2:索）
        let num = self.sup.winning_tile % 9;     // 数字（0〜8）
        let low = kind * 9 + num.saturating_sub(2);  // 順子の開始位置の最小値
        let high = kind * 9 + num.min(6);            // 順子の開始位置の最大値
        // If there is a shuntsu that can cover the winning tile, then always
        // put the winning tile into that shuntsu.
        // 和了牌を含む順子がある場合、順子に入れる（高点法）
        !(low..=high).any(|t| self.menzen_shuntsu.contains(&t))
    }

    /// The caller must assure `self.div.has_chitoi` holds.
    // 七対子の対子を取得（呼び出し側でhas_chitoiを確認すること）
    fn chitoi_pairs(&self) -> impl Iterator<Item = u8> + '_ {
        // tile14の最初の7要素が対子
        self.tile14.iter().take(7).copied()
    }

    // 全ての刻子・槓子を取得
    fn all_kotsu_and_kantsu(&self) -> impl Iterator<Item = u8> + '_ {
        self.menzen_kotsu
            .iter()           // 門前の刻子
            .chain(self.sup.pons)      // ポン
            .chain(self.sup.minkans)   // 明槓
            .chain(self.sup.ankans)    // 暗槓
            .copied()
    }

    // 全ての順子を取得
    fn all_shuntsu(&self) -> impl Iterator<Item = u8> + '_ {
        self.menzen_shuntsu.iter()     // 門前の順子
            .chain(self.sup.chis)      // チー
            .copied()
    }

    // 全ての面子（刻子・槓子・順子）を取得
    fn all_mentsu(&self) -> impl Iterator<Item = u8> + '_ {
        self.all_kotsu_and_kantsu().chain(self.all_shuntsu())
    }

    // 符計算
    fn calc_fu(&self, has_pinfu: bool) -> u8 {
        if self.div.has_chitoi {
            // 七対子は固定25符
            return 25;
        }
        // 基本符20符からスタート
        let mut fu = 20;

        // 門前刻子の符計算
        fu += self
            .menzen_kotsu
            .iter()
            .map(|&t| {
                // `menzen_kotsu` are usually ankou, except when the winning
                // tile makes a minkou and the tile is the winning tile.
                // 和了牌が明刻を作るかどうかで判定
                let is_minkou = self.winning_tile_makes_minkou && t == self.sup.winning_tile;
                match (is_minkou, must_tile!(t).is_yaokyuu()) {
                    (false, true) => 8,              // 暗刻・幺九牌
                    (false, false) | (true, true) => 4,   // 暗刻・中張牌 または 明刻・幺九牌
                    (true, false) => 2,              // 明刻・中張牌
                }
            })
            .sum::<u8>();
        // ポンの符計算
        fu += self
            .sup
            .pons
            .iter()
            .map(|&t| if must_tile!(t).is_yaokyuu() { 4 } else { 2 })  // 幺九牌4符、中張牌2符
            .sum::<u8>();
        // 暗槓の符計算
        fu += self
            .sup
            .ankans
            .iter()
            .map(|&t| if must_tile!(t).is_yaokyuu() { 32 } else { 16 })  // 幺九牌32符、中張牌16符
            .sum::<u8>();
        // 明槓の符計算
        fu += self
            .sup
            .minkans
            .iter()
            .map(|&t| if must_tile!(t).is_yaokyuu() { 16 } else { 8 })  // 幺九牌16符、中張牌8符
            .sum::<u8>();

        // 雀頭の符計算
        if matches_tu8!(self.pair_tile, P | F | C) {
            // 三元牌の雀頭は2符
            fu += 2;
        } else {
            // As per [Tenhou's rule](https://tenhou.net/man/#RULE):
            //
            // > 連風牌は4符
            // 風牌の雀頭
            if self.pair_tile == self.sup.bakaze {
                fu += 2;  // 場風牌
            }
            if self.pair_tile == self.sup.jikaze {
                fu += 2;  // 自風牌
            }
        }

        // 20符の特殊処理
        if fu == 20 {
            return if !self.sup.is_menzen {
                30  // 喰い平和形：30符
            } else if has_pinfu {
                if self.sup.is_ron { 30 } else { 20 }  // 門前平和：ロン30符、ツモ20符
            } else if self.sup.is_ron {
                40  // その他の門前20符ロン：40符
            } else {
                30  // その他の門前20符ツモ：30符
            };
        }

        // 和了形態の符
        if !self.sup.is_ron {
            fu += 2;   // ツモ符2符
        } else if self.sup.is_menzen {
            fu += 10;  // 門前ロン10符
        }

        // 待ちの符
        if !self.winning_tile_makes_minkou {
            if self.pair_tile == self.sup.winning_tile {
                // tanki wait
                // 単騎待ち：2符
                fu += 2;
            } else {
                // 嵌張待ち・辺張待ちの判定
                let is_kanchan_penchan = self.menzen_shuntsu.iter().any(|&s| {
                    s + 1 == self.sup.winning_tile                    // 嵌張（例：46で5待ち）
                        || s % 9 == 0 && s + 2 == self.sup.winning_tile   // 辺張（12で3待ち）
                        || s % 9 == 6 && s == self.sup.winning_tile       // 辺張（78で7待ち）
                });
                if is_kanchan_penchan {
                    fu += 2;
                }
            }
        }

        // 10符単位に切り上げ
        ((fu - 1) / 10 + 1) * 10
    }

    // 役検索の主処理
    // RETURN_IF_ANY: trueの場合は1つでも役があれば即座にreturn（高速判定用）
    fn search_yakus<const RETURN_IF_ANY: bool>(&self) -> Option<Agari> {
        let mut han = 0;      // 飼数カウンタ
        let mut yakuman = 0;  // 役満カウンタ

        // 平和の判定
        let has_pinfu = self.menzen_shuntsu.len() == 4                      // 4つの順子
            && !matches_tu8!(self.pair_tile, P | F | C)                     // 雀頭が三元牌でない
            && self.pair_tile != self.sup.bakaze                            // 雀頭が場風でない
            && self.pair_tile != self.sup.jikaze                            // 雀頭が自風でない
            && self.menzen_shuntsu.iter().any(|&s| {                        // 両面待ちである
                let num = s % 9 + 1;                                         // 順子の2番目の牌（1〜7）
                num <= 6 && s == self.sup.winning_tile                      // 順子の最初が和了牌（両面待ちの下側）
                || num >= 2 && s + 2 == self.sup.winning_tile               // 順子の最後が和了牌（両面待ちの上側）
            });

        // 結果を返すマクロ
        macro_rules! make_return {
            () => {
                return if yakuman > 0 {
                    Some(Agari::Yakuman(yakuman))  // 役満
                } else if han > 0 {
                    // 符計算：高速判定時または5飼以上の場合は省略
                    let fu = if RETURN_IF_ANY || han >= 5 {
                        0
                    } else {
                        self.calc_fu(has_pinfu)
                    };
                    Some(Agari::Normal { fu, han })
                } else {
                    None  // 役なし
                };
            };
        }
        // 早期リターンをチェックするマクロ
        macro_rules! check_early_return {
            ($($block:tt)*) => {{
                $($block)*;  // 役を加算
                if RETURN_IF_ANY {
                    make_return!();  // 高速判定時は即座にreturn
                }
            }};
        }

        // パターンベースの役判定（分割情報から直接判定できるもの）
        if has_pinfu {
            // 平和
            check_early_return! { han += 1 };
        }
        if self.div.has_chitoi {
            // 七対子
            check_early_return! { han += 2 };
        }
        if self.div.has_ryanpeikou {
            // 二盃口
            check_early_return! { han += 3 };
        }
        if self.div.has_chuuren {
            // 九蓮宝燈
            check_early_return! { yakuman += 1 };
        }

        // 断幺九の判定（幺九牌を含まない）
        let has_tanyao = if self.div.has_chitoi {
            // 七対子形：全ての対子が中張牌
            self.chitoi_pairs().all(|t| {
                let kind = t / 9;  // 0:萬、1:筒、2:索、3:字牌
                let num = t % 9;   // 0〜8 (1〜9に対応)
                kind < 3 && num > 0 && num < 8  // 数牌の2〜8
            })
        } else {
            // 通常形：全ての順子が234〜678、全ての刻子・雀頭が中張牌
            self.all_shuntsu().all(|s| {
                let num = s % 9;
                num > 0 && num < 6  // 順子の開始牌が2〜6（234〜678に対応）
            }) && self
                .all_kotsu_and_kantsu()
                .chain(iter::once(self.pair_tile))
                .all(|k| {
                    let kind = k / 9;
                    let num = k % 9;
                    kind < 3 && num > 0 && num < 8  // 数牌の2〜8
                })
        };
        if has_tanyao {
            // 断幺九
            check_early_return! { han += 1 };
        }

        // 対々和の判定（全て刻子・槓子）
        let has_toitoi =
            !self.div.has_chitoi && self.menzen_shuntsu.is_empty() && self.sup.chis.is_empty();
        if has_toitoi {
            // 対々和
            check_early_return! { han += 2 };
        }

        // 一色系の役判定（清一色、混一色、字一色）
        let mut isou_kind = None;               // 数牌の色（0:萬、1:筒、2:索）
        let mut has_jihai = false;              // 字牌を含むか
        let mut is_chinitsu_or_honitsu = true;  // 同一色か
        // 各牌をチェックして色を判定するクロージャ
        let iter_fn = |&m: &u8| {
            let kind = m / 9;
            if kind >= 3 {
                has_jihai = true;  // 字牌を含む
                return true;
            }
            // 数牌の場合、色が統一されているかチェック
            if let Some(prev_kind) = isou_kind {
                if prev_kind != kind {
                    is_chinitsu_or_honitsu = false;  // 色が混在
                    return false;
                }
            } else {
                isou_kind = Some(kind);  // 最初の数牌の色を記録
            }
            true
        };
        // 全ての牌をチェック
        if self.div.has_chitoi {
            self.chitoi_pairs().take_while(iter_fn).for_each(drop);
        } else {
            self.all_mentsu()
                .chain(iter::once(self.pair_tile))
                .take_while(iter_fn)
                .for_each(drop);
        }
        if isou_kind.is_none() {
            // 字一色（数牌がない）
            check_early_return! { yakuman += 1 };
        } else if is_chinitsu_or_honitsu {
            // 混一色（字牌あり：門前2飼、副露2飼）、清一色（字牌なし：門前6飼、副露5飼）
            let n = if has_jihai { 2 } else { 5 } + self.sup.is_menzen as u8;
            check_early_return! { han += n };
        }

        if !self.div.has_chitoi {
            // 一盃口の判定
            if self.div.has_ipeikou {
                check_early_return! { han += 1 };
            } else if !self.sup.ankans.is_empty()
                && self.sup.is_menzen
                && self.menzen_shuntsu.len() >= 2
            {
                // 暗槓がある場合はdiv.has_ipeikouが不完全なので、再度チェック
                let mut shuntsu_marks = [0_u8; 3];  // 各色の順子ビットマップ
                let has_ipeikou = self.menzen_shuntsu.iter().any(|&t| {
                    let kind = t as usize / 9;
                    let num = t % 9;
                    let mark = &mut shuntsu_marks[kind];
                    if (*mark >> num) & 0b1 == 0b1 {
                        true  // 同じ順子が既に存在
                    } else {
                        *mark |= 0b1 << num;  // 初めての順子をマーク
                        false
                    }
                });
                if has_ipeikou {
                    check_early_return! { han += 1 };
                }
            }

            // 一気通貫の判定（123、456、789の順子を同一色で）
            if self.sup.is_menzen && self.div.has_ittsuu {
                // 門前で分割情報に一気通貫フラグがある場合
                check_early_return! { han += 2 };
            } else if self.sup.chis.is_empty() && self.div.has_ittsuu {
                // 副露でチーがなく、分割情報に一気通貫フラグがある場合
                check_early_return! { han += 1 };
            } else if self.menzen_shuntsu.len() + self.sup.chis.len() >= 3 {
                // 順子が3つ以上ある場合、手動でチェック
                let mut kinds = [0; 3];  // 各色の123、456、789の有無をビットで管理
                for s in self.all_shuntsu() {
                    let kind = s as usize / 9;
                    let num = s % 9;
                    match num {
                        0 => kinds[kind] |= 0b001,  // 123
                        3 => kinds[kind] |= 0b010,  // 456
                        6 => kinds[kind] |= 0b100,  // 789
                        _ => (),
                    };
                }
                // いずれかの色で123、456、789が全て揃っている
                if kinds.contains(&0b111) {
                    check_early_return! { han += 1 };
                }
            }

            // 三色同順・三色同刻の判定
            let mut s_counter = [0; 9];  // 各数字でどの色の順子があるか
            for s in self.all_shuntsu() {
                let kind = s / 9;
                let num = s % 9;
                s_counter[num as usize] |= 0b1 << kind;  // その数字でその色の順子をマーク
            }
            if s_counter.contains(&0b111) {
                // 三色同順（同じ数字の順子が3色全てにある）
                let n = if self.sup.is_menzen { 2 } else { 1 };  // 門前2飼、副露1飼
                check_early_return! { han += n };
            } else {
                // 三色同刻のチェック
                let mut k_counter = [0; 9];  // 各数字でどの色の刻子があるか
                for k in self.all_kotsu_and_kantsu() {
                    let kind = k / 9;
                    if kind < 3 {  // 数牌のみ（字牌は除外）
                        let num = k % 9;
                        k_counter[num as usize] |= 1 << kind;
                    }
                }
                if k_counter.contains(&0b111) {
                    // 三色同刻（同じ数字の刻子が3色全てにある）
                    check_early_return! { han += 2 };
                }
            }

            // 暗刻系の役判定
            let ankous_count = self.sup.ankans.len() + self.menzen_kotsu.len()
                - self.winning_tile_makes_minkou as usize;  // ロンで明刻化した場合を除外
            match ankous_count {
                // 四暗刻
                4 => check_early_return! { yakuman += 1 },
                // 三暗刻
                3 => check_early_return! { han += 2 },
                _ => (),
            };

            // 槓子系の役判定
            let kans_count = self.sup.ankans.len() + self.sup.minkans.len();
            match kans_count {
                // 四槓子
                4 => check_early_return! { yakuman += 1 },
                // 三槓子
                3 => check_early_return! { han += 2 },
                _ => (),
            };

            // 緑一色の判定（緑の牌のみ：索子の2,3,4,6,8と発）
            let has_ryuisou = self
                .all_kotsu_and_kantsu()
                .chain(iter::once(self.pair_tile))
                .all(|k| matches_tu8!(k, 2s | 3s | 4s | 6s | 8s | F))  // 刻子・雀頭が全て緑牌
                && self.all_shuntsu().all(|s| s == tu8!(2s));           // 順子は234sのみ可能
            if has_ryuisou {
                // 緑一色
                check_early_return! { yakuman += 1 };
            }

            if !has_tanyao {
                // 役牌 + 大小三元四喜
                let mut has_jihai = [false; 7];  // 字牌の刻子・槓子を記録（東南西北白発中）
                for k in self.all_kotsu_and_kantsu() {
                    if k >= 3 * 9 {
                        has_jihai[k as usize - 3 * 9] = true;
                    }
                }
                // 場風牌の役牌判定
                if has_jihai[self.sup.bakaze as usize - 3 * 9] {
                    // 役牌:場風牌
                    check_early_return! { han += 1 };
                }
                // 自風牌の役牌判定
                if has_jihai[self.sup.jikaze as usize - 3 * 9] {
                    // 役牌:自風牌
                    check_early_return! { han += 1 };
                }

                // 三元牌の役判定
                let saneins = (4..7).filter(|&i| has_jihai[i]).count() as u8;  // 白発中の刻子数
                if saneins > 0 {
                    // 役牌:三元牌（各三元牌を1飼ずつ）
                    check_early_return! { han += saneins };
                    if saneins == 3 {
                        // 大三元（三元牌全てを刻子・槓子で）
                        check_early_return! { yakuman += 1 };
                    } else if saneins == 2 && matches_tu8!(self.pair_tile, P | F | C) {
                        // 小三元（2つを刻子・槓子、1つを雀頭で）
                        check_early_return! { han += 2 };
                    }
                }

                // 四喜牌の役判定
                let winds = (0..4).filter(|&i| has_jihai[i]).count();  // 東南西北の刻子数
                #[allow(clippy::if_same_then_else)]  // コードの意図を明確にするため
                if winds == 4 {
                    // 大四喜（風牌全てを刻子・槓子で）
                    check_early_return! { yakuman += 1 };
                } else if winds == 3 && matches_tu8!(self.pair_tile, E | S | W | N) {
                    // 小四喜（3つを刻子・槓子、1つを雀頭で）
                    check_early_return! { yakuman += 1 };
                }
            }
        }

        // 幺九牌系の役判定（断幺九でない場合のみ）
        if !has_tanyao {
            let mut has_jihai = false;
            // 幺九牌かどうかを判定するクロージャ
            let is_yaokyuu = |k| {
                let kind = k / 9;
                if kind >= 3 {
                    has_jihai = true;  // 字牌を含む
                    true
                } else {
                    let num = k % 9;
                    num == 0 || num == 8  // 1または9
                }
            };
            // 全ての刻子・槓子・雀頭が幺九牌かチェック
            let is_junchan_or_chanta_or_chinroutou_or_honroutou = if self.div.has_chitoi {
                self.chitoi_pairs().all(is_yaokyuu)  // 七対子の場合
            } else {
                self.all_kotsu_and_kantsu()
                    .chain(iter::once(self.pair_tile))
                    .all(is_yaokyuu)  // 通常形の場合
            };
            if is_junchan_or_chanta_or_chinroutou_or_honroutou {
                if self.div.has_chitoi || has_toitoi {
                    // 刻子・対子のみの形
                    if has_jihai {
                        // 混老頭（幺九牌の刻子・対子のみ、字牌あり）
                        check_early_return! { han += 2 };
                    } else {
                        // 清老頭（老頭牌の刻子・対子のみ）
                        check_early_return! { yakuman += 1 };
                    }
                } else {
                    // 順子を含む形
                    let is_junchan_or_chanta = self.all_shuntsu().all(|s| {
                        let num = s % 9;
                        num == 0 || num == 6  // 123または789の順子
                    });
                    if is_junchan_or_chanta {
                        // 混全帯幺九（字牌あり：門前2飼、副露1飼）、純全帯幺九（字牌なし：門前3飼、副露2飼）
                        let n = if has_jihai { 1 } else { 2 } + self.sup.is_menzen as u8;
                        check_early_return! { han += n };
                    }
                }
            }
        }

        // 最終的な結果を返す
        make_return!();
    }
}

// 和了テーブルの初期化を保証（テスト用）
pub fn ensure_init() {
    assert_eq!(AGARI_TABLE.len(), AGARI_TABLE_SIZE);
}

// 手牌配列からキーエンコーディングと実際の牌ID配列を生成
// tiles: 34種×枚数の配列
// 返値: (14枚の牌ID配列, 検索用キー)
fn get_tile14_and_key(tiles: &[u8; 34]) -> ([u8; 14], u32) {
    let mut tile14 = [0; 14];               // 14枚の牌IDを格納する配列
    let mut tile14_iter = tile14.iter_mut(); // tile14への書き込み用イテレータ
    let mut key = 0;                         // ビットパターンキー

    let mut bit_idx = -1;                    // 現在のビット位置
    let mut prev_in_hand = None;             // 直前に手牌にあった牌があるか
    // 数牌（萬子・筒子・索子）の処理
    for (kind, chunk) in tiles.chunks_exact(9).enumerate() {
        for (num, c) in chunk.iter().copied().enumerate() {
            if c > 0 {
                prev_in_hand = Some(());  // 手牌に牌があることを記録
                *tile14_iter.next().unwrap() = (kind * 9 + num) as u8;  // 牌IDをtile14に追加
                bit_idx += 1;  // 牌の存在を示すビット

                // 枚数に応じたビットパターンを追加
                match c {
                    2 => {
                        key |= 0b11 << bit_idx;      // 2枚：11
                        bit_idx += 2;
                    }
                    3 => {
                        key |= 0b1111 << bit_idx;    // 3枚：1111
                        bit_idx += 4;
                    }
                    4 => {
                        key |= 0b11_1111 << bit_idx; // 4枚：111111
                        bit_idx += 6;
                    }
                    // 1枚の場合は追加ビットなし
                    _ => (),
                }
            } else if prev_in_hand.take().is_some() {
                // 直前に牌があったが現在はない場合、区切りビットを追加
                key |= 0b1 << bit_idx;
                bit_idx += 1;
            }
        }
        // 各色の終わりに区切りビットを追加
        if prev_in_hand.take().is_some() {
            key |= 0b1 << bit_idx;
            bit_idx += 1;
        }
    }

    // 字牌の処理
    tiles
        .iter()
        .enumerate()
        .skip(3 * 9)  // 字牌は27番目から
        .filter(|&(_, &c)| c > 0)
        .for_each(|(tile_id, &c)| {
            *tile14_iter.next().unwrap() = tile_id as u8;  // 牌IDをtile14に追加
            bit_idx += 1;  // 牌の存在を示すビット

            // 枚数に応じたビットパターンを追加
            match c {
                2 => {
                    key |= 0b11 << bit_idx;
                    bit_idx += 2;
                }
                3 => {
                    key |= 0b1111 << bit_idx;
                    bit_idx += 4;
                }
                4 => {
                    key |= 0b11_1111 << bit_idx;
                    bit_idx += 6;
                }
                // 1枚の場合は追加ビットなし
                _ => (),
            }
            // 字牌は各牌の後に必ず区切りビットを追加
            key |= 0b1 << bit_idx;
            bit_idx += 1;
        });

    (tile14, key)
}

/// `tehai` must already contain `tile`. `true` is returned if making an ankan
/// with the tile is legal under the riichi'd `tehai`.
///
/// If `strict` is `false`, it is the same as [Tenhou's
/// rule](https://tenhou.net/man/#RULE):
///
/// > リーチ後の暗槓は待ちが変わらない場合のみ。送り槓不可、牌姿や役の増減は不
/// > 問。
///
/// If `strict` is `true`, it will also check the shape of tenpai and agari, but
/// will not check yaku anyways.
///
/// The behavior is undefined if `tehai` is not tenpai.
// 立直後の暗槓が可能かどうかを判定
// tehai: 手牌（ツモった牌を含む）
// len_div3: ((手牌枚数-2)/3)の値
// tile: 槓したい牌
// strict: trueの場合は手牌の形も変わらないことを確認
#[must_use]
pub fn check_ankan_after_riichi(tehai: &[u8; 34], len_div3: u8, tile: Tile, strict: bool) -> bool {
    let tile_id = tile.deaka().as_usize();  // 赤牌を通常牌に変換
    if tehai[tile_id] != 4 {
        // 4枚ないと槓できない
        return false;
    }

    if tile_id >= 3 * 9 {
        // 字牌は常に槓可能（待ちが変わることがない）
        return true;
    }

    // ツモる前の手牌を再現
    let mut tehai_before_tsumo = *tehai;
    tehai_before_tsumo[tile_id] -= 1;

    // 全ての待ち牌についてチェック
    (0..34)
        .filter(|&t| {
            if tehai_before_tsumo[t] == 4 {
                return false;  // 既に4枚ある牌は待ち牌になり得ない
            }
            // Get all waits of the original hand
            // 元の手牌の待ち牌を取得
            let mut tmp = tehai_before_tsumo;
            tmp[t] += 1;
            shanten::calc_all(&tmp, len_div3) == -1  // テンパイになる牌
        })
        .all(|wait| {
            // Cannot kan a waited tile
            // 待ち牌自体を槓することはできない（送り槓禁止）
            if wait == tile_id {
                return false;
            }

            // Test if the hand after ankan can also win with the wait tile
            // 暗槓後も同じ待ち牌で和了できるかチェック
            let mut tehai_after = *tehai;
            tehai_after[tile_id] = 0;  // 槓した牌を0枚に
            tehai_after[wait] += 1;    // 待ち牌を追加
            let (_, key) = get_tile14_and_key(&tehai_after);
            let Some(divs_after) = AGARI_TABLE.get(&key) else {
                // The wait tile set will get smaller after kan.
                // 槓後に和了形でなくなる場合
                return false;
            };

            if strict {
                // Compare if the number of hand divisions are equal before and
                // after ankan, which indicates the shapes of tenpai and agari
                // will not change after ankan. This is implemented by inserting
                // the waited tile to both of them.
                // strictモード：手牌の形が変わらないことも確認
                let mut tehai_before = tehai_before_tsumo;
                tehai_before[wait] += 1;
                let (_, key) = get_tile14_and_key(&tehai_before);
                let divs_before = AGARI_TABLE
                    .get(&key)
                    .expect("invalid riichi detected when testing ankan after riichi");

                // 分割パターン数が変わらないことを確認
                if divs_after.len() != divs_before.len() {
                    return false;
                }
            }

            true
        })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;

    #[test]
    fn ankan_after_riichi() {
        let test_one = |tehai_str, tile_str: &str, len_div3, strict, expected| {
            let mut tehai = hand(tehai_str).unwrap();
            let tile: Tile = tile_str.parse().unwrap();
            tehai[tile.as_usize()] += 1;
            assert_eq!(
                check_ankan_after_riichi(&tehai, len_div3, tile, strict),
                expected,
                "failed for {tehai_str} + {tile_str}, expected {expected}",
            );
        };

        // Always positive
        test_one("12345m 567s 11222z", "S", 4, true, true);
        test_one("12345m 444567s 11z", "4s", 4, true, true);
        test_one("22m 11112356p 444s", "4s", 4, true, true);

        // Always negative
        test_one("123456m 4445s 111z", "4s", 4, true, false);
        test_one("123456m 4445s 111z", "4s", 4, false, false);

        // Shape of tenpai changes
        test_one("1113444p 222z", "1p", 3, true, false);
        test_one("1113444p 222z", "1p", 3, false, true);
        test_one("1113444p 222z", "4p", 3, true, false);
        test_one("1113444p 222z", "S", 3, true, true);

        // Shape of agari changes
        test_one("23m 999p 33345666s", "3s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, false, true);
        test_one("23m 999p 33345666s", "9p", 4, true, true);

        // The 1m kan will make chuuren gone, but in this impl we don't take
        // yaku into account.
        test_one("1113445678999m", "1m", 4, true, true);
        test_one("1113445678999m", "9m", 4, true, false);
    }

    #[test]
    fn agari_calc() {
        let tehai = hand("2234455m 234p 234s 3m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(S),
            winning_tile: tu8!(3m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        assert_eq!(yaku, Agari::Normal { fu: 40, han: 4 });

        let tehai = hand("12334m 345p 22s 777z 2m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(3m),
            is_ron: false,
        };
        let points = calc.agari(2, 0).unwrap().point(true);
        // 立直, 門前清自摸和
        assert_eq!(
            points,
            Point {
                ron: 7700,
                tsumo_oya: 0,
                tsumo_ko: 2600
            }
        );

        let tehai = hand("2255m 445p 667788s 5p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(S),
            winning_tile: tu8!(5p),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        assert_eq!(yaku, Agari::Normal { fu: 25, han: 3 });
        assert_eq!(yaku.point(false).ron, 3200);

        let tehai = hand("22334m 33p 4m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &tu8![2s, 2s],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(S),
            winning_tile: tu8!(4m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        assert_eq!(yaku, Agari::Normal { fu: 30, han: 1 });

        let tehai = hand("223344p 667788s 3m 3m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(S),
            jikaze: tu8!(N),
            winning_tile: tu8!(3m),
            is_ron: false,
        };
        let yaku = calc.search_yakus().unwrap();
        assert_eq!(yaku, Agari::Normal { fu: 30, han: 4 });

        let tehai = hand("234678m 1123488p 8p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(8p),
            is_ron: true,
        };
        assert_eq!(calc.search_yakus(), None);

        let tehai = hand("223344999m 1188p 8p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(8p),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 一盃口 (without ankan)
        assert_eq!(yaku, Agari::Normal { fu: 40, han: 1 });

        let tehai = hand("223344m 1188p 8p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &tu8![9m,],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(8p),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 一盃口 (with ankan)
        assert_eq!(yaku, Agari::Normal { fu: 70, han: 1 });

        let tehai = hand("55566677m 11p 7m").unwrap();
        let mut calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &tu8![9s,],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(7m),
            is_ron: false,
        };
        let yaku = calc.search_yakus().unwrap();
        // 四暗刻
        assert_eq!(yaku, Agari::Yakuman(1));

        calc.is_ron = true;
        let yaku = calc.search_yakus().unwrap();
        // 三暗刻, 対々和
        assert_eq!(yaku, Agari::Normal { fu: 80, han: 4 });

        let tehai = hand("666677778888m 99p").unwrap();
        let mut calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(8m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 平和, 二盃口
        assert_eq!(yaku, Agari::Normal { fu: 30, han: 4 });

        calc.winning_tile = tu8!(7m);
        let yaku = calc.search_yakus().unwrap();
        // 二盃口
        assert_eq!(yaku, Agari::Normal { fu: 40, han: 3 });

        let tehai = hand("12345678m 11p 9m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &tu8![9p,],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(9m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 一気通貫
        assert_eq!(yaku, Agari::Normal { fu: 70, han: 2 });

        let tehai = hand("12345678m 11p 9m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &[],
            pons: &tu8![9p,],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(9m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 一気通貫
        assert_eq!(yaku, Agari::Normal { fu: 30, han: 1 });

        let tehai = hand("111222333m 67p 88s 8p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(8p),
            is_ron: false,
        };
        let yaku = calc.search_yakus().unwrap();
        // 門前清自摸和 is not accounted.
        assert_eq!(yaku, Agari::Normal { fu: 40, han: 2 });

        let tehai = hand("1112223334447z 7z").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(C),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        assert_eq!(yaku, Agari::Yakuman(3));

        let tehai = hand("1m 789p 789s 1m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &tu8![7m, 1s],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(1m),
            is_ron: false,
        };
        let yaku = calc.search_yakus().unwrap();
        // 純全, 三色
        assert_eq!(yaku, Agari::Normal { fu: 30, han: 3 });

        let tehai = hand("111444m 45556s 22z 5s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(S),
            jikaze: tu8!(S),
            winning_tile: tu8!(5s),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 三暗刻 (5s is ankou)
        // 20 + menzenron(10) + kanchan(2) + bakaze(2) + jikaze(2) + 1m(8) + 4m(4)
        // + 5s(4) + = 52
        assert_eq!(yaku, Agari::Normal { fu: 60, han: 2 });

        let tehai = hand("999s 1777z 1z").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &tu8![1p,],
            pons: &tu8![N,],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(S),
            jikaze: tu8!(S),
            winning_tile: tu8!(E),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 混全帯幺九, 役牌*1
        // 20 + tanki(2) + 9s(8) + 7z(8) + 4z(4) = 42
        assert_eq!(yaku, Agari::Normal { fu: 50, han: 2 });

        let tehai = hand("1119m 9m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &[],
            pons: &tu8![S, C],
            minkans: &[],
            ankans: &tu8![N,],
            bakaze: tu8!(S),
            jikaze: tu8!(N),
            winning_tile: tu8!(9m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 混一色, 混老頭, 役牌*3, 対々和
        assert!(matches!(yaku, Agari::Normal { han: 9, .. }));
        let (tile14, key) = get_tile14_and_key(&tehai);
        let divs = AGARI_TABLE.get(&key).unwrap();
        let fu = divs
            .iter()
            .map(|div| DivWorker::new(&calc, &tile14, div))
            .map(|w| w.calc_fu(false))
            .max()
            .unwrap();
        // 20 + tanki(2) + 1m(8) + 2z(4) + 7z(4) + 4z(32) = 70
        assert_eq!(fu, 70);

        // This shape is called 八蓮宝燈, waiting on 12456789
        let tehai = hand("1233334567888m 9m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(9m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 清一色, 一気通貫
        assert!(matches!(yaku, Agari::Normal { han: 8, .. }));

        // This shape is called 七蓮宝燈, waiting on 1235789
        let tehai = hand("2344445666678p 5p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(5p),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 清一色, 断么九
        assert!(matches!(yaku, Agari::Normal { han: 7, .. }));

        // Waits on 13467s
        let tehai = hand("2223445566s 1s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: false,
            chis: &tu8![7s,],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(1s),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 清一色, 一気通貫
        assert!(matches!(yaku, Agari::Normal { han: 6, .. }));

        // Waits on 14m
        let tehai = hand("1123444m 111p 111s 1m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(E),
            jikaze: tu8!(E),
            winning_tile: tu8!(1m),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 清一色, 一気通貫
        assert_eq!(yaku, Agari::Normal { fu: 60, han: 2 });

        let tehai = hand("111s 2225556677z 7z").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            is_menzen: true,
            chis: &[],
            pons: &[],
            minkans: &[],
            ankans: &[],
            bakaze: tu8!(S),
            jikaze: tu8!(S),
            winning_tile: tu8!(C),
            is_ron: true,
        };
        let yaku = calc.search_yakus().unwrap();
        // 三暗刻, 対々和, 混一色, 混老頭, 小三元, double 南, 白, 中
        assert!(matches!(yaku, Agari::Normal { han: 15, .. }));
    }
}
