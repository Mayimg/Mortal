// このファイルは、PlayerState構造体に対するgetterメソッドを定義しています。
// 主な目的：
// 1. Pythonバインディング（PyO3）を通じてPlayerStateのフィールドへの読み取り専用アクセスを提供
// 2. 麻雀ゲームの状態情報（手牌、鳴き、シャンテン数、待ち牌など）への安全なアクセスを提供
// 3. Python側からアクセスする際のデータ型変換（例：Tile -> String）も実装
//
// メソッドは大きく3つのカテゴリに分かれています：
// - 基本的なフィールドへの直接アクセス（#[getter]属性付き）
// - Python用の変換メソッド（_py接尾辞付き）
// - Rust内部用のメソッド（PyO3マクロなし）

use super::{ActionCandidate, PlayerState};  // 同じモジュールからActionCandidate型とPlayerState型をインポート
use crate::tile::Tile;  // 牌を表すTile型をインポート

use pyo3::prelude::*;  // PyO3のプレリュード（Python拡張機能作成用のマクロとトレイト）をインポート

#[pymethods]  // このブロック内のメソッドがPythonから呼び出し可能になることを示すPyO3マクロ
impl PlayerState {  // PlayerState構造体にメソッドを実装
    #[getter]  // PyO3マクロ：このメソッドをPythonのプロパティ（getter）として公開
    #[inline]  // コンパイラヒント：可能な限りこの関数をインライン化（呼び出し箇所に直接埋め込み）
    #[must_use]  // 警告：戻り値を使用しない場合にコンパイラが警告を出す
    pub const fn player_id(&self) -> u8 {  // プレイヤーID（0-3）を返すconstメソッド（コンパイル時評価可能）
        self.player_id  // 自身のプレイヤーIDフィールドを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn kyoku(&self) -> u8 {  // 現在の局数（東1局=0, 東2局=1, ..., 南4局=7）を返す
        self.kyoku  // 局数フィールドを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn honba(&self) -> u8 {  // 本場数（連荘数）を返す
        self.honba  // 本場数フィールドを返す（流局や親の和了で増加）
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn kyotaku(&self) -> u8 {  // 供託棒の数（リーチ棒の積み重ね）を返す
        self.kyotaku  // 供託棒数フィールドを返す（リーチ宣言で増加、和了で精算）
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn is_oya(&self) -> bool {  // このプレイヤーが親（東家）かどうかを返す
        self.oya == 0  // oyaフィールドが0の場合、このプレイヤーが親（true）
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn tehai(&self) -> [u8; 34] {  // 手牌の種類別枚数を返す（各牌種0-34の枚数配列）
        self.tehai  // 34要素の配列：各インデックスが牌種、値が枚数を表す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn akas_in_hand(&self) -> [bool; 3] {  // 手牌内の赤牌の有無を返す（[5m赤, 5p赤, 5s赤]）
        self.akas_in_hand  // 3要素のbool配列：各種赤五を持っているかtrue/false
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn chis(&self) -> &[u8] {  // チー（順子）の一覧を返す（最小牌のIDが格納）
        &self.chis  // チーした順子の最小牌のIDのスライスを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn pons(&self) -> &[u8] {  // ポン（刻子）の一覧を返す（牌のIDが格納）
        &self.pons  // ポンした牌のIDのスライスを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn minkans(&self) -> &[u8] {  // 明カン（他家からのカン）の一覧を返す（牌のIDが格納）
        &self.minkans  // 明カンした牌のIDのスライスを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn ankans(&self) -> &[u8] {  // 暗カン（手牌からのカン）の一覧を返す（牌のIDが格納）
        &self.ankans  // 暗カンした牌のIDのスライスを返す
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn at_turn(&self) -> u8 {  // 現在のターン数（局内で何巡目か）を返す
        self.at_turn  // 現在の巡数フィールドを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn shanten(&self) -> i8 {  // シャンテン数（和了までの最小必要枚数-1）を返す（-1=テンパイ）
        self.shanten  // シャンテン数フィールドを返す（-1～13の範囲）
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn waits(&self) -> [bool; 34] {  // 待ち牌の一覧を返す（各牌種が和了牌かどうかのbool配列）
        self.waits  // 34要素のbool配列：各牌種が和了待ち牌かtrue/false
    }

    #[inline]  // インライン化推奨
    #[pyo3(name = "last_self_tsumo")]  // Python側でのメソッド名を"last_self_tsumo"に指定
    fn last_self_tsumo_py(&self) -> Option<String> {  // 最後にツモった牌をPython向け文字列として返す
        self.last_self_tsumo.map(|t| t.to_string())  // Option<Tile>をOption<String>に変換（NoneまたはTileの文字列表現）
    }
    #[inline]  // インライン化推奨
    #[pyo3(name = "last_kawa_tile")]  // Python側でのメソッド名を"last_kawa_tile"に指定
    fn last_kawa_tile_py(&self) -> Option<String> {  // 最後に河（捨て牌）に出た牌をPython向け文字列として返す
        self.last_kawa_tile.map(|t| t.to_string())  // Option<Tile>をOption<String>に変換（NoneまたはTileの文字列表現）
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn last_cans(&self) -> ActionCandidate {  // 最後のアクション候補（実行可能な行動の一覧）を返す
        self.last_cans  // ActionCandidate構造体を返す（各種行動の可否情報を含む）
    }

    #[inline]  // インライン化推奨
    #[pyo3(name = "ankan_candidates")]  // Python側でのメソッド名を"ankan_candidates"に指定
    fn ankan_candidates_py(&self) -> Vec<String> {  // 暗カン可能な牌のリストをPython向け文字列ベクタとして返す
        self.ankan_candidates  // 暗カン候補牌のスライスに対して
            .iter()  // イテレータを作成
            .map(|t| t.to_string())  // 各Tileを文字列に変換
            .collect()  // Vec<String>に収集
    }
    #[inline]  // インライン化推奨
    #[pyo3(name = "kakan_candidates")]  // Python側でのメソッド名を"kakan_candidates"に指定
    fn kakan_candidates_py(&self) -> Vec<String> {  // 加カン可能な牌のリストをPython向け文字列ベクタとして返す
        self.kakan_candidates  // 加カン候補牌（既にポンした牌にもう1枚加える）のスライスに対して
            .iter()  // イテレータを作成
            .map(|t| t.to_string())  // 各Tileを文字列に変換
            .collect()  // Vec<String>に収集
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn can_w_riichi(&self) -> bool {  // ダブルリーチ可能かどうかを返す（親の第一ツモテンパイ時）
        self.can_w_riichi  // ダブルリーチ可能フラグを返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn self_riichi_declared(&self) -> bool {  // 自分がリーチを宣言したかどうかを返す
        self.riichi_declared[0]  // riichi_declared配列の0番目（自分）の値を返す
    }
    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn self_riichi_accepted(&self) -> bool {  // 自分のリーチが成立したかどうかを返す（捨て牌が通ったか）
        self.riichi_accepted[0]  // riichi_accepted配列の0番目（自分）の値を返す
    }

    #[getter]  // Pythonプロパティとして公開
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn at_furiten(&self) -> bool {  // フリテン状態かどうかを返す（捨て牌で和了形ができる状態）
        self.at_furiten  // フリテンフラグを返す
    }
}  // Pythonバインディング用のメソッド実装ブロック終了

// Rust内部用のメソッド実装（PyO3マクロなし、Pythonに公開されない）
impl PlayerState {
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn last_self_tsumo(&self) -> Option<Tile> {  // 最後にツモった牌をTile型で返す（Rust内部用）
        self.last_self_tsumo  // Option<Tile>を直接返す（文字列変換なし）
    }
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub const fn last_kawa_tile(&self) -> Option<Tile> {  // 最後に河に出た牌をTile型で返す（Rust内部用）
        self.last_kawa_tile  // Option<Tile>を直接返す（文字列変換なし）
    }

    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn ankan_candidates(&self) -> &[Tile] {  // 暗カン候補牌のTileスライスを返す（Rust内部用）
        &self.ankan_candidates  // Tileスライスの参照を返す（文字列変換なし）
    }
    #[inline]  // インライン化推奨
    #[must_use]  // 戻り値の使用を強制
    pub fn kakan_candidates(&self) -> &[Tile] {  // 加カン候補牌のTileスライスを返す（Rust内部用）
        &self.kakan_candidates  // Tileスライスの参照を返す（文字列変換なし）
    }
}  // Rust内部用メソッドの実装ブロック終了
