// ファイル概要：
// このファイルはGRP（Game Record Parser）の実装です。
// 麻雀ゲームの各局の情報（場風、局数、本場、供託、各プレイヤーの得点）を
// パースして機械学習用の特徴量として保存します。また最終的な順位と得点も記録します。

// GRP特徴量のサイズ定数（通常は7: 通算局数、本場、供託、4人分の得点）
use crate::consts::GRP_SIZE;
// mjaiイベント定義
use crate::mjai::Event;
// 順位計算用の構造体
use crate::rankings::Rankings;
// 牌を表すu8型への変換マクロ
use crate::tu8;
// ベクトルの要素ごとの加算関数
use crate::vec_ops::vec_add_assign;
// ファイル操作
use std::fs::File;
// 入出力処理
use std::io;
// メモリ操作（所有権の移動）
use std::mem;

// エラーハンドリング（Context、Result）
use anyhow::{Context, Result};
// gzip圧縮ファイルの読み込み
use flate2::read::GzDecoder;
// 多次元配列処理
use ndarray::prelude::*;
// NumPy配列とのインターフェース
use numpy::PyArray2;
// Python（PyO3）バインディング
use pyo3::prelude::*;
// Pythonから渡された文字列のバックエンド
use pyo3::pybacked::PyBackedStr;
// 並列処理
use rayon::prelude::*;
// JSONパース
use serde_json as json;
// 固定サイズの小さな配列用
use tinyvec::array_vec;

// GRP（Game Record Parser）構造体：ゲーム記録の解析結果を保持
#[pyclass]  // Pythonクラスとして公開
#[derive(Clone, Default)]  // Clone可能、デフォルト値を持つ
pub struct Grp {
    // [grand_kyoku, honba, kyotaku, [score[i] / 10000]] where i is player_id
    // 特徴量配列：[通算局数, 本場, 供託, 各プレイヤーの得点（万点単位）]
    // 各行が1局分の情報を表す
    pub feature: Array2<f64>,
    
    // 各プレイヤーの最終順位（0が1位、3が4位）
    pub rank_by_player: [u8; 4],
    
    // 各プレイヤーの最終得点
    pub final_scores: [i32; 4],
}

// GrpのPythonインターフェース実装
#[pymethods]
impl Grp {
    // 静的メソッド：生のログ文字列からGRPデータを読み込む
    #[staticmethod]
    fn load_log(raw_log: &str) -> Result<Self> {
        // ログを行ごとに分割してJSONパース
        let events = raw_log
            .lines()                                      // 行ごとに分割
            .map(json::from_str)                         // 各行をJSONとしてパース
            .collect::<Result<Vec<Event>, _>>()          // 結果をVec<Event>に収集
            .context("failed to parse log")?;            // エラー時のコンテキストを追加
        // イベントリストからGRPデータを生成
        Self::load_events(&events)
    }

    // 静的メソッド：gzip圧縮されたログファイルを読み込む（Python向けラッパー）
    #[staticmethod]
    #[pyo3(name = "load_gz_log_files")]  // Python側での名前を指定
    fn load_gz_log_files_py(gzip_filenames: Vec<PyBackedStr>) -> Result<Vec<Self>> {
        // 実際の処理は同名のRustメソッドに委譲
        Self::load_gz_log_files(gzip_filenames)
    }

    /// Returns List[List[np.ndarray]]
    /// 特徴量配列を取り出してPythonのNumPy配列として返す
    pub fn take_feature<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // mem::takeで所有権を移動（self.featureは空になる）
        PyArray2::from_owned_array(py, mem::take(&mut self.feature))
    }
    
    // 各プレイヤーの順位を返す（constメソッドなので不変）
    pub const fn take_rank_by_player(&self) -> [u8; 4] {
        self.rank_by_player
    }
    
    // 各プレイヤーの最終得点を返す（constメソッドなので不変）
    pub const fn take_final_scores(&self) -> [i32; 4] {
        self.final_scores
    }
}

// GrpのRust側実装
impl Grp {
    // 特徴量配列の行数（局数）を返す
    #[inline]  // インライン展開を推奨（パフォーマンス向上）
    pub fn len(&self) -> usize {
        // 0軸（行方向）の長さを取得
        self.feature.len_of(Axis(0))
    }

    // 特徴量配列が空かどうかを判定
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // 複数のgzip圧縮ログファイルを並列で読み込む
    pub fn load_gz_log_files<V, S>(gzip_filenames: V) -> Result<Vec<Self>>
    where
        V: IntoParallelIterator<Item = S>,      // 並列イテレータに変換可能な型
        S: AsRef<str>,                          // 文字列参照として扱える型
    {
        gzip_filenames
            .into_par_iter()                    // 並列イテレータに変換
            .map(|f| {
                let filename = f.as_ref();      // ファイル名を文字列参照として取得
                // 内部処理をクロージャとして定義（エラー処理を簡潔にするため）
                let inner = || {
                    let file = File::open(filename)?;           // ファイルを開く
                    let gz = GzDecoder::new(file);              // gzipデコーダーを作成
                    let raw = io::read_to_string(gz)?;          // 解凍して文字列として読み込む
                    Self::load_log(&raw)                        // ログをパースしてGRPに変換
                };
                // エラー時にファイル名を含むコンテキストを追加
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()                          // 結果を収集（並列処理の完了を待つ）
    }

    // イベントリストからGRPデータを生成
    pub fn load_events(events: &[Event]) -> Result<Self> {
        // 各局の情報を格納するベクター
        let mut game_info = vec![];
        // 最終順位（オプション型、最初のHoraまたはRyukyokuイベントで確定）
        let mut rank_by_player_opt = None;
        // 最終的な得点変動の累計
        let mut final_deltas = [0; 4];
        // 最終得点
        let mut final_scores = [0; 4];

        // イベントを逆順で処理（最新から過去へ）
        // これにより最終結果を先に確定できる
        for ev in events.iter().rev() {
            match *ev {
                // 和了または流局イベント：得点変動を記録
                Event::Hora { deltas, .. } | Event::Ryukyoku { deltas, .. } => {
                    // まだ順位が確定していない場合（最初の和了/流局）
                    if rank_by_player_opt.is_none() {
                        // deltasフィールドが必須
                        let ds = deltas.context(
                            "invalid log: field `deltas` is required for Hora and Ryukyoku of AL",
                        )?;
                        // 得点変動を累計に加算
                        vec_add_assign(&mut final_deltas, &ds);
                    }
                }
                // リーチ受理イベント：リーチ棒（1000点）を減算
                Event::ReachAccepted { actor } => {
                    if rank_by_player_opt.is_none() {
                        final_deltas[actor as usize] -= 1000;
                    }
                }
                // 局開始イベント：局の情報を記録
                Event::StartKyoku {
                    bakaze,      // 場風（東、南、西）
                    kyoku,       // 局数（1-4）
                    honba,       // 本場数
                    kyotaku,     // 供託（リーチ棒の数）
                    scores,      // 各プレイヤーの得点
                    ..
                } => {
                    // 最初のStartKyokuイベント（最終局）で順位を確定
                    if rank_by_player_opt.is_none() {
                        // 開始時の得点を保存
                        final_scores = scores;
                        // 得点変動を加算して最終得点を計算
                        vec_add_assign(&mut final_scores, &final_deltas);

                        // 順位を計算
                        let rk = Rankings::new(final_scores);

                        // assume the sum of scores to be 100k
                        // 得点の合計が10万点未満の場合、1位に差分を加算
                        // （通常は配給原点が25000点×4=100000点）
                        let sum: i32 = final_scores.iter().sum();
                        if sum < 100_000 {
                            final_scores[rk.player_by_rank[0] as usize] += 100_000 - sum;
                        }

                        // 順位を確定
                        rank_by_player_opt = Some(rk.rank_by_player);
                    }

                    // 局の情報を配列に格納
                    let mut kyoku_info = array_vec!([_; GRP_SIZE]);
                    // 通算局数を計算（東1=0, 東2=1, ..., 南1=4, ..., 西1=8, ...）
                    let grand_kyoku = match bakaze.as_u8() {
                        tu8!(E) => kyoku - 1,     // 東場：0-3
                        tu8!(S) => 3 + kyoku,     // 南場：4-7
                        _ => 7 + kyoku,           // 西場以降：8-
                    };
                    kyoku_info.push(grand_kyoku as f64);        // 通算局数
                    kyoku_info.push(honba as f64);              // 本場数
                    kyoku_info.push(kyotaku as f64);            // 供託数
                    // assume player 0 is the oya at E1
                    // 各プレイヤーの得点を万点単位で追加（東1ではプレイヤー0が親と仮定）
                    kyoku_info.extend(scores.iter().map(|&score| score as f64 / 10000.));
                    // サイズチェック（通常は7要素）
                    assert_eq!(kyoku_info.len(), GRP_SIZE);

                    // 先頭に挿入（逆順処理のため）
                    game_info.insert(0, kyoku_info);
                }
                _ => (),  // その他のイベントは無視
            }
        }

        // 順位が確定していない場合はエラー
        let rank_by_player =
            rank_by_player_opt.context("invalid log: no Hora or Ryukyoku after a StartKyoku")?;
        
        // 2次元配列の形状を定義
        let shape = (game_info.len(), GRP_SIZE);
        // フラット化してから形状を適用して2次元配列を作成
        let feature =
            Array::from_iter(game_info.into_iter().flatten()).into_shape_with_order(shape)?;

        // GRP構造体を構築して返す
        Ok(Self {
            feature,
            rank_by_player,
            final_scores,
        })
    }
}
