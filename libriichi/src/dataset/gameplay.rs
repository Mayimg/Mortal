// ファイル概要：
// このファイルは麻雀ゲームプレイデータの読み込みと解析を行うモジュールです。
// 主にmjai形式のゲームログをパースし、機械学習用の特徴量（観測値）と
// 行動（アクション）のペアを生成します。GameplayLoaderクラスがログの読み込みと
// 設定を管理し、Gameplayクラスが実際のゲームデータを保持します。

// 同じモジュール内のGrp（ゲームレコードパーサー）とInvisible（不可視情報）を使用
use super::{Grp, Invisible};
// チー（鳴き）の種類を定義する列挙型
use crate::chi_type::ChiType;
// mjai形式のイベント定義
use crate::mjai::Event;
// プレイヤーの状態を管理する構造体
use crate::state::PlayerState;
// 配列操作用のユーティリティ
use std::array;
// ファイル操作用
use std::fs::File;
// 入出力処理用
use std::io;
// メモリ操作用（所有権の移動など）
use std::mem;

// 高速なハッシュセット実装
use ahash::AHashSet;
// エラーハンドリング用のライブラリ（Context、Result、bail!マクロ）
use anyhow::{Context, Result, bail};
// カスタマイズ可能なDeriveマクロ（Debug実装のカスタマイズ用）
use derivative::Derivative;
// gzip圧縮ファイルの読み込み用
use flate2::read::GzDecoder;
// 多次元配列処理ライブラリ
use ndarray::prelude::*;
// NumPy配列とのインターフェース（PythonのNumPy配列をRustで扱う）
use numpy::{PyArray1, PyArray2};
// Python（PyO3）バインディング用のプレリュード
use pyo3::prelude::*;
// 並列処理用のライブラリ
use rayon::prelude::*;
// JSONパース用（mjaiログはJSON形式）
use serde_json as json;
// 固定サイズの小さな配列/ベクター用（スタック上で動作）
use tinyvec::ArrayVec;

// GameplayLoaderクラス：ゲームログの読み込みと設定を管理
#[pyclass] // Pythonクラスとして公開
#[derive(Derivative)] // Derivativeマクロでカスタマイズ可能なDeriveを実装
#[derivative(Debug)] // Debug trait実装をカスタマイズ
pub struct GameplayLoader {
    // 特徴量エンコーディングのバージョン番号
    #[pyo3(get)] // Pythonから読み取り可能な属性として公開
    version: u32,
    
    // オラクル（全情報が見える）モードかどうか
    #[pyo3(get)]
    oracle: bool,
    
    // 解析対象のプレイヤー名リスト（指定された場合のみ解析）
    #[pyo3(get)]
    player_names: Vec<String>,
    
    // 除外するプレイヤー名リスト
    #[pyo3(get)]
    excludes: Vec<String>,
    
    // シード値を信頼するかどうか（山牌の並び順を正確に再現するか）
    #[pyo3(get)]
    trust_seed: bool,
    
    // 槓（カン）選択を常に含めるかどうか
    #[pyo3(get)]
    always_include_kan_select: bool,
    
    // データ拡張（augmentation）を行うかどうか
    #[pyo3(get)]
    augmented: bool,

    // player_namesのハッシュセット版（高速な検索用）
    #[derivative(Debug = "ignore")] // Debug出力時は無視（重複情報のため）
    player_names_set: AHashSet<String>,
    
    // excludesのハッシュセット版（高速な検索用）
    #[derivative(Debug = "ignore")]
    excludes_set: AHashSet<String>,
}

// Gameplayクラス：実際のゲームプレイデータを保持
#[pyclass] // Pythonクラスとして公開
#[derive(Clone, Default)] // Clone（複製可能）とDefault（デフォルト値）を実装
pub struct Gameplay {
    // === 各手番（move）ごとのデータ ===
    
    // 観測値（特徴量）：各手番での盤面状態を表す2次元配列のリスト
    // 形状: [手番数, チャネル数, 特徴次元]
    pub obs: Vec<Array2<f32>>,
    
    // 不可視観測値：オラクルモード時の他家の手牌など見えない情報
    // 形状: [手番数, チャネル数, 特徴次元]
    pub invisible_obs: Vec<Array2<f32>>,
    
    // 実行したアクション（行動）のID：0-36=打牌、37=リーチ、38-40=チー、41=ポン、42=カン、43=ロン、44=九種九牌、45=パス
    pub actions: Vec<i64>,
    
    // アクションマスク：各アクションが実行可能かどうかを示すブール配列
    pub masks: Vec<Array1<bool>>,
    
    // どの局（きょく）でのアクションかを示すインデックス
    pub at_kyoku: Vec<u8>,
    
    // エピソード終了フラグ：局の終了時にtrue
    pub dones: Vec<bool>,
    
    // 割引率（gamma）を適用するかどうか：打牌とカンの場合のみtrue
    pub apply_gamma: Vec<bool>,
    
    // 何巡目のアクションかを示す
    pub at_turns: Vec<u8>,
    
    // シャンテン数：-1=聴牌、0-8=シャンテン数
    pub shantens: Vec<i8>,

    // === ゲーム（実際は局）ごとのデータ ===
    
    // GRP（Game Record Parser）形式のゲームレコード
    pub grp: Grp, // actually per kyoku though（実際は局単位）
    
    // このデータのプレイヤーID（0-3）
    pub player_id: u8,
    
    // プレイヤー名
    pub player_name: String,
}

// ローダーコンテキスト：ゲームログ解析時の一時的な状態を保持する内部構造体
struct LoaderContext<'a> {
    // GameplayLoaderの設定への参照
    config: &'a GameplayLoader,
    
    // 不可視情報（オラクルモード時のみ使用）への参照
    invisibles: Option<&'a [Invisible]>,

    // 現在のプレイヤーの状態
    state: PlayerState,
    
    // 現在処理中の局のインデックス
    kyoku_idx: usize,

    // === 以下のフィールドはオラクルモード時のみ使用 ===
    
    // 対戦相手3人の状態（他家の手牌などを追跡）
    opponent_states: [PlayerState; 3],
    
    // 次のツモが嶺上牌（リンシャンパイ）からかどうか
    from_rinshan: bool,
    
    // 山牌（ヤマ）の現在のインデックス（何枚目をツモるか）
    yama_idx: usize,
    
    // 嶺上牌の現在のインデックス
    rinshan_idx: usize,
}

// GameplayLoaderのPythonインターフェース実装
#[pymethods]
impl GameplayLoader {
    // コンストラクタ：新しいGameplayLoaderインスタンスを作成
    #[new]
    #[pyo3(signature = (
        version,
        *,  // 以降は名前付き引数のみ
        oracle = true,
        player_names = None,
        excludes = None,
        trust_seed = false,
        always_include_kan_select = true,
        augmented = false,
    ))]
    fn new(
        version: u32,                              // 特徴量エンコーディングのバージョン
        oracle: bool,                              // オラクルモード（全情報が見える）
        player_names: Option<Vec<String>>,         // 解析対象プレイヤー名（None=全員）
        excludes: Option<Vec<String>>,             // 除外プレイヤー名
        trust_seed: bool,                          // シード値を信頼して山牌を再現
        always_include_kan_select: bool,           // カン選択を常に含める
        augmented: bool,                           // データ拡張を行う
    ) -> Self {
        // player_namesがNoneの場合は空のVecに変換
        let player_names = player_names.unwrap_or_default();
        // 高速検索用にHashSetに変換
        let player_names_set = player_names.iter().cloned().collect();
        // excludesがNoneの場合は空のVecに変換
        let excludes = excludes.unwrap_or_default();
        // 高速検索用にHashSetに変換
        let excludes_set = excludes.iter().cloned().collect();
        
        // 構造体を構築して返す
        Self {
            version,
            oracle,
            player_names,
            excludes,
            trust_seed,
            always_include_kan_select,
            augmented,
            player_names_set,
            excludes_set,
        }
    }

    // 生のログ文字列からゲームプレイデータを読み込む
    // Nested result is too hard to handle...（ネストしたResultは扱いづらいのでこの形式）
    fn load_log(&self, raw_log: &str) -> Result<Vec<Gameplay>> {
        // ログを行ごとに分割してJSONパース
        let mut events = raw_log
            .lines()                                      // 行ごとに分割
            .map(json::from_str)                         // 各行をJSONとしてパース
            .collect::<Result<Vec<Event>, _>>()          // 結果をVec<Event>に収集
            .context("failed to parse log")?;            // エラー時のコンテキストを追加
            
        // データ拡張が有効な場合、各イベントを拡張
        if self.augmented {
            events.iter_mut().for_each(Event::augment);
        }
        
        // イベントリストからGameplayデータを生成
        self.load_events(&events)
    }

    // gzip圧縮されたログファイルを読み込む（Python向けラッパー）
    #[pyo3(name = "load_gz_log_files")]  // Python側での名前を指定
    fn load_gz_log_files_py(&self, gzip_filenames: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        // 実際の処理は同名のRustメソッドに委譲
        self.load_gz_log_files(gzip_filenames)
    }

    // Python向けの文字列表現（repr()）を実装
    fn __repr__(&self) -> String {
        // Debug trait実装を使って文字列化
        format!("{self:?}")
    }
}

// GameplayLoaderのRust側実装
impl GameplayLoader {
    // 複数のgzip圧縮ログファイルを並列で読み込む
    pub fn load_gz_log_files<V, S>(&self, gzip_filenames: V) -> Result<Vec<Vec<Gameplay>>>
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
                    self.load_log(&raw)                         // ログをパースしてGameplayに変換
                };
                // エラー時にファイル名を含むコンテキストを追加
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()                          // 結果を収集（並列処理の完了を待つ）
    }

    // イベントリストからGameplayデータを生成（各プレイヤー分）
    pub fn load_events(&self, events: &[Event]) -> Result<Vec<Gameplay>> {
        // オラクルモードの場合、不可視情報を生成
        let invisibles = self.oracle.then(|| Invisible::new(events, self.trust_seed));

        // 最初のイベントがStartGameでない場合はエラー
        let [Event::StartGame { names, .. }, ..] = events else {
            bail!("empty or invalid game log");
        };
        
        names
            .iter()                                      // プレイヤー名をイテレート
            .enumerate()                                 // インデックス付きで列挙
            .filter(|&(_, name)| {                      // フィルタリング条件
                // player_namesが指定されている場合
                if !self.player_names_set.is_empty() {
                    // 指定されたプレイヤー名に含まれるかチェック
                    return self.player_names_set.contains(name);
                }
                // excludesが指定されている場合
                if !self.excludes_set.is_empty() {
                    // 除外リストに含まれないかチェック
                    return !self.excludes_set.contains(name);
                }
                // どちらも指定されていない場合は全員を対象
                true
            })
            .map(|(i, _)| i as u8)                      // インデックスをu8型のプレイヤーIDに変換
            .collect::<ArrayVec<[_; 4]>>()              // 最大4人分の固定長配列に収集
            .into_par_iter()                            // 並列イテレータに変換
            .map(|&player_id| {                         // 各プレイヤーIDについて
                // プレイヤー視点のGameplayデータを生成
                Gameplay::load_events_by_player(self, events, player_id, invisibles.as_deref())
            })
            .collect()                                  // 結果を収集
    }
}

// GameplayのPythonインターフェース実装
// 注意：すべてのtake_メソッドは所有権を移動（ムーブ）してフィールドを空にする
#[pymethods]
impl Gameplay {
    // 観測値（特徴量）を取り出してPythonのNumPy配列として返す
    fn take_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.obs)                        // フィールドの所有権を取得（self.obsは空になる）
            .into_iter()                                 // Vecをイテレータに変換
            .map(|v| PyArray2::from_owned_array(py, v)) // 各Array2をPyArray2に変換
            .collect()                                   // 結果をVecに収集
    }
    
    // 不可視観測値を取り出してPythonのNumPy配列として返す
    fn take_invisible_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.invisible_obs)              // フィールドの所有権を取得
            .into_iter()                                 // Vecをイテレータに変換
            .map(|v| PyArray2::from_owned_array(py, v)) // 各Array2をPyArray2に変換
            .collect()                                   // 結果をVecに収集
    }
    
    // アクションIDのリストを取り出して返す
    fn take_actions(&mut self) -> Vec<i64> {
        mem::take(&mut self.actions)                    // フィールドの所有権を取得して返す
    }
    
    // アクションマスクを取り出してPythonのNumPy配列として返す
    fn take_masks<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<bool>>> {
        mem::take(&mut self.masks)                      // フィールドの所有権を取得
            .into_iter()                                 // Vecをイテレータに変換
            .map(|v| PyArray1::from_owned_array(py, v)) // 各Array1をPyArray1に変換
            .collect()                                   // 結果をVecに収集
    }
    
    // 局インデックスのリストを取り出して返す
    fn take_at_kyoku(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_kyoku)                   // フィールドの所有権を取得して返す
    }
    
    // エピソード終了フラグのリストを取り出して返す
    fn take_dones(&mut self) -> Vec<bool> {
        mem::take(&mut self.dones)                      // フィールドの所有権を取得して返す
    }
    
    // 割引率適用フラグのリストを取り出して返す
    fn take_apply_gamma(&mut self) -> Vec<bool> {
        mem::take(&mut self.apply_gamma)                // フィールドの所有権を取得して返す
    }
    
    // ターン数のリストを取り出して返す
    fn take_at_turns(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_turns)                   // フィールドの所有権を取得して返す
    }
    
    // シャンテン数のリストを取り出して返す
    fn take_shantens(&mut self) -> Vec<i8> {
        mem::take(&mut self.shantens)                   // フィールドの所有権を取得して返す
    }

    // GRPデータを取り出して返す
    fn take_grp(&mut self) -> Grp {
        mem::take(&mut self.grp)                        // フィールドの所有権を取得して返す
    }

    // プレイヤーIDを返す（所有権の移動なし、constメソッド）
    const fn take_player_id(&self) -> u8 {
        self.player_id                                  // プレイヤーIDをコピーして返す
    }
}

// GameplayのRust側実装
impl Gameplay {
    // 特定のプレイヤー視点でイベントを解析してGameplayデータを生成
    fn load_events_by_player(
        config: &GameplayLoader,                        // ローダー設定
        events: &[Event],                               // ゲームイベントのリスト
        player_id: u8,                                  // 対象プレイヤーID（0-3）
        invisibles: Option<&[Invisible]>,               // オラクルモード時の不可視情報
    ) -> Result<Self> {
        // GRP形式でイベントを読み込む
        let grp = Grp::load_events(events)?;

        // Gameplayデータの初期化
        let mut data = Self {
            grp,
            player_id,
            ..Default::default()                        // 残りのフィールドはデフォルト値
        };

        // ローダーコンテキストの初期化
        let mut ctx = LoaderContext {
            config,
            invisibles,
            state: PlayerState::new(player_id),                               // プレイヤー状態を初期化
            kyoku_idx: 0,                                                     // 局インデックスを0から開始
            // end_state: EndState::Passive,                                 // （コメントアウトされた終了状態）
            opponent_states: array::from_fn(|i| {                            // 対戦相手3人の状態を初期化
                PlayerState::new((player_id + i as u8 + 1) % 4)              // player_idの次から3人分を生成
            }),
            from_rinshan: false,                                              // 嶺上牌からのツモフラグ
            yama_idx: 0,                                                      // 山牌インデックス
            rinshan_idx: 0,                                                   // 嶺上牌インデックス
        };

        // It is guaranteed that there are at least 4 events.
        // 最低4つのイベントが存在することが保証されている
        // tsumo/dahai -> ryukyoku/hora -> end kyoku -> end game
        // ツモ/打牌 -> 流局/和了 -> 局終了 -> ゲーム終了
        for wnd in events.windows(4) {                                        // 4イベントの窓を移動させながら処理
            data.extend_from_event_window(&mut ctx, wnd.try_into().unwrap())?;
        }

        // donesフラグの生成：局が変わったところでtrue
        data.dones = data.at_kyoku
            .windows(2)                                 // 隣接する2要素のペアを見る
            .map(|w| w[1] > w[0])                      // 局番号が増えたらtrue
            .collect();
        data.dones.push(true);                         // 最後のエピソードは必ずtrue

        Ok(data)
    }

    // 4イベントのウィンドウから学習データを生成
    fn extend_from_event_window(
        &mut self,
        ctx: &mut LoaderContext<'_>,                   // ローダーコンテキスト
        wnd: &[Event; 4],                               // 4イベントのウィンドウ
    ) -> Result<()> {
        // コンテキストの各フィールドを分解して取得（可変参照）
        let LoaderContext {
            config,
            invisibles,
            state,
            kyoku_idx,
            opponent_states,
            from_rinshan,
            yama_idx,
            rinshan_idx,
        } = ctx;

        // 現在のイベントは窓の最初
        let cur = &wnd[0];
        // 次のイベントの決定（ReachAcceptedやDoraはスキップ）
        let next = if matches!(wnd[1], Event::ReachAccepted { .. } | Event::Dora { .. }) {
            &wnd[2]  // これらのイベントは行動決定に影響しないのでスキップ
        } else {
            &wnd[1]  // 通常は次のイベント
        };

        // 現在のイベントに基づいた処理
        match cur {
            Event::StartGame { names, .. } => {
                // ゲーム開始時：プレイヤー名を保存
                self.player_name.clone_from(&names[self.player_id as usize]);
            }
            Event::EndKyoku => *kyoku_idx += 1,         // 局終了時：局インデックスを増やす
            _ => (),                                    // その他のイベントは特に処理なし
        }

        // オラクルモード時の追加処理（不可視情報がある場合）
        if invisibles.is_some() {
            match cur {
                Event::EndKyoku => {
                    // 局終了時：山牌関連の状態をリセット
                    *from_rinshan = false;              // 嶺上牌フラグをリセット
                    *yama_idx = 0;                      // 山牌インデックスをリセット
                    *rinshan_idx = 0;                   // 嶺上牌インデックスをリセット
                }
                Event::Tsumo { .. } => {
                    // ツモ時：山牌または嶺上牌のインデックスを更新
                    if *from_rinshan {
                        // 嶺上牌からのツモ（カン後）
                        *rinshan_idx += 1;              // 嶺上牌インデックスを進める
                        *from_rinshan = false;          // フラグをリセット
                    } else {
                        // 通常の山牌からのツモ
                        *yama_idx += 1;                 // 山牌インデックスを進める
                    }
                }
                // カン行為の後は嶺上牌からツモる
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                    *from_rinshan = true;               // 次のツモは嶺上牌から
                }
                _ => (),
            };

            // 対戦相手の状態も更新（オラクル情報の追跡）
            for s in opponent_states {
                s.update(cur)?;
            }
        }

        // プレイヤー状態を更新し、可能な行動を取得
        let cans = state.update(cur)?;
        // 行動できない場合は処理をスキップ
        if !cans.can_act() {
            return Ok(());
        }

        // カン選択時の牌を記録するための変数
        let mut kan_select = None;
        
        // 次のイベントから実際に実行されたアクションのラベルを決定
        let label_opt = match *next {
            // 打牌：牌のインデックス（0-36）をラベルとする
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            
            // リーチ宣言：37番
            Event::Reach { .. } => Some(37),
            
            // チー：38-40番（チーの種類による）
            Event::Chi {
                actor,
                pai,
                consumed,
                ..
            } if actor == self.player_id => match ChiType::new(consumed, pai) {
                ChiType::Low => Some(38),       // 下チー（例：123の1でチー）
                ChiType::Mid => Some(39),       // 中チー（例：234の3でチー）
                ChiType::High => Some(40),      // 上チー（例：345の5でチー）
            },
            
            // ポン：41番（自分がポンした場合のみ）
            Event::Pon { actor, .. } if actor == self.player_id => Some(41),
            
            // 大明槓：42番（自分が大明槓した場合のみ）
            Event::Daiminkan { actor, pai, .. } if actor == self.player_id => {
                // 設定により、カン選択も記録する場合
                if config.always_include_kan_select {
                    kan_select = Some(pai.deaka().as_usize()); // 赤牌を通常牌として扱う
                }
                Some(42)
            }
            
            // 加槓：42番
            Event::Kakan { pai, .. } => {
                // 常に記録するか、複数の選択肢がある場合のみ記録
                if config.always_include_kan_select || state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(42)
            }
            
            // 暗槓：42番
            Event::Ankan { consumed, .. } => {
                // 常に記録するか、複数の選択肢がある場合のみ記録
                if config.always_include_kan_select || state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].deaka().as_usize());
                }
                Some(42)
            }
            
            // 九種九牌流局：44番（流局可能な場合のみ）
            Event::Ryukyoku { .. } if cans.can_ryukyoku => Some(44),
            
            // その他の場合（ロンまたはパス）
            _ => {
                let mut ret = None;

                // 誰かがロンしたかチェック
                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    // Check if the POV is one of those who made Hora.
                    // 自分がロンした場合をチェック
                    for ev in &wnd[1..] {
                        match *ev {
                            Event::EndKyoku => break,   // 局終了まで到達
                            Event::Hora { actor, .. } if actor == self.player_id => {
                                ret = Some(43);         // ロン：43番
                                break;
                            }
                            _ => (),
                        };
                    }
                }

                // 自分がロンしなかった場合
                if ret.is_none() {
                    // It is now proven there is no ron from the POV.
                    // 自分がロンしなかったことが確定
                    
                    // チー可能だったが見送った、または
                    // ポン/大明槓/ロン可能だったが見送った場合
                    if cans.can_chi() && matches!(next, Event::Tsumo { .. })
                        || (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari)
                            && !has_any_ron
                    {
                        // Can chi, but actively denied instead of being
                        // interrupted by other's pon/daiminkan/ron.
                        // チー可能だったが、他者の鳴きやロンに邪魔されたわけではなく
                        // 自発的に見送った
                        //
                        // or
                        // または
                        //
                        // Can pon/daiminkan/ron, but actively denied
                        // instead of being interrupted by other's ron.
                        // ポン/大明槓/ロン可能だったが、他者のロンに邪魔されたわけではなく
                        // 自発的に見送った
                        ret = Some(45);                 // パス：45番
                    }
                }

                ret
            }
        };

        // ラベルが決定した場合、エントリーを追加
        if let Some(label) = label_opt {
            self.add_entry(ctx, false, label);         // 通常のアクションを記録
            if let Some(kan) = kan_select {
                self.add_entry(ctx, true, kan);         // カン選択も記録（必要な場合）
            }
        }
        Ok(())
    }

    // 学習データのエントリーを追加
    fn add_entry(&mut self, ctx: &LoaderContext<'_>, at_kan_select: bool, label: usize) {
        // 現在の状態から特徴量とアクションマスクをエンコード
        let (feature, mask) = ctx.state.encode_obs(ctx.config.version, at_kan_select);
        
        // 観測値（特徴量）を追加
        self.obs.push(feature);
        
        // 実行したアクションのラベルを追加
        self.actions.push(label as i64);
        
        // アクションマスクを追加
        self.masks.push(mask);
        
        // 現在の局番号を追加
        self.at_kyoku.push(ctx.kyoku_idx as u8);
        
        // only discard and kan will discount
        // 割引率を適用するかどうか（打牌とカンの場合のみtrue）
        // label <= 37は打牌（0-36）とリーチ（37）を含むが、
        // リーチは打牌と同時に行われるため、実質的に打牌のみ
        self.apply_gamma.push(label <= 37);
        
        // 現在のターン数を追加
        self.at_turns.push(ctx.state.at_turn());
        
        // 現在のシャンテン数を追加
        self.shantens.push(ctx.state.shanten());

        // オラクルモード時：不可視情報もエンコードして追加
        if let Some(invisibles) = ctx.invisibles {
            // 対戦相手の状態、山牌、嶺上牌の情報をエンコード
            let invisible_obs = invisibles[ctx.kyoku_idx].encode(
                &ctx.opponent_states,                   // 対戦相手3人の状態
                ctx.yama_idx,                           // 山牌の現在位置
                ctx.rinshan_idx,                        // 嶺上牌の現在位置
                ctx.config.version,                     // エンコーディングバージョン
            );
            self.invisible_obs.push(invisible_obs);
        }
    }
}
