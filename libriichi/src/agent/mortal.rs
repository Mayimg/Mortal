//! Mortal麻雀AIのバッチエージェント実装
//! 
//! このモジュールは、Pythonで実装されたニューラルネットワークモデルとRustのゲームエンジンを
//! 橋渡しする役割を持つ。複数のゲーム状態を同時に処理（バッチ処理）することで、GPUの
//! 並列計算能力を最大限に活用し、高速な推論を実現する。
//! 
//! 主な機能：
//! - バッチ処理による効率的な推論
//! - 並列処理による特徴量エンコーディングの高速化
//! - クイック評価による単純な捨て牌の最適化
//! - ルールベース和了ガードによる安全性の向上
//! - 詳細なメタデータ記録によるデバッグ・分析支援

use super::{BatchAgent, InvisibleState};  // BatchAgentトレイトと隠れ状態の型をインポート
use crate::consts::ACTION_SPACE;           // アクション空間のサイズ（46）
use crate::mjai::{Event, EventExt, Metadata};  // mjaiプロトコルのイベントとメタデータ
use crate::state::PlayerState;             // プレイヤーの状態を管理する構造体
use crate::{must_tile, tu8};               // 牌のユーティリティマクロ
use std::mem;                              // メモリ操作用（take関数など）
use std::sync::Arc;                        // スレッド間共有のための参照カウント型
use std::time::{Duration, Instant};        // 実行時間計測用

use anyhow::{Context, Result, ensure};     // エラーハンドリング用
use crossbeam::sync::WaitGroup;            // 並列処理の同期用
use ndarray::prelude::*;                   // 多次元配列操作
use numpy::{PyArray1, PyArray2};           // Python NumPy配列との相互変換
use parking_lot::Mutex;                    // 高速なMutex実装
use pyo3::intern;                          // Python文字列のインターン化（高速化）
use pyo3::prelude::*;                      // Python統合用

/// Mortal AIのバッチ処理エージェント
/// 複数のゲーム状態を同時に処理し、効率的な推論を行う
pub struct MortalBatchAgent {
    /// Pythonで実装されたニューラルネットワークエンジン
    engine: PyObject,
    /// オラクルモード（完全情報）で動作するかどうか
    is_oracle: bool,
    /// 観測エンコーディングのバージョン（特徴量の形式を決定）
    version: u32,
    /// クイック評価機能の有効化フラグ（単純な捨て牌を高速化）
    enable_quick_eval: bool,
    /// ルールベース和了ガードの有効化フラグ（危険な和了を防ぐ）
    enable_rule_based_agari_guard: bool,
    /// エージェントの名前（ログやデバッグ用）
    name: String,
    /// 各バッチインデックスに対応するプレイヤーID（0-3）
    player_ids: Vec<u8>,

    /// 最後の評価で選択されたアクションのインデックス（0-45）
    actions: Vec<usize>,
    /// 各アクションのQ値（期待される将来の報酬）
    q_values: Vec<[f32; ACTION_SPACE]>,
    /// 各アクションが合法かどうかを示すマスク
    masks_recv: Vec<[bool; ACTION_SPACE]>,
    /// 貪欲法（最大Q値）で選択されたかどうか
    is_greedy: Vec<bool>,
    /// 最後の評価にかかった時間
    last_eval_elapsed: Duration,
    /// 最後のバッチサイズ
    last_batch_size: usize,

    /// 現在のバッチが評価済みかどうか
    evaluated: bool,
    /// クイック評価で決定されたリアクション（キャッシュ）
    quick_eval_reactions: Vec<Option<Event>>,

    /// 並列処理の同期用WaitGroup
    wg: WaitGroup,
    /// スレッド間で共有される同期フィールド
    sync_fields: Arc<Mutex<SyncFields>>,
}

/// スレッド間で共有される同期用フィールド
/// 並列処理された特徴量エンコーディングの結果を格納
struct SyncFields {
    /// ゲーム状態の特徴量（ニューラルネットワークへの入力）
    states: Vec<Array2<f32>>,
    /// 隠れ状態の特徴量（オラクルモード時のみ使用）
    invisible_states: Vec<Array2<f32>>,
    /// 合法手のマスク（実行可能なアクションを示す）
    masks: Vec<Array1<bool>>,
    /// 各ゲームインデックスに対応する通常アクションのインデックス
    action_idxs: Vec<usize>,
    /// カン選択時の追加アクションのインデックス（カンが複数候補ある場合）
    kan_action_idxs: Vec<Option<usize>>,
}

impl MortalBatchAgent {
    /// 新しいMortalBatchAgentを作成
    /// 
    /// # Arguments
    /// * `engine` - Pythonのニューラルネットワークエンジン
    /// * `player_ids` - 各バッチインデックスに対応するプレイヤーID（0-3）
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        // すべてのプレイヤーIDが有効な範囲（0-3）内にあることを確認
        ensure!(player_ids.iter().all(|&id| matches!(id, 0..=3)));

        // Pythonエンジンから必要な属性を取得
        let (name, is_oracle, version, enable_quick_eval, enable_rule_based_agari_guard) =
            Python::with_gil(|py| {  // GIL（Global Interpreter Lock）を取得してPython操作
                let obj = engine.bind_borrowed(py);
                // react_batchメソッドが存在し、呼び出し可能であることを確認
                ensure!(
                    obj.getattr("react_batch")?.is_callable(),
                    "missing method react_batch",
                );

                // エンジンの各種設定を取得
                let name = obj.getattr("name")?.extract()?;  // モデル名
                let is_oracle = obj.getattr("is_oracle")?.extract()?;  // オラクルモード
                let version = obj.getattr("version")?.extract()?;  // 観測バージョン
                let enable_quick_eval = obj.getattr("enable_quick_eval")?.extract()?;  // クイック評価
                let enable_rule_based_agari_guard =  // ルールベース和了ガード
                    obj.getattr("enable_rule_based_agari_guard")?.extract()?;
                Ok((
                    name,
                    is_oracle,
                    version,
                    enable_quick_eval,
                    enable_rule_based_agari_guard,
                ))
            })?;

        let size = player_ids.len();  // バッチサイズ
        // クイック評価が有効な場合のみ、リアクションキャッシュを初期化
        let quick_eval_reactions = if enable_quick_eval {
            vec![None; size]
        } else {
            vec![]
        };
        // スレッド間共有用の同期フィールドを初期化
        let sync_fields = Arc::new(Mutex::new(SyncFields {
            states: vec![],                     // 空の状態ベクター
            invisible_states: vec![],           // 空の隠れ状態ベクター
            masks: vec![],                      // 空のマスクベクター
            action_idxs: vec![0; size],         // 各インデックスのアクション位置
            kan_action_idxs: vec![None; size],  // カン選択用のインデックス（初期値なし）
        }));

        // MortalBatchAgentインスタンスを作成して返す
        Ok(Self {
            engine,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            name,
            player_ids: player_ids.to_vec(),

            actions: vec![],                   // 空の初期状態
            q_values: vec![],                  // 空の初期状態
            masks_recv: vec![],                // 空の初期状態
            is_greedy: vec![],                 // 空の初期状態
            last_eval_elapsed: Duration::ZERO, // 評価時間の初期値
            last_batch_size: 0,                // バッチサイズの初期値

            evaluated: false,                  // 未評価状態
            quick_eval_reactions,              // クイック評価のキャッシュ

            wg: WaitGroup::new(),              // 新しいWaitGroup
            sync_fields,                       // 共有同期フィールド
        })
    }

    /// バッチ化された状態をニューラルネットワークで評価
    /// すべての特徴量エンコーディングが完了するのを待ってから、
    /// Pythonエンジンに送信して推論を実行
    fn evaluate(&mut self) -> Result<()> {
        // すべての特徴量エンコーディングが完了するまで待機
        // mem::takeでWaitGroupの所有権を移動し、新しいWaitGroupに置き換える
        mem::take(&mut self.wg).wait();
        // 同期フィールドのロックを取得
        let mut sync_fields = self.sync_fields.lock();

        // 評価する状態がない場合は早期リターン
        if sync_fields.states.is_empty() {
            return Ok(());
        }

        let start = Instant::now();  // 評価時間の計測開始
        self.last_batch_size = sync_fields.states.len();  // バッチサイズを記録

        // PythonのGILを取得してニューラルネットワークを実行
        (self.actions, self.q_values, self.masks_recv, self.is_greedy) = Python::with_gil(|py| {
            // ゲーム状態をPyArrayに変換（所有権を移動してコピーを避ける）
            let states: Vec<_> = sync_fields
                .states
                .drain(..)  // すべての要素を取り出し、ベクターを空にする
                .map(|v| PyArray2::from_owned_array(py, v))  // Rust配列をPython配列に変換
                .collect();
            // 合法手マスクをPyArrayに変換
            let masks: Vec<_> = sync_fields
                .masks
                .drain(..)
                .map(|v| PyArray1::from_owned_array(py, v))
                .collect();
            // オラクルモードの場合のみ、隠れ状態を変換
            let invisible_states: Option<Vec<_>> = self.is_oracle.then(|| {
                sync_fields
                    .invisible_states
                    .drain(..)
                    .map(|v| PyArray2::from_owned_array(py, v))
                    .collect()
            });

            // Pythonメソッドの引数を準備
            let args = (states, masks, invisible_states);
            // Pythonエンジンのreact_batchメソッドを呼び出し
            self.engine
                .bind_borrowed(py)
                .call_method1(intern!(py, "react_batch"), args)  // intern!で文字列を最適化
                .context("failed to execute `react_batch` on Python engine")?
                .extract()  // Python結果をRust型に変換
                .context("failed to extract to Rust type")
        })?;

        // 評価にかかった時間を記録（オーバーフロー対策付き）
        self.last_eval_elapsed = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO);

        Ok(())
    }

    /// アクションのメタデータを生成
    /// デバッグや分析のために、Q値、マスク、ゲーム状態などの情報を記録
    fn gen_meta(&self, state: &PlayerState, action_idx: usize) -> Metadata {
        let q_values = self.q_values[action_idx];    // 全アクションのQ値
        let masks = self.masks_recv[action_idx];     // 合法手マスク
        let is_greedy = self.is_greedy[action_idx];  // 貪欲選択かどうか

        // マスクビットとコンパクトなQ値配列を生成
        let mut mask_bits = 0;  // 合法手をビット表現で記録
        let q_values_compact = q_values
            .into_iter()
            .zip(masks)
            .enumerate()
            .filter(|&(_, (_, m))| m)  // 合法手のみをフィルタ
            .map(|(i, (q, _))| {
                mask_bits |= 0b1 << i;  // i番目のビットを立てる
                q  // Q値を返す
            })
            .collect();  // 合法手のQ値のみを収集

        // メタデータ構造体を構築
        Metadata {
            q_values: Some(q_values_compact),      // 合法手のQ値のみ
            mask_bits: Some(mask_bits),            // 合法手のビットマスク
            is_greedy: Some(is_greedy),            // 貪欲選択フラグ
            shanten: Some(state.shanten()),        // 現在のシャンテン数
            at_furiten: Some(state.at_furiten()),  // フリテン状態かどうか
            ..Default::default()                   // その他のフィールドはデフォルト値
        }
    }
}

// BatchAgentトレイトの実装
impl BatchAgent for MortalBatchAgent {
    /// エージェントの名前を返す（インライン最適化）
    #[inline]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// オラクルモードの場合のみ観測バージョンを返す（インライン最適化）
    #[inline]
    fn oracle_obs_version(&self) -> Option<u32> {
        self.is_oracle.then_some(self.version)  // is_oracleがtrueならSome(version)、falseならNone
    }

    /// ゲームシーンをセットアップし、必要に応じて特徴量エンコーディングを開始
    /// 
    /// # Arguments
    /// * `index` - バッチ内のインデックス
    /// * `_` - イベント履歴（未使用）
    /// * `state` - 現在のプレイヤー状態
    /// * `invisible_state` - 隠れ状態（オラクルモード時のみ）
    fn set_scene(
        &mut self,
        index: usize,
        _: &[EventExt],  // イベント履歴は使用しない
        state: &PlayerState,
        invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        self.evaluated = false;  // 評価フラグをリセット
        let cans = state.last_cans();  // 実行可能なアクションを取得

        // クイック評価：単純な捨て牌のみ可能な場合の最適化
        if self.enable_quick_eval  // クイック評価が有効
            && cans.can_discard     // 捨て牌可能
            && !cans.can_riichi     // リーチ不可
            && !cans.can_tsumo_agari // ツモ和了不可
            && !cans.can_ankan      // 暗槓不可
            && !cans.can_kakan      // 加槓不可
            && !cans.can_ryukyoku   // 流局宣言不可
        {
            // 捨て牌候補を取得（赤牌考慮）
            let candidates = state.discard_candidates_aka();
            let mut only_candidate = None;
            // 唯一の候補を探す
            for (tile, &flag) in candidates.iter().enumerate() {
                if !flag {  // この牌が候補でない場合はスキップ
                    continue;
                }
                match only_candidate.take() {
                    None => only_candidate = Some(tile),  // 最初の候補を記録
                    Some(_) => break,  // 2つ目の候補が見つかったら終了
                }
            }

            // 唯一の候補がある場合、即座にリアクションを決定
            if let Some(tile_id) = only_candidate {
                let actor = self.player_ids[index];
                let pai = must_tile!(tile_id);  // tile_idをTile型に変換
                // ツモ切りかどうかを判定（最後にツモった牌と同じか）
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                let ev = Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                };
                self.quick_eval_reactions[index] = Some(ev);  // キャッシュに保存
                return Ok(());  // 特徴量エンコーディングをスキップ
            }
        }

        // カン選択が必要かどうかを判定
        let need_kan_select = if !cans.can_ankan && !cans.can_kakan {
            false  // カン自体ができない場合
        } else if !self.enable_quick_eval {
            true   // クイック評価が無効な場合は常に必要
        } else {
            // 複数のカン候補がある場合のみ選択が必要
            state.ankan_candidates().len() + state.kakan_candidates().len() > 1
        };

        // 並列処理用に必要な値をクローン
        let version = self.version;
        let state = state.clone();  // PlayerStateをクローン（並列処理で使用）
        let sync_fields = Arc::clone(&self.sync_fields);  // 共有フィールドの参照をクローン
        let wg = self.wg.clone();  // WaitGroupをクローン
        // 新しいスレッドで特徴量エンコーディングを実行
        rayon::spawn(move || {
            let _wg = wg;  // スコープを抜ける時に自動的にwaitが減る

            // 並列処理で特徴量をエンコード
            // これは特にv4以降のsp（シャンテン進行）特徴量で計算量が多いため重要
            // カン選択が必要な場合は、カン用の特徴量も生成
            let kan = need_kan_select.then(|| state.encode_obs(version, true));
            // 通常のアクション用の特徴量を生成
            let (feature, mask) = state.encode_obs(version, false);

            // 同期フィールドのロックを取得して更新
            let SyncFields {
                states,
                invisible_states,
                masks,
                action_idxs,
                kan_action_idxs,
            } = &mut *sync_fields.lock();
            
            // カン選択用の特徴量がある場合は先に追加
            if let Some((kan_feature, kan_mask)) = kan {
                kan_action_idxs[index] = Some(states.len());  // 現在の位置を記録
                states.push(kan_feature);      // カン用の特徴量を追加
                masks.push(kan_mask);          // カン用のマスクを追加
                if let Some(invisible_state) = invisible_state.clone() {
                    invisible_states.push(invisible_state);  // オラクル用の隠れ状態
                }
            }

            // 通常のアクション用の特徴量を追加
            action_idxs[index] = states.len();  // 現在の位置を記録
            states.push(feature);               // 特徴量を追加
            masks.push(mask);                   // マスクを追加
            if let Some(invisible_state) = invisible_state {
                invisible_states.push(invisible_state);  // オラクル用の隠れ状態
            }
        });

        Ok(())
    }

    /// プレイヤーのリアクション（アクション）を取得
    /// 
    /// # Arguments
    /// * `index` - バッチ内のインデックス
    /// * `_` - イベント履歴（未使用）
    /// * `state` - 現在のプレイヤー状態
    /// * `_` - 隠れ状態（未使用）
    fn get_reaction(
        &mut self,
        index: usize,
        _: &[EventExt],  // イベント履歴は使用しない
        state: &PlayerState,
        _: Option<InvisibleState>,  // 隠れ状態は使用しない
    ) -> Result<EventExt> {
        // クイック評価でキャッシュされたリアクションがあればそれを返す
        if self.enable_quick_eval {
            if let Some(ev) = self.quick_eval_reactions[index].take() {  // takeで所有権を移動
                return Ok(EventExt::no_meta(ev));  // メタデータなしで返す
            }
        }

        // まだ評価していない場合は、バッチ評価を実行
        if !self.evaluated {
            self.evaluate()?;  // ニューラルネットワークで推論
            self.evaluated = true;  // 評価済みフラグを立てる
        }
        let start = Instant::now();  // リアクション生成の時間計測開始

        // 同期フィールドから必要なインデックスを取得
        let mut sync_fields = self.sync_fields.lock();
        let action_idx = sync_fields.action_idxs[index];  // このゲームのアクション位置
        let kan_select_idx = sync_fields.kan_action_idxs[index].take();  // カン選択のインデックス（あれば）

        let actor = self.player_ids[index];         // プレイヤーID
        let akas_in_hand = state.akas_in_hand();    // 手牌の赤牌情報
        let cans = state.last_cans();               // 実行可能なアクション

        let orig_action = self.actions[action_idx]; // ニューラルネットワークが選択したアクション
        // ルールベース和了ガードのチェック
        let action =
            if self.enable_rule_based_agari_guard && orig_action == 43 && !state.rule_based_agari()
            {
                // エンジンは和了を選択したが、ルールベースエンジンが危険と判断
                // この場合、和了以外の最良の選択肢を強制的に実行
                let mut q_values = self.q_values[action_idx];  // Q値をコピー
                q_values[43] = f32::MIN;  // 和了のQ値を最小値に設定
                q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, l), (_, r)| l.total_cmp(r))  // 最大Q値を持つアクションを探す
                    .unwrap()
                    .0  // インデックスを取得
            } else {
                orig_action  // 通常はオリジナルのアクションを使用
            };

        // アクションインデックスを実際のmjaiイベントに変換
        let event = match action {
            // 0-36: 捨て牌アクション
            0..=36 => {
                // 捨て牌が可能であることを確認
                ensure!(
                    cans.can_discard,
                    "failed discard check: {}",
                    state.brief_info()
                );

                let pai = must_tile!(action);  // アクションインデックスを牌に変換
                // ツモ切りかどうかを判定
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                }
            }

            // 37: リーチ宣言
            37 => {
                // リーチが可能であることを確認
                ensure!(
                    cans.can_riichi,
                    "failed riichi check: {}",
                    state.brief_info()
                );

                Event::Reach { actor }
            }

            // 38: チー（下位順子）- 例：3で[1,2]を鳴く
            38 => {
                // チー（下位）が可能であることを確認
                ensure!(
                    cans.can_chi_low,
                    "failed chi low check: {}",
                    state.brief_info()
                );

                // 鳴く牌を取得
                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let first = pai.next();  // 次の牌（例：3なら4）

                // 消費する牌に赤牌を使えるかチェック
                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(3m) | tu8!(4m) => akas_in_hand[0],  // 萬子の赤5
                    tu8!(3p) | tu8!(4p) => akas_in_hand[1],  // 筒子の赤5
                    tu8!(3s) | tu8!(4s) => akas_in_hand[2],  // 索子の赤5
                    _ => false,
                };
                // 消費する牌（赤牌優先）
                let consumed = if can_akaize_consumed {
                    [first.akaize(), first.next().akaize()]  // 赤牌を使用
                } else {
                    [first, first.next()]  // 通常牌を使用
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,  // 鳴く相手
                    pai,                        // 鳴く牌
                    consumed,                   // 消費する牌
                }
            }
            // 39: チー（中位順子）- 例：5で[4,6]を鳴く
            39 => {
                // チー（中位）が可能であることを確認
                ensure!(
                    cans.can_chi_mid,
                    "failed chi mid check: {}",
                    state.brief_info()
                );

                // 鳴く牌を取得
                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                // 消費する牌に赤牌を使えるかチェック
                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(4m) | tu8!(6m) => akas_in_hand[0],  // 萬子の赤5
                    tu8!(4p) | tu8!(6p) => akas_in_hand[1],  // 筒子の赤5
                    tu8!(4s) | tu8!(6s) => akas_in_hand[2],  // 索子の赤5
                    _ => false,
                };
                // 消費する牌（赤牌優先）
                let consumed = if can_akaize_consumed {
                    [pai.prev().akaize(), pai.next().akaize()]  // 前後の牌を赤牌化
                } else {
                    [pai.prev(), pai.next()]  // 通常の前後の牌
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            // 40: チー（上位順子）- 例：7で[5,6]を鳴く
            40 => {
                // チー（上位）が可能であることを確認
                ensure!(
                    cans.can_chi_high,
                    "failed chi high check: {}",
                    state.brief_info()
                );

                // 鳴く牌を取得
                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let last = pai.prev();  // 前の牌（例：7なら6）

                // 消費する牌に赤牌を使えるかチェック
                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(6m) | tu8!(7m) => akas_in_hand[0],  // 萬子の赤5
                    tu8!(6p) | tu8!(7p) => akas_in_hand[1],  // 筒子の赤5
                    tu8!(6s) | tu8!(7s) => akas_in_hand[2],  // 索子の赤5
                    _ => false,
                };
                // 消費する牌（赤牌優先）
                let consumed = if can_akaize_consumed {
                    [last.prev().akaize(), last.akaize()]  // 5,6を赤牌化
                } else {
                    [last.prev(), last]  // 通常の5,6
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            // 41: ポン（刻子）
            41 => {
                // ポンが可能であることを確認
                ensure!(cans.can_pon, "failed pon check: {}", state.brief_info());

                // 鳴く牌を取得
                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                // 消費する牌に赤牌を使えるかチェック（5のポンの場合）
                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(5m) => akas_in_hand[0],  // 萬子の赤5
                    tu8!(5p) => akas_in_hand[1],  // 筒子の赤5
                    tu8!(5s) => akas_in_hand[2],  // 索子の赤5
                    _ => false,
                };
                // 消費する牌（赤牌優先）
                let consumed = if can_akaize_consumed {
                    [pai.akaize(), pai.deaka()]  // 赤5と通常5
                } else {
                    [pai.deaka(); 2]  // 通常牌2枚
                };
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            // 42: カン（槓子）- 大明槓、暗槓、加槓のいずれか
            42 => {
                // いずれかのカンが可能であることを確認
                ensure!(
                    cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                    "failed kan check: {}",
                    state.brief_info()
                );

                // カン候補を取得
                let ankan_candidates = state.ankan_candidates();  // 暗槓候補
                let kakan_candidates = state.kakan_candidates();  // 加槓候補

                // カンする牌を決定
                let tile = if let Some(kan_idx) = kan_select_idx {
                    // カン選択がある場合（複数候補から選択）
                    let tile = must_tile!(self.actions[kan_idx]);
                    // 選択した牌が候補に含まれることを確認
                    ensure!(
                        ankan_candidates.contains(&tile) || kakan_candidates.contains(&tile),
                        "kan choice not in kan candidates: {}",
                        state.brief_info()
                    );
                    tile
                } else if cans.can_daiminkan {
                    // 大明槓の場合は捨て牌をカン
                    state
                        .last_kawa_tile()
                        .context("invalid state: no last kawa tile")?
                } else if cans.can_ankan {
                    // 暗槓の場合は最初の候補
                    ankan_candidates[0]
                } else {
                    // 加槓の場合は最初の候補
                    kakan_candidates[0]
                };

                // カンの種類に応じてイベントを生成
                if cans.can_daiminkan {
                    // 大明槓
                    let consumed = if tile.is_aka() {
                        [tile.deaka(); 3]  // 赤牌を鳴いた場合、手牌の通常牌3枚
                    } else {
                        [tile.akaize(), tile, tile]  // 通常牌を鳴いた場合、赤牌優先
                    };
                    Event::Daiminkan {
                        actor,
                        target: cans.target_actor,
                        pai: tile,
                        consumed,
                    }
                } else if cans.can_ankan && ankan_candidates.contains(&tile.deaka()) {
                    // 暗槓
                    Event::Ankan {
                        actor,
                        consumed: [tile.akaize(), tile, tile, tile],  // 赤牌優先で4枚
                    }
                } else {
                    // 加槓
                    let can_akaize_target = match tile.as_u8() {
                        tu8!(5m) => akas_in_hand[0],
                        tu8!(5p) => akas_in_hand[1],
                        tu8!(5s) => akas_in_hand[2],
                        _ => false,
                    };
                    // 加槓する牌と消費する牌を決定
                    let (pai, consumed) = if can_akaize_target {
                        (tile.akaize(), [tile.deaka(); 3])  // 赤5を加槓
                    } else {
                        (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])  // 通常牌を加槓
                    };
                    Event::Kakan {
                        actor,
                        pai,
                        consumed,
                    }
                }
            }

            // 43: 和了（アガリ）
            43 => {
                // 和了が可能であることを確認
                ensure!(
                    cans.can_agari(),
                    "failed hora check: {}",
                    state.brief_info(),
                );

                Event::Hora {
                    actor,
                    target: cans.target_actor,  // ロンの相手（ツモならNone）
                    deltas: None,               // 点数変動（後で計算）
                    ura_markers: None,          // 裏ドラ（後で公開）
                }
            }

            // 44: 流局（九種九牌など）
            44 => {
                // 流局宣言が可能であることを確認
                ensure!(
                    cans.can_ryukyoku,
                    "failed ryukyoku check: {}",
                    state.brief_info()
                );

                Event::Ryukyoku { deltas: None }  // 点数変動（後で計算）
            }

            // 45: パス（何もしない）
            _ => Event::None,
        };

        // メタデータを生成
        let mut meta = self.gen_meta(state, action_idx);
        // 評価時間を計算（ナノ秒単位）
        let eval_time_ns = Instant::now()
            .checked_duration_since(start)       // リアクション生成時間
            .unwrap_or(Duration::ZERO)
            .saturating_add(self.last_eval_elapsed)  // NN評価時間を加算
            .as_nanos()                          // ナノ秒に変換
            .try_into()                          // u64に変換
            .unwrap_or(u64::MAX);                // オーバーフロー時は最大値

        // メタデータに追加情報を設定
        meta.eval_time_ns = Some(eval_time_ns);         // 評価時間
        meta.batch_size = Some(self.last_batch_size);   // バッチサイズ
        // カン選択があった場合、そのメタデータも記録
        meta.kan_select = kan_select_idx.map(|kan_idx| Box::new(self.gen_meta(state, kan_idx)));

        // イベントとメタデータを返す
        Ok(EventExt {
            event,
            meta: Some(meta),
        })
    }
}
