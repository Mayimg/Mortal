// mjaiボット実装 - mjaiプロトコルで通信するAIボットの実装
//
// このファイルは、mjaiプロトコルを使って麻雀AIと通信するボットクラスを実装します。
// Botクラスは、mjaiイベントを受け取り、AIエージェントを使って適切な行動を決定し、
// mjaiフォーマットで応答を返します。
//
// 主な機能：
// - mjaiイベントの受信と解析
// - プレイヤー状態の管理
// - AIエージェント（MortalBatchAgent）との連携
// - mjaiフォーマットでの応答生成

use super::EventWithCanAct; // アクション可能フラグ付きイベント型
use super::{Event, EventExt}; // mjaiイベント型
use crate::agent::{BatchAgent, MortalBatchAgent}; // AIエージェント
use crate::state::PlayerState; // プレイヤー状態管理

use anyhow::{Context, Result}; // エラーハンドリング
use pyo3::prelude::*; // Python バインディング
use serde_json as json; // JSON処理

// Pythonから利用可能なボットクラス
#[pyclass]
pub struct Bot {
    agent: MortalBatchAgent, // AIエージェント（推論エンジン）
    state: PlayerState,      // 現在のプレイヤー状態
    log: Vec<EventExt>,      // 現在の局のイベントログ
}

// Pythonメソッドの実装
#[pymethods]
impl Bot {
    // コンストラクタ - 新しいBotインスタンスを作成
    #[new]
    fn new(engine: PyObject, player_id: u8) -> Result<Self> {
        // AIエージェントを初期化（推論エンジンとプレイヤーIDを指定）
        let agent = MortalBatchAgent::new(engine, &[player_id])?;
        // プレイヤー状態を初期化
        let state = PlayerState::new(player_id);
        Ok(Self {
            agent,
            state,
            log: vec![], // イベントログは空で初期化
        })
    }

    /// Returns the reaction to `line`, if it can react, `None` otherwise.
    ///
    /// Set `can_act` or `line_json['can_act']` to `False` to force the bot to
    /// only update its state without making any reaction.
    ///
    /// Both `line` and the return value are JSON strings representing one
    /// single mjai event.
    // Pythonメソッド: mjaiイベントに対する反応を返す
    #[pyo3(name = "react")] // Python側でのメソッド名は"react"
    #[pyo3(signature = (line, /, *, can_act=true))] // can_actはキーワード引数、デフォルトtrue
    fn react_py(&mut self, line: &str, can_act: bool, py: Python<'_>) -> Result<Option<String>> {
        // GILを解放して処理を実行（パフォーマンス向上のため）
        py.allow_threads(move || self.react(line, can_act))
    }
}

// Botの内部実装
impl Bot {
    // mjaiイベントを処理し、必要に応じて反応を返す
    fn react(&mut self, line: &str, can_act: bool) -> Result<Option<String>> {
        // JSON文字列をEventWithCanAct型にパース
        let data: EventWithCanAct =
            json::from_str(line).with_context(|| format!("failed to parse event {line}"))?;

        // イベントタイプに応じた処理
        match data.event {
            Event::StartGame { .. } => {
                // ゲーム開始時：エージェントに通知
                self.agent.start_game(0)?;
            }
            Event::EndKyoku => {
                // 局終了時：ログをクリアし、エージェントに通知
                self.log.clear();
                self.agent.end_kyoku(0)?;
            }
            Event::EndGame => {
                // ゲーム終了時：エージェントに通知
                self.agent.end_game(0, &Default::default())?;
            }
            _ => {
                // その他のイベント：ログに追加
                self.log.push(EventExt::no_meta(data.event.clone()));
            }
        };

        // プレイヤー状態を更新し、可能なアクションを取得
        let cans = self.state.update(&data.event)?;
        // アクション不可の判定：
        // - can_actパラメータがfalse
        // - イベントのcan_actフィールドがfalse
        // - 状態から判断してアクション不可
        if !can_act || matches!(data.can_act, Some(false)) || !cans.can_act() {
            return Ok(None); // アクションなし
        }

        // AIエージェントに現在の状態を設定
        self.agent
            .set_scene(0, &self.log, &self.state, None) // 0はシーンインデックス
            .context("failed to add state")?;
        // AIエージェントから反応（アクション）を取得
        let reaction = self
            .agent
            .get_reaction(0, &self.log, &self.state, None)
            .context("failed to get reaction")?;

        // 反応をJSON文字列に変換して返す
        let ret = json::to_string(&reaction)?;
        Ok(Some(ret))
    }
}
