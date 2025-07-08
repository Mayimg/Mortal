// このプログラムは、麻雀ゲームのログファイル（MJAIフォーマット）を検証するツールです。
// 指定されたディレクトリ内のすべてのログファイル（.json, .json.gz）を並列処理で検証し、
// ゲーム内の各アクション（打牌、鳴き、リーチ、和了など）が麻雀のルールに従っているかをチェックします。
// 不正なアクションが検出された場合、エラーメッセージとともに該当箇所の詳細情報を出力します。

use riichi::chi_type::ChiType;  // チー（順子）の種類（Low/Mid/High）を判定するための型
use riichi::mjai::Event;  // MJAIプロトコルのイベント（打牌、鳴き、和了などのゲームアクション）を表す型
use riichi::state::{ActionCandidate, PlayerState};  // プレイヤーの状態管理と可能なアクションの候補を表す型
use std::convert::identity;  // Result型の変換で使用するidentity関数
use std::env;  // コマンドライン引数を取得するため
use std::fs::File;  // ファイル読み込み用
use std::io;  // ファイルI/O操作用
use std::panic::catch_unwind;  // パニックを捕捉してエラーとして処理するため
use std::path::Path;  // ファイルパスの操作用
use std::time::Duration;  // プログレスバーの更新間隔設定用

use anyhow::{Context, Result, anyhow, ensure};  // エラーハンドリングとコンテキスト付きエラーメッセージ生成
use flate2::read::GzDecoder;  // gzip圧縮されたファイルの読み込み用
use glob::glob;  // ワイルドカードパターンでファイルを検索するため
use indicatif::{ProgressBar, ProgressStyle};  // 処理進捗を表示するプログレスバー
use rayon::prelude::*;  // 並列処理のためのクレート
use serde_json as json;  // JSONのパース用（MJAIフォーマットはJSON行形式）

const USAGE: &str = "Usage: validate_logs <DIR>";  // プログラムの使用方法

fn main() -> Result<()> {  // メイン関数。エラー処理のためResult型を返す
    let args: Vec<_> = env::args().collect();  // コマンドライン引数をベクトルに収集
    let dir = args.get(1).context(USAGE)?;  // 第1引数（検証対象ディレクトリ）を取得。なければエラー

    // プログレスバーのテンプレート定義：スピナー、経過時間、処理済みファイル数、処理速度を表示
    const TEMPLATE: &str = "{spinner:.cyan} [{elapsed_precise}] {pos} ({per_sec})";
    let bar = ProgressBar::new_spinner()  // スピナー型のプログレスバーを作成
        .with_style(ProgressStyle::with_template(TEMPLATE)?.tick_chars(".oO°Oo*"));  // アニメーション文字を設定
    bar.enable_steady_tick(Duration::from_millis(150));  // 150ミリ秒ごとにスピナーを更新

    glob(&format!("{dir}/**/*.json"))?  // 指定ディレクトリ以下のすべての.jsonファイルを検索
        .chain(glob(&format!("{dir}/**/*.json.gz"))?)  // .json.gzファイルも検索対象に追加
        .par_bridge()  // 並列処理用のイテレータに変換（rayonクレートの機能）
        .try_for_each(|path| {  // 各ファイルパスに対して並列で処理を実行
            bar.inc(1);  // プログレスバーのカウンタをインクリメント
            let path = path?;  // glob結果からパスを取得（エラーハンドリング込み）

            // パニックを捕捉してエラーとして処理（検証中の予期しないエラーにも対応）
            let result = catch_unwind(|| process_path(&path))  // process_path関数を実行し、パニックを捕捉
                .map_err(|pnc| {  // パニックをanyhowエラーに変換
                    if let Some(v) = pnc.downcast_ref::<String>() {  // String型のパニックメッセージ
                        anyhow!("{v}")
                    } else if let Some(v) = pnc.downcast_ref::<&str>() {  // &str型のパニックメッセージ
                        anyhow!("{v}")
                    } else {
                        anyhow!("Non-string panic")  // その他の型のパニック
                    }
                })
                .and_then(identity)  // Result<Result<T, E>, E>をResult<T, E>に平坦化
                .with_context(|| format!("error in log {}", path.display()));  // エラーにファイルパスのコンテキストを追加
            if let Err(err) = result {  // エラーが発生した場合
                println!("\n{err:?}");  // エラー詳細を標準出力に表示（デバッグ形式）
            }

            anyhow::Ok(())  // 処理継続のためOkを返す（個別ファイルのエラーで全体を停止しない）
        })?;

    bar.abandon();  // プログレスバーを終了（画面に残す）

    Ok(())  // 正常終了
}

fn process_path(path: &Path) -> Result<()> {  // 各ログファイルを検証する関数
    // ファイルの拡張子に応じて読み込み方法を切り替え
    let raw_log = if path
        .extension()  // ファイル拡張子を取得
        .is_some_and(|s| s.eq_ignore_ascii_case("gz"))  // 拡張子が"gz"かを大文字小文字を無視してチェック
    {
        io::read_to_string(GzDecoder::new(File::open(path)?))?  // gzip圧縮ファイルを解凍して読み込み
    } else {
        io::read_to_string(File::open(path)?)?  // 通常のファイルを読み込み
    };
    // MJAIフォーマット（各行が1つのJSONイベント）をパースしてEventのベクトルに変換
    let events: Vec<Event> = raw_log
        .lines()  // ファイルを行単位で分割
        .map(|l| Ok(json::from_str(l)?))  // 各行をJSONとしてパースし、Event型にデシリアライズ
        .collect::<Result<_>>()?;  // 結果を収集し、エラーがあれば早期リターン

    // 4人のプレイヤー状態を初期化（0から3はプレイヤーID）
    let mut states = [
        PlayerState::new(0),  // プレイヤー0の状態
        PlayerState::new(1),  // プレイヤー1の状態
        PlayerState::new(2),  // プレイヤー2の状態
        PlayerState::new(3),  // プレイヤー3の状態
    ];
    // 各プレイヤーが実行可能なアクション候補を格納する配列
    let mut cans = [ActionCandidate::default(); 4];  // 4人分のアクション候補をデフォルト値で初期化

    // 各イベントを順番に検証
    for (idx, ev) in events.iter().enumerate() {  // enumerate()でインデックス付きでイテレート
        let line = idx + 1;  // ログファイルの行番号（1ベース）を計算
        match ev {  // イベントの種類に応じた検証処理
            // 打牌（ツモ切り・手出し）イベントの検証
            Event::Dahai { actor, pai, .. } => {  // actor: 打牌したプレイヤーID、pai: 打牌した牌、..: その他のフィールドを無視
                // プレイヤーが打牌可能な状態かをチェック
                ensure!(  // 条件を満たさない場合はエラーを返すマクロ
                    cans[*actor as usize].can_discard,  // 該当プレイヤーのcan_discardフラグを確認
                    "fails can_discard at line {line}\naction: {ev:?}\nstate:\n{}",  // エラーメッセージ
                    states[*actor as usize].brief_info(),  // プレイヤー状態の要約情報を含む
                );

                // 打牌した牌が実際に手牌に存在するかをチェック
                let discard_candidates = states[*actor as usize].discard_candidates_aka();  // 赤ドラを考慮した打牌可能牌のビットマップを取得
                ensure!(  // 打牌した牌が候補に含まれているかを検証
                    discard_candidates[pai.as_usize()],  // 牌のインデックスでビットマップを参照
                    "fails discard_candidates at line {line}\naction: {ev:?}\nstate:\n{}",  // エラーメッセージ
                    states[*actor as usize].brief_info(),  // プレイヤー状態情報
                );
            }
            // チー（順子を作る鳴き）イベントの検証
            Event::Chi {
                actor,     // チーを宣言したプレイヤーID
                consumed,  // 手牌から使用する2枚の牌の配列
                pai,       // チーした牌（他家から取得した牌）
                target,    // チーの対象となったプレイヤーID（打牌したプレイヤー）
            } => {
                // チーは上家（左側のプレイヤー）からのみ可能であることを検証
                ensure!(
                    (target + 1) % 4 == *actor,  // targetの次のプレイヤーがactorであるかをチェック（左回り）
                    "chi from non-kamicha at line {}\naction: {ev:?}\nstate:\n{}",  // 上家以外からのチーエラー
                    line,
                    states[*actor as usize].brief_info(),
                );

                // チーの種類（Low/Mid/High）を判定し、対応するアクションが可能かを検証
                match ChiType::new(*consumed, *pai) {  // consumedの2枚とpaiを組み合わせてチーの種類を判定
                    ChiType::Low => {  // Low: paiが最小値（例：1-2-3の1）
                        ensure!(
                            cans[*actor as usize].can_chi_low,  // Lowチーが可能かを確認
                            "fails can_chi_low at line {}\naction: {ev:?}\nstate:\n{}",
                            line,
                            states[*actor as usize].brief_info(),
                        );
                    }
                    ChiType::Mid => {  // Mid: paiが中間値（例：1-2-3の2）
                        ensure!(
                            cans[*actor as usize].can_chi_mid,  // Midチーが可能かを確認
                            "fails can_chi_mid at line {}\naction: {ev:?}\nstate:\n{}",
                            line,
                            states[*actor as usize].brief_info(),
                        );
                    }
                    ChiType::High => {  // High: paiが最大値（例：1-2-3の3）
                        ensure!(
                            cans[*actor as usize].can_chi_high,  // Highチーが可能かを確認
                            "fails can_chi_high at line {}\naction: {ev:?}\nstate:\n{}",
                            line,
                            states[*actor as usize].brief_info(),
                        );
                    }
                }
            }
            // ポン（刻子を作る鳴き）イベントの検証
            Event::Pon { actor, .. } => {  // actor: ポンを宣言したプレイヤー、..: その他のフィールドは省略
                ensure!(
                    cans[*actor as usize].can_pon,  // ポンが可能な状態かをチェック
                    "fails can_pon at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );
            }
            // 大明槓（他家の捨て牌から明槓を作る）イベントの検証
            Event::Daiminkan { actor, .. } => {  // actor: 大明槓を宣言したプレイヤー
                ensure!(
                    cans[*actor as usize].can_daiminkan,  // 大明槓が可能な状態かをチェック
                    "fails can_daiminkan at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );
            }
            // 暗槓（手牌から4枚同じ牌で槓を作る）イベントの検証
            Event::Ankan { actor, consumed } => {  // actor: 暗槓を宣言したプレイヤー、consumed: 使用した4枚の牌
                ensure!(
                    cans[*actor as usize].can_ankan,  // 暗槓が可能な状態かをチェック
                    "fails can_ankan at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );

                // 暗槓した牌が実際に暗槓可能な候補に含まれているかを検証
                let ankan_candidates = states[*actor as usize].ankan_candidates();  // 暗槓可能な牌のリストを取得
                ensure!(
                    ankan_candidates.contains(&consumed[0].deaka()),  // 最初の牌（赤ドラを除去）が候補に含まれているか
                    "fails ankan_candidates at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );
            }
            // 加槓（ポンした刻子に1枚加えて槓にする）イベントの検証
            Event::Kakan { actor, pai, .. } => {  // actor: 加槓を宣言したプレイヤー、pai: 加える牌
                ensure!(
                    cans[*actor as usize].can_kakan,  // 加槓が可能な状態かをチェック
                    "fails can_kakan at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );

                // 加槓した牌が実際に加槓可能な候補に含まれているかを検証
                let kakan_candidates = states[*actor as usize].kakan_candidates();  // 加槓可能な牌のリストを取得
                ensure!(
                    kakan_candidates.contains(&pai.deaka()),  // 追加する牌（赤ドラを除去）が候補に含まれているか
                    "fails kakan_candidates at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );
            }
            // リーチ宣言イベントの検証
            Event::Reach { actor } => {  // actor: リーチを宣言したプレイヤー
                ensure!(
                    cans[*actor as usize].can_riichi,  // リーチが可能な状態かをチェック（テンパイ、門前、持ち点1000点以上など）
                    "fails can_riichi at line {line}\naction: {ev:?}\nstate:\n{}",
                    states[*actor as usize].brief_info(),
                );
            }
            // 和了（あがり）イベントの検証
            Event::Hora {
                actor,        // 和了したプレイヤーID
                target,       // 和了の対象プレイヤーID（ロンの場合は放銃者、ツモの場合は自分自身）
                ura_markers,  // 裏ドラ表示牌のリスト（リーチ和了時のみ）
                deltas,       // 各プレイヤーの点数変動
            } => {
                let is_ron = actor != target;  // actorとtargetが異なる場合はロン、同じ場合はツモ
                if is_ron {
                    // ロン和了の場合の検証
                    ensure!(
                        cans[*actor as usize].can_ron_agari,  // ロン和了が可能な状態かをチェック
                        "fails can_ron_agari at line {line}\naction: {ev:?}\nstate:\n{}",
                        states[*actor as usize].brief_info(),
                    );
                } else {
                    // ツモ和了の場合の検証
                    ensure!(
                        cans[*actor as usize].can_tsumo_agari,  // ツモ和了が可能な状態かをチェック
                        "fails can_tsumo_agari at line {line}\naction: {ev:?}\nstate:\n{}",
                        states[*actor as usize].brief_info(),
                    );
                }

                // 点数計算の検証（概算チェック）
                // TODO: fix bug for double chankan ron
                let ura = ura_markers
                    .as_ref()
                    .context("missing field `ura_markers`")?;  // 裏ドラ表示牌を取得（必須フィールド）
                let deltas = deltas.context("missing field `deltas`")?;  // 点数変動を取得（必須フィールド）
                let points = states[*actor as usize]
                    .agari_points(is_ron, ura)  // 和了点数を計算（役、符、翻数から）
                    .with_context(|| {
                        format!(
                            "failed to get agari points at line {line}\naction: {ev:?}\nstate:\n{}",
                            states[*actor as usize].brief_info()
                        )
                    })?;

                // 実際の点数変動が計算された和了点以上であることを検証
                if is_ron {
                    ensure!(deltas[*actor as usize] >= points.ron);  // ロン和了の場合の点数チェック
                } else if states[*actor as usize].is_oya() {
                    ensure!(deltas[*actor as usize] >= points.tsumo_oya);  // 親のツモ和了の場合の点数チェック
                } else {
                    ensure!(deltas[*actor as usize] >= points.tsumo_ko);  // 子のツモ和了の場合の点数チェック
                }
            }
            _ => (),  // その他のイベント（StartGame、Tsumo、Doraなど）は検証不要
        }

        // すべてのプレイヤーの状態をイベントに応じて更新し、次のアクション候補を取得
        for (s, c) in states.iter_mut().zip(&mut cans) {  // 各プレイヤーの状態とアクション候補をペアでイテレート
            *c = s.update_with_keep_cans(ev, true)?;  // イベントを適用して状態を更新、新しいアクション候補を返す
                                                      // 第2引数のtrueは、アクション候補を保持することを指定
        }
    }

    Ok(())  // すべてのイベントの検証が成功した場合、正常終了
}
