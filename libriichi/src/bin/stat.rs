// このプログラムは、指定されたディレクトリ内の麻雀ゲームログファイル（.json, .json.gz）を解析し、
// 特定のプレイヤーの統計情報を集計・表示するコマンドラインツールです。
// 統計情報には、ゲーム数、順位分布、平均得点、和了率、放銃率、リーチ率、鳴き率など、
// プレイヤーのパフォーマンスを評価するための詳細なメトリクスが含まれます。

use std::env;  // コマンドライン引数を取得するための標準ライブラリ

use anyhow::{Context, Result};  // エラーハンドリングを簡潔に行うためのクレート（anyhow）
use riichi::stat::Stat;  // 麻雀統計情報を管理・計算するStat構造体

const USAGE: &str = "Usage: stat <DIR> <PLAYER_NAME>";  // プログラムの使用方法を示す定数文字列

fn main() -> Result<()> {  // メイン関数。エラーが発生する可能性があるため、Result型を返す
    let args: Vec<_> = env::args().collect();  // コマンドライン引数を収集してベクトルに格納
    let dir = args.get(1).context(USAGE)?;  // 第1引数（ディレクトリパス）を取得。存在しない場合はUSAGEを含むエラーを返す
    let player_name = args.get(2).context(USAGE)?;  // 第2引数（プレイヤー名）を取得。存在しない場合はUSAGEを含むエラーを返す

    let stat = Stat::from_dir(dir, player_name, false)?;  // 指定ディレクトリ内のログファイルを解析し、プレイヤーの統計情報を生成
                                                           // 第3引数のfalseは、詳細なデバッグ情報を出力しないことを示す
    println!("{stat}");  // Stat構造体のDisplay実装を使用して、統計情報を整形して標準出力に表示

    Ok(())  // 正常終了を示すOk(())を返す
}
