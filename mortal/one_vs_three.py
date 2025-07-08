"""
One vs Three 評価スクリプト
=========================

このスクリプトは麻雀AIモデルの性能評価を行うためのツールです。
1つのチャレンジャーモデルが3つのチャンピオンモデル（または赤穂AI）と対戦し、
その相対的な強さを測定します。

主な機能：
1. チャレンジャーとチャンピオンの2つのモデルをロード
2. それぞれのモデルからMortalEngineインスタンスを作成
3. Rustで実装されたOneVsThreeアリーナを使用して対戦を実行
4. チャレンジャーは4つの座席位置すべてでプレイして公平な評価を実現
5. 順位分布と平均順位・平均得点を計算して表示

設定はconfig.tomlの[1v3]セクションで管理されます。
"""

# preludeモジュールをインポート（プロジェクト固有の初期化処理を含む）
import prelude

# 数値計算用のNumPyライブラリ
import numpy as np
# PyTorchディープラーニングフレームワーク
import torch
# 暗号学的に安全な乱数生成のためのライブラリ
import secrets
# OS関連の操作（環境変数の設定など）
import os
# モデル定義（BrainエンコーダーとDQNデコーダー）
from model import Brain, DQN
# 推論エンジン（モデルを使用して実際の行動決定を行う）
from engine import MortalEngine
# Rustで実装された1対3対戦アリーナ
from libriichi.arena import OneVsThree
# プロジェクトの設定を管理
from config import config

def main():
    """メイン関数：1対3の対戦評価を実行"""
    # config.tomlから[1v3]セクションの設定を取得
    cfg = config['1v3']
    # 1イテレーションあたりのゲーム数（4の倍数である必要がある）
    games_per_iter = cfg['games_per_iter']
    # 1イテレーションあたりのシード数（各シードで4ゲーム実行されるため4で割る）
    seeds_per_iter = games_per_iter // 4
    # 実行するイテレーション数
    iters = cfg['iters']
    # ゲームログを保存するディレクトリ
    log_dir = cfg['log_dir']
    # 赤穂AI（Akochan）をチャンピオンとして使用するかどうか
    use_akochan = cfg['akochan']['enabled']

    # シードキーの設定（walrus演算子:=を使用して代入と条件判定を同時に行う）
    # 設定ファイルにseed_keyが指定されていない場合（-1）は、ランダムな64ビット値を生成
    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    # 赤穂AIを使用する場合の設定
    if use_akochan:
        # 赤穂AIのインストールディレクトリを環境変数に設定
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        # 赤穂AIの戦術設定ファイルのパスを環境変数に設定
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        # Pythonモデルをチャンピオンとして使用する場合
        # チャンピオンモデルの状態をファイルからロード
        # weights_only=Trueで重みのみロード（セキュリティ向上）、CPUメモリに一旦ロード
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        # 保存された設定情報を取得
        cham_cfg = state['config']
        # モデルのバージョン（デフォルトは1）
        version = cham_cfg['control'].get('version', 1)
        # ResNetの畳み込みチャンネル数
        conv_channels = cham_cfg['resnet']['conv_channels']
        # ResNetのブロック数
        num_blocks = cham_cfg['resnet']['num_blocks']
        # Brainエンコーダーネットワークを作成し、評価モードに設定
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        # DQNデコーダーネットワークを作成し、評価モードに設定
        dqn = DQN(version=version).eval()
        # 保存されたBrainの重みをロード
        mortal.load_state_dict(state['mortal'])
        # 保存されたDQNの重みをロード
        dqn.load_state_dict(state['current_dqn'])
        # PyTorch 2.0のコンパイル機能を有効にする場合（推論速度向上）
        if cfg['champion']['enable_compile']:
            mortal.compile()
            dqn.compile()
        # チャンピオン用のMortalEngineインスタンスを作成
        engine_cham = MortalEngine(
            mortal,  # Brainエンコーダー
            dqn,     # DQNデコーダー
            is_oracle = False,  # 完全情報を使用しない（通常のプレイ）
            version = version,  # モデルバージョン
            device = torch.device(cfg['champion']['device']),  # 実行デバイス（CPU/GPU）
            enable_amp = cfg['champion']['enable_amp'],  # 自動混合精度演算の有効化
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],  # ルールベースの和了判断を使用
            name = cfg['champion']['name'],  # エンジンの識別名
        )

    # チャレンジャーモデルの設定（チャンピオンと同じ処理）
    # チャレンジャーモデルの状態をファイルからロード
    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    # 保存された設定情報を取得
    chal_cfg = state['config']
    # モデルのバージョン
    version = chal_cfg['control'].get('version', 1)
    # ResNetの畳み込みチャンネル数
    conv_channels = chal_cfg['resnet']['conv_channels']
    # ResNetのブロック数  
    num_blocks = chal_cfg['resnet']['num_blocks']
    # Brainエンコーダーネットワークを作成し、評価モードに設定
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    # DQNデコーダーネットワークを作成し、評価モードに設定
    dqn = DQN(version=version).eval()
    # 保存されたBrainの重みをロード
    mortal.load_state_dict(state['mortal'])
    # 保存されたDQNの重みをロード
    dqn.load_state_dict(state['current_dqn'])
    # PyTorch 2.0のコンパイル機能を有効にする場合
    if cfg['challenger']['enable_compile']:
        mortal.compile()
        dqn.compile()
    # チャレンジャー用のMortalEngineインスタンスを作成
    engine_chal = MortalEngine(
        mortal,  # Brainエンコーダー
        dqn,     # DQNデコーダー
        is_oracle = False,  # 完全情報を使用しない
        version = version,  # モデルバージョン
        device = torch.device(cfg['challenger']['device']),  # 実行デバイス
        enable_amp = cfg['challenger']['enable_amp'],  # 自動混合精度演算
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],  # ルールベース和了判断
        name = cfg['challenger']['name'],  # エンジンの識別名
    )

    # 乱数シードの開始値（再現性のため固定値を使用）
    seed_start = 10000
    # 各イテレーションを実行するループ
    # seed_startからseeds_per_iterずつ増加させながらiters回繰り返す
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        # イテレーション区切りの表示
        print('-' * 50)
        # 現在のイテレーション番号を表示
        print('#', i)
        # OneVsThreeアリーナのインスタンスを作成
        env = OneVsThree(
            disable_progress_bar = False,  # プログレスバーを表示
            log_dir = log_dir,  # ログファイルの保存先ディレクトリ
        )
        # 対戦モードに応じて実行
        if use_akochan:
            # 赤穂AI（3人）vs Pythonモデル（1人）の対戦
            rankings = env.ako_vs_py(
                engine = engine_chal,  # チャレンジャーのPythonモデル
                seed_start = (seed, key),  # 乱数シード（シード値とキーのタプル）
                seed_count = seeds_per_iter,  # 実行するシード数
            )
        else:
            # Pythonモデル同士の対戦（チャレンジャー1人 vs チャンピオン3人）
            rankings = env.py_vs_py(
                challenger = engine_chal,  # チャレンジャーモデル
                champion = engine_cham,    # チャンピオンモデル
                seed_start = (seed, key),  # 乱数シード
                seed_count = seeds_per_iter,  # 実行するシード数
            )
        # 順位分布をNumPy配列に変換（[1位の回数, 2位の回数, 3位の回数, 4位の回数]）
        rankings = np.array(rankings)
        # 平均順位を計算（内積を使用：順位×回数の総和 / 総ゲーム数）
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        # 平均得点を計算（内積を使用：順位別得点×回数の総和 / 総ゲーム数）
        # 得点配分：1位+90点、2位+45点、3位0点、4位-135点
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        # 結果を表示（順位分布、平均順位、平均得点）
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

# スクリプトが直接実行された場合のみmain関数を呼び出す
if __name__ == '__main__':
    try:
        # メイン関数を実行
        main()
    except KeyboardInterrupt:
        # Ctrl+Cによる中断を優雅に処理（エラーメッセージを表示しない）
        pass
