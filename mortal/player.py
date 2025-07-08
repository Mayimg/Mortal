"""
player.py - 麻雀AIプレイヤーのテストと訓練用クラス

このファイルは、Mortal麻雀AIの性能評価と訓練のための2つのプレイヤークラスを定義しています。

1. TestPlayer: 
   - 新しいモデルとベースラインモデルを対戦させて性能を評価
   - 固定されたシード値で再現可能なテストを実行
   - 統計情報を収集して返す

2. TrainPlayer:
   - 訓練中のモデルをベースラインモデルと対戦させて評価
   - Boltzmannサンプリングで確率的な行動選択を行う
   - 訓練データとなるゲームログを生成

両クラスは共通して、OneVsThree環境（1対3の対戦形式）を使用し、
チャレンジャー（評価対象）が4つの座席で各3人のチャンピオン（ベースライン）と対戦します。
"""

import torch  # PyTorchディープラーニングフレームワーク
import numpy as np  # 数値計算ライブラリ
import os  # OS関連の操作（環境変数の取得など）
import shutil  # ファイル・ディレクトリ操作（ログディレクトリの削除など）
import secrets  # 暗号論的に安全な乱数生成（訓練用シードの生成）
import logging  # ログ出力
from os import path  # パス操作用のサブモジュール
from model import Brain, DQN  # ニューラルネットワークモデルクラス
from engine import MortalEngine  # 麻雀AI推論エンジン
from libriichi.stat import Stat  # ゲーム統計情報の収集・計算
from libriichi.arena import OneVsThree  # 1対3の対戦環境
from config import config  # 設定ファイルの読み込み

class TestPlayer:
    """テスト用プレイヤークラス - 新しいモデルの性能を評価"""
    
    def __init__(self):
        # ベースラインモデルの設定を取得（config.tomlのbaseline.testセクション）
        baseline_cfg = config['baseline']['test']
        # 推論に使用するデバイス（CPU/CUDA）を指定
        device = torch.device(baseline_cfg['device'])

        # ベースラインモデルの状態（重み）を読み込む
        # weights_only=True: セキュリティ向上のため重みのみ読み込み
        # map_location='cpu': 一旦CPUに読み込んでから必要に応じてGPUに転送
        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        # 保存されたモデルの設定情報を取得
        cfg = state['config']
        # モデルのバージョン（観測形式とアーキテクチャを決定）デフォルトは1
        version = cfg['control'].get('version', 1)
        # ResNetの畳み込みチャンネル数
        conv_channels = cfg['resnet']['conv_channels']
        # ResNetブロックの数（ネットワークの深さ）
        num_blocks = cfg['resnet']['num_blocks']
        # Brainモデル（特徴抽出器）のインスタンスを作成し、評価モードに設定
        # eval()により、BatchNormやDropoutが推論用の動作になる
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        # DQNモデル（行動価値関数）のインスタンスを作成し、評価モードに設定
        stable_dqn = DQN(version=version).eval()
        # 保存された重みをBrainモデルに読み込む
        stable_mortal.load_state_dict(state['mortal'])
        # 保存された重みをDQNモデルに読み込む
        stable_dqn.load_state_dict(state['current_dqn'])
        # PyTorch 2.0のcompile機能が有効な場合、モデルをコンパイル
        # コンパイルにより推論速度が向上する可能性がある
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        # ベースライン用のMortalEngineインスタンスを作成
        self.baseline_engine = MortalEngine(
            stable_mortal,  # 特徴抽出器
            stable_dqn,  # 行動価値関数
            is_oracle = False,  # 相手の手牌を見ない通常モード
            version = version,  # モデルバージョン
            device = device,  # 実行デバイス
            enable_amp = True,  # Automatic Mixed Precision（混合精度演算）を有効化
            enable_rule_based_agari_guard = True,  # ルールベースの和了判断を有効化
            name = 'baseline',  # エンジンの識別名
        )
        # チャレンジャー（評価対象）のモデルバージョン
        self.chal_version = config['control']['version']
        # ゲームログの保存ディレクトリ（絶対パスに変換）
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(self, seed_count, mortal, dqn, device):
        """
        テスト対戦を実行し、統計情報を収集
        
        Args:
            seed_count: 対戦するゲーム数（実際は4倍の局数が実行される）
            mortal: 評価対象のBrainモデル
            dqn: 評価対象のDQNモデル
            device: 実行デバイス
        
        Returns:
            Stat: ゲーム統計情報
        """
        # CuDNNのベンチマークを無効化（再現性を確保するため）
        torch.backends.cudnn.benchmark = False
        # チャレンジャー用のMortalEngineを作成
        engine_chal = MortalEngine(
            mortal,  # 評価対象の特徴抽出器
            dqn,  # 評価対象の行動価値関数
            is_oracle = False,  # 通常モード
            version = self.chal_version,  # モデルバージョン
            device = device,  # 実行デバイス
            enable_amp = True,  # 混合精度演算を有効化
            name = 'mortal',  # エンジン名（ログでの識別用）
        )

        # ログディレクトリが既に存在する場合は削除（クリーンな状態から始める）
        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        # 1対3の対戦環境を作成
        env = OneVsThree(
            disable_progress_bar = False,  # 進捗バーを表示
            log_dir = self.log_dir,  # ゲームログの保存先
        )
        # Pythonエンジン同士の対戦を実行
        env.py_vs_py(
            challenger = engine_chal,  # 評価対象のエンジン
            champion = self.baseline_engine,  # ベースラインエンジン
            seed_start = (10000, 0x2000),  # 乱数シードの開始値（再現性を保証）
            seed_count = seed_count,  # 実行するゲーム数
        )

        # ログディレクトリから'mortal'プレイヤーの統計情報を収集
        stat = Stat.from_dir(self.log_dir, 'mortal')
        # CuDNNベンチマーク設定を元に戻す
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        # 統計情報を返す
        return stat

class TrainPlayer:
    """訓練用プレイヤークラス - 訓練中のモデルを評価し、訓練データを生成"""
    
    def __init__(self):
        # 訓練用ベースラインモデルの設定を取得
        baseline_cfg = config['baseline']['train']
        # 推論に使用するデバイスを指定
        device = torch.device(baseline_cfg['device'])

        # ベースラインモデルの状態を読み込む
        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        # 保存されたモデルの設定情報を取得
        cfg = state['config']
        # モデルバージョン
        version = cfg['control'].get('version', 1)
        # ResNetの設定
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        # BrainモデルとDQNモデルのインスタンスを作成
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        # 保存された重みを読み込む
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        # コンパイルが有効な場合はコンパイル
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        # ベースライン用のMortalEngineを作成
        self.baseline_engine = MortalEngine(
            stable_mortal,  # 特徴抽出器
            stable_dqn,  # 行動価値関数
            is_oracle = False,  # 通常モード
            version = version,  # モデルバージョン
            device = device,  # 実行デバイス
            enable_amp = True,  # 混合精度演算を有効化
            enable_rule_based_agari_guard = True,  # ルールベースの和了判断
            name = 'baseline',  # エンジン名
        )

        # 環境変数から訓練プロファイルを取得（デフォルトは'default'）
        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        # 使用するプロファイルをログに記録
        logging.info(f'using profile {profile}')
        # 訓練用の設定を取得
        cfg = config['train_play'][profile]
        # チャレンジャーのモデルバージョン
        self.chal_version = config['control']['version']
        # ログディレクトリ（絶対パス）
        self.log_dir = path.abspath(cfg['log_dir'])
        # 暗号論的に安全な64ビットの乱数を訓練用キーとして生成
        self.train_key = secrets.randbits(64)
        # 訓練用シードの初期値
        self.train_seed = 10000

        # ゲーム数からシード数を計算（1シードで4ゲーム）
        self.seed_count = cfg['games'] // 4
        # Boltzmannサンプリングのエプシロン値（探索の確率）
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        # Boltzmannサンプリングの温度パラメータ（ランダム性の強さ）
        self.boltzmann_temp = cfg['boltzmann_temp']
        # Top-pサンプリングの確率闾値
        self.top_p = cfg['top_p']

        # 同じシードで何回対戦を繰り返すか
        self.repeats = cfg['repeats']
        # 繰り返し回数のカウンター
        self.repeat_counter = 0

    def train_play(self, mortal, dqn, device):
        """
        訓練用の対戦を実行し、ゲームログを生成
        
        Args:
            mortal: 訓練中のBrainモデル
            dqn: 訓練中のDQNモデル
            device: 実行デバイス
        
        Returns:
            rankings: 各ゲームの順位結果の配列
            file_list: 生成されたログファイルのリスト
        """
        # CuDNNベンチマークを無効化（再現性を確保）
        torch.backends.cudnn.benchmark = False
        # 訓練中のモデル用のMortalEngineを作成
        engine_chal = MortalEngine(
            mortal,  # 訓練中の特徴抽出器
            dqn,  # 訓練中の行動価値関数
            is_oracle = False,  # 通常モード
            version = self.chal_version,  # モデルバージョン
            boltzmann_epsilon = self.boltzmann_epsilon,  # 探索確率
            boltzmann_temp = self.boltzmann_temp,  # 温度パラメータ
            top_p = self.top_p,  # Top-pサンプリングの闾値
            device = device,  # 実行デバイス
            enable_amp = True,  # 混合精度演算を有効化
            name = 'trainee',  # エンジン名（訓練中のモデル）
        )

        # ログディレクトリが存在する場合は削除
        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        # 1対3の対戦環境を作成
        env = OneVsThree(
            disable_progress_bar = False,  # 進捗バーを表示
            log_dir = self.log_dir,  # ログの保存先
        )
        # Pythonエンジン同士の対戦を実行
        rankings = env.py_vs_py(
            challenger = engine_chal,  # 訓練中のモデル
            champion = self.baseline_engine,  # ベースラインモデル
            seed_start = (self.train_seed, self.train_key),  # 乱数シード
            seed_count = self.seed_count,  # ゲーム数
        )
        # 繰り返しカウンターを増やす
        self.repeat_counter += 1
        # 指定回数繰り返したら、次のシードに進む
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count  # シードを進める
            self.repeat_counter = 0  # カウンターをリセット

        # 順位結果をNumPy配列に変換
        rankings = np.array(rankings)
        # ログディレクトリ内の全ファイルの絶対パスを取得
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        # CuDNNベンチマーク設定を元に戻す
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        # 順位結果とログファイルリストを返す
        return rankings, file_list
