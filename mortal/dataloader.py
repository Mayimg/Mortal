"""
Mortal Dataloader Module
======================

このモジュールは麻雀AIの強化学習のためのデータローダーを実装しています。
主な機能：
1. 圧縮された麻雀ゲームログファイル（.gz形式）から学習データを読み込む
2. GRP（Game Result Predictor）モデルを使用して各局面での報酬を計算
3. PyTorchのIterableDatasetとして学習用バッチを生成
4. データ拡張（牌の回転）をサポート
5. マルチプロセスでのデータロードに対応

主要なクラス：
- FileDatasetsIter: ゲームログファイルから学習データを生成するイテラブルデータセット
"""

import random  # ファイルリストとバッファのシャッフル用
import torch  # PyTorchフレームワーク、モデルのロードとデータセット用
import numpy as np  # 数値計算、配列操作用
from torch.utils.data import IterableDataset  # PyTorchのイテラブルデータセット基底クラス
from model import GRP  # 麻雀の順位予測モデル（Game Result Predictor）
from reward_calculator import RewardCalculator  # GRPを使用して報酬を計算するユーティリティ
from libriichi.dataset import GameplayLoader  # Rustで実装された高速なゲームログローダー
from config import config  # 設定ファイルから読み込まれた設定情報

class FileDatasetsIter(IterableDataset):
    """
    麻雀ゲームログファイルから学習データを生成するイテラブルデータセット。
    複数のgzファイルを効率的に読み込み、学習用のバッチを生成します。
    """
    def __init__(
        self,
        version,  # データセットのバージョン（フォーマットの違いに対応）
        file_list,  # 読み込むgzファイルのパスリスト
        pts,  # 順位に対応するポイント値（例：[3, 1, -1, -3]）
        oracle = False,  # Trueの場合、不可視情報（他家の手牌など）も含める
        file_batch_size = 20, # hint: around 660 instances per file  # 一度に読み込むファイル数
        reserve_ratio = 0,  # バッファに残しておくデータの割合（0-1）
        player_names = None,  # 特定のプレイヤー名でフィルタリング（Noneで全プレイヤー）
        excludes = None,  # 除外するゲームのパターン（正規表現）
        num_epochs = 1,  # エポック数（データセット全体を何回繰り返すか）
        enable_augmentation = False,  # データ拡張（牌の回転）を有効にするか
        augmented_first = False,  # 拡張データを先に生成するか
    ):
        super().__init__()  # IterableDatasetの初期化
        self.version = version  # データセットバージョンを保存
        self.file_list = file_list  # 処理するファイルリストを保存
        self.pts = pts  # 順位ポイント値を保存
        self.oracle = oracle  # オラクルモード（不可視情報の使用）フラグを保存
        self.file_batch_size = file_batch_size  # ファイルバッチサイズを保存
        self.reserve_ratio = reserve_ratio  # バッファ予約率を保存
        self.player_names = player_names  # プレイヤー名フィルタを保存
        self.excludes = excludes  # 除外パターンを保存
        self.num_epochs = num_epochs  # エポック数を保存
        self.enable_augmentation = enable_augmentation  # 拡張フラグを保存
        self.augmented_first = augmented_first  # 拡張順序フラグを保存
        self.iterator = None  # イテレータ（初回アクセス時に初期化）

    def build_iter(self):
        """
        データのイテレータを構築する。
        GRPモデルのロードをここで行うのは、Windowsでのマルチプロセス対応のため。
        """
        # do not put it in __init__, it won't work on Windows
        # GRPモデルのインスタンスを作成（設定ファイルからネットワーク構造を読み込む）
        self.grp = GRP(**config['grp']['network'])
        # 学習済みGRPモデルの重みをロード（CPUに読み込み、weights_onlyで安全性を確保）
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        # モデルの重みを設定
        self.grp.load_state_dict(grp_state['model'])
        # 報酬計算器を初期化（GRPモデルと順位ポイントを使用）
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        # 指定されたエポック数だけデータを生成
        for _ in range(self.num_epochs):
            # 最初のパス：augmented_firstフラグに従ってデータを生成
            yield from self.load_files(self.augmented_first)
            # データ拡張が有効な場合、2回目のパスで反対の拡張設定でデータを生成
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        """
        ファイルリストからデータを読み込み、学習用エントリを生成する。
        
        Args:
            augmented: データ拡張（牌の回転）を適用するかどうか
        """
        # shuffle the file list for each epoch
        # ファイルリストをシャッフル（各エポックで順序を変えて学習の偏りを防ぐ）
        random.shuffle(self.file_list)

        # Rustで実装されたゲームプレイローダーを初期化
        self.loader = GameplayLoader(
            version = self.version,  # データセットのバージョン
            oracle = self.oracle,  # 不可視情報を含めるか
            player_names = self.player_names,  # プレイヤー名フィルタ
            excludes = self.excludes,  # 除外パターン
            augmented = augmented,  # データ拡張（牌の回転）を適用するか
        )
        # 学習データを一時的に保存するバッファ
        self.buffer = []

        # ファイルをバッチ単位で処理
        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            # バッファの現在のサイズを記録
            old_buffer_size = len(self.buffer)
            # 次のバッチのファイルからデータを読み込み、バッファに追加
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            # バッファの新しいサイズを取得
            buffer_size = len(self.buffer)

            # 予約サイズを計算（新しく追加されたデータの一部を次のバッチのために残す）
            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            # 予約サイズがバッファサイズを超える場合はスキップ
            if reserved_size > buffer_size:
                continue

            # バッファをシャッフル（データの順序をランダム化）
            random.shuffle(self.buffer)
            # 予約分を除いたデータを生成
            yield from self.buffer[reserved_size:]
            # 生成したデータをバッファから削除（予約分のみ残す）
            del self.buffer[reserved_size:]
        # 最後に残ったバッファのデータをシャッフル
        random.shuffle(self.buffer)
        # 残りのデータをすべて生成
        yield from self.buffer
        # バッファをクリア
        self.buffer.clear()

    def populate_buffer(self, file_list):
        """
        指定されたファイルリストからゲームデータを読み込み、バッファに追加する。
        
        Args:
            file_list: 読み込むgzファイルのパスリスト
        """
        # Rustローダーを使用してgz圧縮されたログファイルを読み込む
        data = self.loader.load_gz_log_files(file_list)
        # 各ファイルを処理
        for file in data:
            # 各ゲームを処理
            for game in file:
                # per move（各行動に関するデータ）
                # 観測可能な状態（手牌、場の情報など）
                obs = game.take_obs()
                # オラクルモードの場合、不可視情報（他家の手牌など）も取得
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                # 実行されたアクション（打牌、鳴きなど）
                actions = game.take_actions()
                # 有効なアクションのマスク（ルール上可能な行動）
                masks = game.take_masks()
                # 各行動がどの局（東1局、南2局など）で行われたか
                at_kyoku = game.take_at_kyoku()
                # 各行動が局の終了時点かどうか
                dones = game.take_dones()
                # 割引率（gamma）を適用するかどうか（通常の行動は1、流局などは0）
                apply_gamma = game.take_apply_gamma()

                # per game（ゲーム全体に関するデータ）
                # GRP用のゲーム状態情報（局数、本場、供託、各プレイヤーの点数）
                grp = game.take_grp()
                # このゲームデータがどのプレイヤー視点か（0-3）
                player_id = game.take_player_id()

                # ゲーム内の総行動数
                game_size = len(obs)

                # GRP特徴量（各局の開始時点での状態）を取得
                grp_feature = grp.take_feature()
                # 各局終了時の順位（プレイヤーインデックス -> 順位）
                rank_by_player = grp.take_rank_by_player()
                # 各局での期待報酬（ポイント差分）を計算
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                # 報酬の数が局数以上であることを確認（通常は等しいが、最終局で行動がない場合は異なる）
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                # ゲーム終了時の最終得点
                final_scores = grp.take_final_scores()
                # 各局の得点系列（GRP特徴量の得点部分を1万倍に戻し、最終得点を追加）
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                # 各局での順位を計算（高得点順にソートして順位を取得、stableソートで同点時の順位を保持）
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                # 対象プレイヤーの各局での順位
                player_ranks = rank_by_player_seq[:, player_id]

                # 各ステップから局終了までのステップ数を計算（TD学習用）
                steps_to_done = np.zeros(game_size, dtype=np.int64)
                # 逆順に処理（終了から開始に向かって）
                for i in reversed(range(game_size)):
                    # 局が終了していない場合
                    if not dones[i]:
                        # 次のステップまでの距離 + 現在のステップでgammaを適用するか（0 or 1）
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                # 各行動について学習用エントリを作成
                for i in range(game_size):
                    # 学習データエントリを構築
                    entry = [
                        obs[i],  # 観測可能な状態
                        actions[i],  # 実行されたアクション
                        masks[i],  # 有効なアクションのマスク
                        steps_to_done[i],  # 局終了までのステップ数
                        kyoku_rewards[at_kyoku[i]],  # この局の期待報酬
                        player_ranks[at_kyoku[i] + 1],  # 次の局でのプレイヤー順位（0-3）
                    ]
                    # オラクルモードの場合、不可視情報を2番目に挿入
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    # エントリをバッファに追加
                    self.buffer.append(entry)

    def __iter__(self):
        """
        イテレータプロトコルの実装。
        初回アクセス時にイテレータを構築し、以降は同じイテレータを返す。
        """
        # イテレータが未初期化の場合、構築する
        if self.iterator is None:
            self.iterator = self.build_iter()
        # イテレータを返す
        return self.iterator

def worker_init_fn(*args, **kwargs):
    """
    PyTorchのDataLoaderでマルチプロセス処理を行う際の初期化関数。
    各ワーカーに異なるファイルセットを割り当てることで、データの重複を防ぐ。
    
    Args:
        *args, **kwargs: DataLoaderから渡される引数（使用しない）
    """
    # 現在のワーカー情報を取得
    worker_info = torch.utils.data.get_worker_info()
    # データセットインスタンスを取得
    dataset = worker_info.dataset
    # 各ワーカーが処理するファイル数を計算（切り上げ）
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    # このワーカーが処理するファイルの開始インデックス
    start = worker_info.id * per_worker
    # このワーカーが処理するファイルの終了インデックス
    end = start + per_worker
    # データセットのファイルリストをこのワーカー用にスライス
    dataset.file_list = dataset.file_list[start:end]
