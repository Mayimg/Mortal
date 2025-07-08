"""
RewardCalculator: 麻雀AIの強化学習における報酬計算クラス

このクラスは、事前学習済みのGRP（Game Result Predictor）モデルを使用して、
麻雀ゲームの各局面における期待順位の変化を計算し、それを報酬信号として提供します。

主な機能：
1. GRPモデルによる順位確率予測
2. 期待ポイントの計算（順位確率 × ポイント配分）
3. 連続する局面間の期待ポイント差分を報酬として算出

これにより、最終的な勝敗のみでなく、ゲーム中の各アクションの価値を
より細かく評価できるようになり、効率的な強化学習を実現します。
"""

import torch  # PyTorchライブラリ：ニューラルネットワークの推論に使用
import numpy as np  # NumPyライブラリ：数値計算とデータ変換に使用

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False):
        """
        RewardCalculatorの初期化メソッド
        
        Args:
            grp: 事前学習済みのGRP（Game Result Predictor）モデル
                 GRUベースのニューラルネットワークで、ゲーム状態から順位確率を予測
            pts: 順位に対するポイント配分リスト（デフォルト: [3, 1, -1, -3]）
                 1位から4位までのポイントを定義（例: 1位=+3pt, 2位=+1pt, 3位=-1pt, 4位=-3pt）
            uniform_init: Trueの場合、初期状態の順位確率を一様分布（各25%）に設定
                         学習の安定性向上のためのオプション
        """
        # CPUデバイスを明示的に指定（報酬計算は推論のみでGPUを使用しない）
        self.device = torch.device('cpu')
        
        # GRPモデルをCPUに移動し、評価モード（推論モード）に設定
        # .eval()により、BatchNormやDropoutが推論用の動作になる
        self.grp = grp.to(self.device).eval()
        
        # 初期状態で一様分布を使用するかどうかのフラグを保存
        self.uniform_init = uniform_init

        # ポイント配分が指定されていない場合、デフォルト値を使用
        # 麻雀の一般的な順位点：1位=+3万点相当、4位=-3万点相当
        pts = pts or [3, 1, -1, -3]
        
        # ポイント配分をPyTorchテンソルに変換（float64で高精度計算）
        # CPUデバイスに配置して、後の行列計算で使用
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        """
        GRP特徴量から順位確率行列を計算するメソッド
        
        Args:
            grp_feature: ゲーム状態の時系列データ（各局の特徴量）
                        shape: (局数, 7) - 7次元は[局番号, 本場, 供託, 4人の得点]
        
        Returns:
            matrix: 各時点での各プレイヤーの順位確率行列
                   shape: (局数+1, 4, 4) - [時点, プレイヤー, 順位]
        """
        # 各時点までの累積シーケンスを作成
        # 例: 5局のデータがある場合、[1局目まで], [1-2局目まで], ..., [1-5局目まで]の5つのシーケンス
        seq = list(map(
            # 各インデックスに対して、そこまでの部分シーケンスをテンソル化
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),  # 0から局数-1までのインデックス
        ))

        # 推論モードで実行（勾配計算を無効化して高速化）
        with torch.inference_mode():
            # GRPモデルに全シーケンスを入力し、各時点での順位予測logitsを取得
            # logitsは24クラス分類（4人の順位の全順列）の生の出力値
            logits = self.grp(seq)
        
        # logitsから順位確率行列に変換
        # calc_matrixメソッドは、24クラスの確率を4×4の順位確率行列に変換
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        """
        特定プレイヤーの各時点での順位確率を計算するメソッド
        
        Args:
            player_id: 対象プレイヤーのID（0-3）
            grp_feature: ゲーム状態の時系列データ
            rank_by_player: 最終順位の配列（player_id -> 順位(0-3)のマッピング）
                           例: [2, 0, 3, 1] = プレイヤー0は3位、プレイヤー1は1位など
        
        Returns:
            rank_prob: 各時点での順位確率
                      shape: (局数+2, 4) - [時点, 順位確率]
                      最初の行は初期状態、最後の行は実際の最終順位（one-hot）
        """
        # GRP特徴量から全プレイヤーの順位確率行列を計算
        matrix = self.calc_grp(grp_feature)

        # 最終順位をone-hotベクトルで表現（確率1.0の確定状態）
        final_ranking = torch.zeros((1, 4), device=self.device)  # shape: (1, 4)
        # 該当プレイヤーの実際の最終順位に1.0を設定
        final_ranking[0, rank_by_player[player_id]] = 1.
        
        # 対象プレイヤーの順位確率の時系列を構築
        # matrix[:, player_id]: 各時点での該当プレイヤーの順位確率
        # final_ranking: 最終的な実際の順位（確定値）
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        
        # uniform_initがTrueの場合、初期状態を一様分布に設定
        # これにより、ゲーム開始時の不確実性を表現し、学習を安定化
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4  # 各順位の確率を25%に設定
            
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        """
        期待ポイントの変化量（報酬）を計算するメソッド
        
        このメソッドは強化学習の中核となる報酬計算を行います。
        各アクション後の期待順位の変化を、ポイント変化として定量化します。
        
        Args:
            player_id: 対象プレイヤーのID（0-3）
            grp_feature: ゲーム状態の時系列データ
            rank_by_player: 最終順位の配列
        
        Returns:
            reward: 各局での期待ポイントの変化量（numpy配列）
                   正の値は期待順位の改善、負の値は悪化を示す
        """
        # 各時点での順位確率を取得
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        
        # 期待ポイントを計算：順位確率と各順位のポイントの内積
        # 例: 1位30%, 2位40%, 3位20%, 4位10%の場合
        #     期待値 = 0.3*3 + 0.4*1 + 0.2*(-1) + 0.1*(-3) = 0.8
        exp_pts = rank_prob @ self.pts  # shape: (局数+2,)
        
        # 連続する局間の期待ポイントの差分を報酬として計算
        # reward[i] = exp_pts[i+1] - exp_pts[i]
        # 正の値：期待順位が上昇（良いアクション）
        # 負の値：期待順位が下降（悪いアクション）
        reward = exp_pts[1:] - exp_pts[:-1]
        
        # PyTorchテンソルからNumPy配列に変換して返す
        return reward.cpu().numpy()

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        """
        実際の得点変化を計算するメソッド（代替的な報酬計算方法）
        
        期待ポイントではなく、実際のゲーム内得点の変化を報酬として使用する場合に使用。
        より直接的だが、ノイズが多い報酬信号となる。
        
        Args:
            player_id: 対象プレイヤーのID（0-3）
            grp_feature: ゲーム状態の時系列データ
            final_scores: 最終得点の配列（各プレイヤーの最終得点）
        
        Returns:
            delta_points: 各局での実際の得点変化量
        """
        # GRP特徴量から該当プレイヤーの得点推移を抽出
        # grp_feature[:, 3 + player_id]: player_idの各局での得点（100点単位）
        # * 1e4: 100点単位を実際の得点に変換（例: 250 → 25000点）
        # 最後に最終得点を追加して完全な得点推移を作成
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        
        # 連続する局間の得点差分を計算
        # delta_points[i] = seq[i+1] - seq[i]
        # 正の値：得点増加、負の値：得点減少
        delta_points = seq[1:] - seq[:-1]
        
        return delta_points
