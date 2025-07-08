"""Mortal麻雀AIエンジンモジュール

このモジュールは麻雀AI「Mortal」の推論エンジンを実装しています。
主に2つのエンジンクラスを提供：

1. MortalEngine: ニューラルネットワークベースの高度なAIエンジン
   - BrainとDQNモデルを使用して観測データから最適な行動を選択
   - Boltzmann探索やtop-pサンプリングなどの確率的手法をサポート
   - バッチ処理による効率的な推論が可能

2. ExampleMjaiLogEngine: シンプルなログベースのエンジン
   - mjai形式のイベントログを処理
   - 常にツモ切り（最後に引いた牌を捨てる）を行う単純な実装

モジュールには以下のヘルパー関数も含まれています：
- sample_top_p: top-pサンプリングアルゴリズムの実装
"""

import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class MortalEngine:
    """ニューラルネットワークベースの麻雀AIエンジン
    
    MortalEngineは、深層学習モデル（BrainとDQN）を使用して
    麻雀の観測データから最適な行動を選択するAIエンジンです。
    """
    
    def __init__(
        self,
        brain,  # Brainモデル: 観測データから特徴表現を抽出するエンコーダーネットワーク
        dqn,    # DQNモデル: 特徴表現から各行動のQ値（期待価値）を推定するネットワーク
        is_oracle,  # bool: オラクルモード（完全情報を使用）かどうか
        version,    # int: モデルのバージョン（1-4）、異なるネットワーク構造に対応
        device = None,  # torch.device: 推論に使用するデバイス（CPU/GPU）、Noneの場合CPUを使用
        stochastic_latent = False,  # bool: 潜在表現を確率的にサンプリングするか（バージョン1のみ）
        enable_amp = False,  # bool: Automatic Mixed Precision（自動混合精度）を有効にするか
        enable_quick_eval = True,  # bool: 高速評価モードを有効にするか
        enable_rule_based_agari_guard = False,  # bool: ルールベースの和了ガードを有効にするか
        name = 'NoName',  # str: エンジンの名前（ログ等で使用）
        boltzmann_epsilon = 0,  # float: Boltzmann探索のイプシロン値（0-1）、探索の確率
        boltzmann_temp = 1,     # float: Boltzmann探索の温度パラメータ、行動選択の確率分布の鋭さを制御
        top_p = 1,              # float: top-pサンプリングのパラメータ（0-1）、累積確率の閾値
    ):
        self.engine_type = 'mortal'  # エンジンタイプを'mortal'に設定（他のエンジンと区別するため）
        self.device = device or torch.device('cpu')  # デバイスが指定されていない場合はCPUを使用
        assert isinstance(self.device, torch.device)  # デバイスがtorch.deviceインスタンスであることを確認
        self.brain = brain.to(self.device).eval()  # Brainモデルを指定デバイスに移動し、評価モードに設定
        self.dqn = dqn.to(self.device).eval()      # DQNモデルを指定デバイスに移動し、評価モードに設定
        self.is_oracle = is_oracle  # オラクルモード（完全情報使用）のフラグを保存
        self.version = version      # モデルバージョンを保存（1-4の異なるアーキテクチャに対応）
        self.stochastic_latent = stochastic_latent  # 潜在表現の確率的サンプリングフラグ（v1のみ）

        # 各種機能フラグを保存
        self.enable_amp = enable_amp  # AMP（自動混合精度）の有効化フラグ
        self.enable_quick_eval = enable_quick_eval  # 高速評価モードの有効化フラグ
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard  # ルールベース和了ガードの有効化フラグ
        self.name = name  # エンジンの名前

        # 探索パラメータを保存
        self.boltzmann_epsilon = boltzmann_epsilon  # ε-greedy法のεに相当、探索確率
        self.boltzmann_temp = boltzmann_temp        # Boltzmann分布の温度、低いほど最良手に集中
        self.top_p = top_p                          # top-pサンプリングの累積確率閾値

    def react_batch(self, obs, masks, invisible_obs):
        """バッチ処理で複数の観測に対する行動を決定
        
        Args:
            obs: 観測データのリスト（各要素は1ゲームの観測）
            masks: 有効な行動のマスク（True=有効、False=無効）のリスト
            invisible_obs: オラクルモード時の完全情報観測のリスト（通常はNone）
            
        Returns:
            tuple: (actions, q_values, masks, is_greedy)
                - actions: 選択された行動のインデックスのリスト
                - q_values: 全行動のQ値のリスト
                - masks: 入力されたマスクのリスト（そのまま返す）
                - is_greedy: 各行動がgreedy選択かどうかのリスト
        """
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),  # AMPコンテキスト：有効時は自動的に混合精度を使用
                torch.inference_mode(),  # 推論モード：勾配計算を無効化してメモリ効率と速度を向上
            ):
                return self._react_batch(obs, masks, invisible_obs)  # 実際の処理は_react_batchメソッドに委譲
        except Exception as ex:
            # エラーが発生した場合、元の例外とトレースバックを含む新しい例外を発生
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        """実際のバッチ推論処理を実行（内部メソッド）
        
        このメソッドは以下の処理を行います：
        1. 入力データをPyTorchテンソルに変換
        2. モデルバージョンに応じた順伝播処理
        3. 行動選択（greedy or 確率的）
        4. 結果をPythonリストに変換して返す
        """
        # 入力データをPyTorchテンソルに変換
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)  # 観測をスタックしてテンソル化
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)  # マスクをスタックしてテンソル化
        if invisible_obs is not None:  # オラクルモードの場合
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)  # 完全情報観測もテンソル化
        batch_size = obs.shape[0]  # バッチサイズを取得（同時に処理するゲーム数）

        # モデルバージョンに応じた順伝播処理
        match self.version:
            case 1:  # バージョン1: VAE風のアーキテクチャ
                mu, logsig = self.brain(obs, invisible_obs)  # Brainが平均と対数分散を出力
                if self.stochastic_latent:  # 確率的潜在表現を使用する場合
                    # 正規分布からサンプリング（再パラメータ化トリック）
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()  # 1e-6は数値安定性のため
                else:  # 決定的潜在表現を使用する場合
                    latent = mu  # 平均値をそのまま使用
                q_out = self.dqn(latent, masks)  # DQNで潜在表現からQ値を計算
            case 2 | 3 | 4:  # バージョン2-4: 直接的なアーキテクチャ
                phi = self.brain(obs)  # Brainが特徴表現を直接出力
                q_out = self.dqn(phi, masks)  # DQNで特徴表現からQ値を計算

        # 行動選択処理
        if self.boltzmann_epsilon > 0:  # 探索を行う場合
            # 各ゲームでgreedy選択する確率を(1-epsilon)としてベルヌーイ分布からサンプリング
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            # Q値を温度で割ってBoltzmann分布のロジットを計算、無効な行動は-∞でマスク
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            # top-pサンプリングで確率的に行動を選択
            sampled = sample_top_p(logits, self.top_p)
            # greedyフラグに応じて最大Q値の行動か確率的選択を使用
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:  # 常にgreedy選択する場合
            # 全てのゲームでgreedy選択
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            # 最大Q値を持つ行動を選択
            actions = q_out.argmax(-1)

        # 結果をPythonリストに変換して返す
        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def sample_top_p(logits, p):
    """top-pサンプリング（nucleus sampling）の実装
    
    確率の高い行動から順に累積確率がpを超えるまでの行動候補から
    サンプリングを行う。これにより確率の低い行動を除外し、
    より安定した確率的選択を実現する。
    
    Args:
        logits: 各行動のロジット値（対数オッズ）のテンソル
        p: 累積確率の閾値（0-1）
        
    Returns:
        サンプリングされた行動インデックス
    """
    if p >= 1:  # p=1の場合、全ての行動が候補（通常のカテゴリカル分布からサンプリング）
        return Categorical(logits=logits).sample()
    if p <= 0:  # p=0の場合、常に最大確率の行動を選択（決定的）
        return logits.argmax(-1)
    
    # ロジットを確率に変換（softmax）
    probs = logits.softmax(-1)
    # 確率を降順にソート（高い確率から順に並べる）
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    # 累積確率を計算
    probs_sum = probs_sort.cumsum(-1)
    # 累積確率がpを超える位置を特定（その行動自体を含めるとpを超える）
    mask = probs_sum - probs_sort > p
    # pを超える位置以降の確率を0にする（サンプリング候補から除外）
    probs_sort[mask] = 0.
    # 残った確率分布から多項分布でサンプリング、元のインデックスを取得
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    """シンプルなmjaiログベースのエンジン
    
    このクラスは、mjai形式のイベントログを処理して行動を決定する
    最小限の実装例です。常にツモ切り（最後に引いた牌を捨てる）を
    行う単純な戦略を実装しています。
    
    主にテストやベースライン比較用として使用されます。
    """
    
    def __init__(self, name: str):
        """エンジンを初期化
        
        Args:
            name: エンジンの名前
        """
        self.engine_type = 'mjai-log'  # エンジンタイプを'mjai-log'に設定
        self.name = name               # エンジン名を保存
        self.player_ids = None         # プレイヤーIDのリスト（後でset_player_idsで設定）

    def set_player_ids(self, player_ids: List[int]):
        """プレイヤーIDを設定
        
        Args:
            player_ids: 各ゲームでのプレイヤーIDのリスト
        """
        self.player_ids = player_ids

    def react_batch(self, game_states):
        """バッチ処理で複数のゲーム状態に対する行動を決定
        
        Args:
            game_states: ゲーム状態のリスト（各要素はゲーム状態オブジェクト）
            
        Returns:
            list: 各ゲームに対するJSON形式の行動文字列のリスト
        """
        res = []  # 結果を格納するリスト
        for game_state in game_states:  # 各ゲーム状態に対して処理
            game_idx = game_state.game_index  # ゲームのインデックスを取得
            state = game_state.state          # 現在のゲーム状態を取得
            events_json = game_state.events_json  # mjaiイベントのJSON文字列を取得

            events = json.loads(events_json)  # JSONをパース
            assert events[0]['type'] == 'start_kyoku'  # 最初のイベントが局の開始であることを確認

            player_id = self.player_ids[game_idx]  # このゲームでのプレイヤーIDを取得
            cans = state.last_cans  # 実行可能な行動の情報を取得
            if cans.can_discard:  # 打牌可能な場合
                tile = state.last_self_tsumo()  # 最後にツモった牌を取得
                # ツモ切りの行動をJSON形式で作成
                res.append(json.dumps({
                    'type': 'dahai',     # 行動タイプ：打牌
                    'actor': player_id,  # 行動者のID
                    'pai': tile,         # 捨てる牌
                    'tsumogiri': True,   # ツモ切りフラグ
                }))
            else:  # 打牌以外の行動が必要な場合
                res.append('{"type":"none"}')  # 何もしない（パス）
        return res

    # ゲームのライフサイクルイベントに対応するメソッド
    # これらは特定のイベントで呼び出される。実装は必須だが、
    # このシンプルなエンジンでは何もしない（no-op）
    def start_game(self, game_idx: int):
        """ゲーム開始時に呼ばれる
        
        Args:
            game_idx: ゲームのインデックス
        """
        pass
        
    def end_kyoku(self, game_idx: int):
        """局終了時に呼ばれる
        
        Args:
            game_idx: ゲームのインデックス
        """
        pass
        
    def end_game(self, game_idx: int, scores: List[int]):
        """ゲーム終了時に呼ばれる
        
        Args:
            game_idx: ゲームのインデックス
            scores: 各プレイヤーの最終スコアのリスト
        """
        pass
