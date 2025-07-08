"""麻雀AI Mortalのニューラルネットワークモデル定義

 このファイルは麻雀AI「Mortal」で使用される深層学習モデルを定義しています。
 主要なコンポーネント：
 
 1. ChannelAttention: チャンネル方向の注意機構（SE-Netスタイル）
 2. ResBlock: 残差接続とチャンネルアテンションを持つ畳み込みブロック
 3. ResNet: 複数のResBlockから構成される特徴抽出器
 4. Brain: メインのエンコーダーネットワーク（観測から特徴表現を生成）
 5. AuxNet: 補助タスク用のネットワーク
 6. DQN: Deep Q-Network（行動価値関数を推定）
 7. GRP: Game Result Predictor（ゲーム結果予測用RNN）

 モデルは観測データ（obs_shape）を入力として受け取り、
 各可能な行動（ACTION_SPACE=46）に対するQ値を出力します。
 oracleモードでは追加の完全情報（oracle_obs_shape）も使用可能です。
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import *
from functools import partial
from itertools import permutations
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

class ChannelAttention(nn.Module):
    """チャンネル方向の注意機構（Squeeze-and-Excitation風）
    
    各チャンネルの重要度を学習し、動的に重み付けを行います。
    平均プーリングと最大プーリングの両方を使用してグローバル情報を集約します。
    """
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        # channels: 入力チャンネル数
        # ratio: 圧縮率（中間層のチャンネル数 = channels // ratio）
        # actv_builder: 活性化関数のコンストラクタ
        # bias: 線形層でバイアス項を使用するか
        super().__init__()
        # 共有MLPネットワーク：チャンネル数を一旦圧縮してから復元
        # これにより各チャンネルの重要度を学習
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),  # 圧縮層
            actv_builder(),                                      # 活性化関数
            nn.Linear(channels // ratio, channels, bias=bias),   # 復元層
        )
        # バイアス項を0で初期化（初期状態でニュートラルな重み付け）
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        # x: (batch, channels, spatial) の形状を想定
        # 平均プーリング：各チャンネルの平均的な活性度を計算
        avg_out = self.shared_mlp(x.mean(-1))  # (batch, channels)
        # 最大プーリング：各チャンネルの最大活性度を計算
        max_out = self.shared_mlp(x.amax(-1))  # (batch, channels)
        # 両方の情報を統合してsigmoidで[0,1]の重みに変換
        weight = (avg_out + max_out).sigmoid()  # (batch, channels)
        # 元の入力に重みを適用（チャンネルごとにスケーリング）
        x = weight.unsqueeze(-1) * x  # (batch, channels, 1) * (batch, channels, spatial)
        return x

class ResBlock(nn.Module):
    """残差ブロック with チャンネルアテンション
    
    ResNetスタイルの残差接続を持つブロック。
    pre-activation版とpost-activation版の両方をサポート。
    チャンネルアテンション機構も組み込まれている。
    """
    def __init__(
        self,
        channels,  # チャンネル数（入力・出力で同じ）
        *,
        norm_builder = nn.Identity,  # 正規化層のコンストラクタ（デフォルトは何もしない）
        actv_builder = nn.ReLU,      # 活性化関数のコンストラクタ
        pre_actv = False,            # pre-activation構造を使うか（True: BN→ReLU→Conv、False: Conv→BN→ReLU）
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            # Pre-activation構造：正規化→活性化→畳み込みの順
            # He et al. (2016)のIdentity Mappings in Deep Residual Networksで提案
            self.res_unit = nn.Sequential(
                norm_builder(),         # BatchNorm等
                actv_builder(),         # 活性化関数
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),  # 1次元畳み込み（バイアスなし、BNがあるため）
                norm_builder(),         # 2つ目のBatchNorm
                actv_builder(),         # 2つ目の活性化関数
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),  # 2つ目の畳み込み
            )
        else:
            # Post-activation構造：畳み込み→正規化→活性化の順（従来のResNet）
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),  # 1次元畳み込み
                norm_builder(),         # BatchNorm
                actv_builder(),         # 活性化関数
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),  # 2つ目の畳み込み
                norm_builder(),         # 2つ目のBatchNorm
            )
            # post-activationの場合、残差接続後に活性化が必要
            self.actv = actv_builder()
        # チャンネルアテンション層を追加
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        # 残差ユニットを通す（2層の畳み込み）
        out = self.res_unit(x)
        # チャンネルアテンションを適用
        out = self.ca(out)
        # 残差接続（入力をショートカット）
        out = out + x
        # post-activationの場合、最後に活性化関数を適用
        if not self.pre_actv:
            out = self.actv(out)
        return out

class ResNet(nn.Module):
    """ResNetベースの特徴抽出器
    
    複数のResBlockを積み重ねた深層ネットワーク。
    入力の観測データから1024次元の特徴ベクトルを生成。
    """
    def __init__(
        self,
        in_channels,    # 入力チャンネル数（観測データのチャンネル数）
        conv_channels,  # 畳み込み層のチャンネル数
        num_blocks,     # ResBlockの数
        *,
        norm_builder = nn.Identity,  # 正規化層のコンストラクタ
        actv_builder = nn.ReLU,      # 活性化関数のコンストラクタ
        pre_actv = False,            # pre-activation構造を使うか
    ):
        super().__init__()

        # 指定された数のResBlockを作成
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(
                conv_channels,
                norm_builder = norm_builder,
                actv_builder = actv_builder,
                pre_actv = pre_actv,
            ))

        # ネットワーク全体の構築
        # 最初の畳み込み層：入力チャンネルを内部チャンネル数に変換
        layers = [nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        
        if pre_actv:
            # pre-activationの場合：blocks → norm → actv
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            # post-activationの場合：norm → actv → blocks
            layers += [norm_builder(), actv_builder(), *blocks]
        
        # 最終層：特徴マップを1024次元のベクトルに変換
        layers += [
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),  # チャンネル数を32に削減
            actv_builder(),                                           # 活性化
            nn.Flatten(),                                             # (batch, 32, 34) → (batch, 32*34)
            nn.Linear(32 * 34, 1024),                                 # 全結合層で1024次元に
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, in_channels, 34) → (batch, 1024)
        return self.net(x)

class Brain(nn.Module):
    """メインのエンコーダーネットワーク
    
    観測データを受け取り、高次元の特徴表現を生成する中核となるネットワーク。
    複数のバージョンをサポートし、oracleモード（完全情報）にも対応。
    """
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1):
        # conv_channels: 畳み込み層のチャンネル数
        # num_blocks: ResBlockの数
        # is_oracle: 完全情報モード（相手の手牌も見える）
        # version: モデルのバージョン（1-4、それぞれ異なる観測形式とアーキテクチャ）
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        # 入力チャンネル数を決定（バージョンによって異なる）
        # version 1: 938, version 2: 942, version 3: 934, version 4: 1012
        in_channels = obs_shape(version)[0]
        if is_oracle:
            # oracleモードでは追加の観測チャンネルを連結
            # version 1: +211, version 2-4: +217
            in_channels += oracle_obs_shape(version)[0]

        # デフォルト設定（version 2以降）
        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01)  # BatchNormのmomentumを0.01に設定
        actv_builder = partial(nn.Mish, inplace=True)  # Mish活性化関数（滑らかなReLU variant）
        pre_actv = True  # pre-activation構造を使用

        # バージョン固有の設定
        match version:
            case 1:
                # version 1は古い設定：ReLU + post-activation
                actv_builder = partial(nn.ReLU, inplace=True)
                pre_actv = False
                # version 1のみ変分オートエンコーダ風の潜在変数を使用
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                )
                self.mu_head = nn.Linear(512, 512)      # 平均μ
                self.logsig_head = nn.Linear(512, 512)  # 対数標準偏差log(σ)
            case 2:
                pass  # デフォルト設定をそのまま使用
            case 3 | 4:
                # version 3, 4はBatchNormのepsを小さく設定（数値安定性の調整）
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

        # メインのエンコーダーネットワークを構築
        self.encoder = ResNet(
            in_channels = in_channels,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = norm_builder,
            actv_builder = actv_builder,
            pre_actv = pre_actv,
        )
        # version 2以降で使用する最終活性化関数
        self.actv = actv_builder()

        # BatchNormのフリーズフラグ（EMA: Exponential Moving Average、CMA: Cumulative Moving Average）
        # Trueの場合、BatchNormの統計量を更新せず、推論モードで固定
        self._freeze_bn = False

    def forward(self, obs: Tensor, invisible_obs: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        # obs: 通常の観測データ (batch, channels, 34)
        # invisible_obs: oracle用の追加観測データ（相手の手牌など）
        
        if self.is_oracle:
            # oracleモードでは両方の観測を連結
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)  # チャンネル方向に連結
        
        # エンコーダーで特徴抽出 → 1024次元のベクトル
        phi = self.encoder(obs)

        # バージョンによって出力形式が異なる
        match self.version:
            case 1:
                # version 1: 変分推論用の平均と対数標準偏差を返す
                latent_out = self.latent_net(phi)     # 1024 → 512
                mu = self.mu_head(latent_out)         # 512 → 512 (平均)
                logsig = self.logsig_head(latent_out) # 512 → 512 (対数標準偏差)
                return mu, logsig
            case 2 | 3 | 4:
                # version 2-4: 活性化後の特徴ベクトルを直接返す
                return self.actv(phi)  # 1024次元
            case _:
                raise ValueError(f'Unexpected version {self.version}')

    def train(self, mode=True):
        # 通常のtrainモード設定
        super().train(mode)
        # BatchNormフリーズが有効な場合
        if self._freeze_bn:
            # すべてのBatchNorm層を評価モードに設定（統計量を更新しない）
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()  # running_mean/varを固定
                    # 勾配計算は無効化しない（パラメータは更新可能のまま）
                    # module.requires_grad_(False)  # コメントアウト：効果がないため
        return self

    def reset_running_stats(self):
        """すべてのBatchNorm層の統計量（running_mean/var）をリセット"""
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()

    def freeze_bn(self, value: bool):
        """BatchNormのフリーズ設定を変更"""
        self._freeze_bn = value
        # 現在のモードを再適用（freeze設定を反映）
        return self.train(self.training)

class AuxNet(nn.Module):
    """補助タスク用のネットワーク
    
    メインタスク以外の補助的な予測（例：次の牌の予測、点数予測など）を行う。
    複数の補助タスクを同時に学習する際に使用。
    """
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims  # 各補助タスクの出力次元のリスト
        # 1024次元の特徴から、すべての補助タスクの出力を一度に生成
        self.net = nn.Linear(1024, sum(dims), bias=False)  # バイアスなし

    def forward(self, x):
        # x: (batch, 1024) → 各補助タスクの出力に分割
        # 例：dims=[10, 20, 30]なら、60次元の出力を10, 20, 30に分割
        return self.net(x).split(self.dims, dim=-1)

class DQN(nn.Module):
    """Deep Q-Network（Dueling DQN architecture）
    
    状態価値V(s)とアドバンテージA(s,a)を分離して学習するDueling DQN。
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))の形で行動価値を計算。
    """
    def __init__(self, *, version=1):
        super().__init__()
        self.version = version
        match version:
            case 1:
                # version 1: シンプルな線形層（Brainのversion 1は512次元出力なので注意）
                self.v_head = nn.Linear(512, 1)              # 状態価値V(s)
                self.a_head = nn.Linear(512, ACTION_SPACE)   # アドバンテージA(s,a)、46アクション分
            case 2 | 3:
                # version 2, 3: 隠れ層を持つネットワーク
                hidden_size = 512 if version == 2 else 256  # version 3は小さめ
                # 状態価値ヘッド
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                # アドバンテージヘッド
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                # version 4: V(s)とA(s,a)を単一の層で同時に出力（効率化）
                self.net = nn.Linear(1024, 1 + ACTION_SPACE)  # 1(V) + 46(A) = 47次元
                nn.init.constant_(self.net.bias, 0)  # バイアスを0で初期化

    def forward(self, phi, mask):
        # phi: 特徴ベクトル (batch, feature_dim)
        # mask: 有効なアクションを示すマスク (batch, ACTION_SPACE)
        
        if self.version == 4:
            # version 4: 一度の計算でVとAを取得
            v, a = self.net(phi).split((1, ACTION_SPACE), dim=-1)
        else:
            # その他: 別々のヘッドで計算
            v = self.v_head(phi)   # (batch, 1)
            a = self.a_head(phi)   # (batch, ACTION_SPACE)
        
        # Dueling DQNの計算：Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        # ただし、平均は有効なアクションのみで計算
        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)  # 無効なアクションを0にして合計
        mask_sum = mask.sum(-1, keepdim=True)                   # 有効なアクションの数
        a_mean = a_sum / mask_sum                               # 有効なアクションの平均
        
        # Q値を計算し、無効なアクションには-infを設定
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q

class GRP(nn.Module):
    """Game Result Predictor - ゲーム結果予測用RNN
    
    ゲームの進行状況（局数、本場、供託、各プレイヤーの点数）から
    最終的な順位の確率分布を予測する。
    4人の順位の全順列（4! = 24通り）の確率を出力。
    """
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        # GRU（Gated Recurrent Unit）でゲームの時系列を処理
        # 入力：7次元（GRP_SIZE）、隠れ状態：hidden_size次元
        self.rnn = nn.GRU(input_size=GRP_SIZE, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # 最終的な順位予測用の全結合層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),  # 全層の隠れ状態を使用
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_layers, 24),  # 24通りの順位組み合わせ
        )
        
        # float64（倍精度）を使用（精度が重要なため）
        for mod in self.modules():
            mod.to(torch.float64)

        # 4人の順位の全順列を事前計算（0=1位、1=2位、2=3位、3=4位）
        # 例：[0,1,2,3] = プレイヤー0が1位、プレイヤー1が2位...
        perms = torch.tensor(list(permutations(range(4))))  # 24通りの順列
        perms_t = perms.transpose(0, 1)  # 転置版（効率的な検索用）
        self.register_buffer('perms', perms)     # (24, 4) - 各順列のプレイヤー順位
        self.register_buffer('perms_t', perms_t) # (4, 24) - 各プレイヤーの各順列での順位

    # 入力形式: [grand_kyoku, honba, kyotaku, s[0], s[1], s[2], s[3]]
    # grand_kyoku: 局数（東1局=0, 南4局=7, 西4局=11）
    # honba: 本場数
    # kyotaku: 供託棒の数
    # s[0-3]: 各プレイヤーの現在の点数（s[0]はプレイヤーID 0の点数）
    # 注：東1局開始時の点数は25000点（s値では2.5）
    def forward(self, inputs: List[Tensor]):
        # 可変長シーケンスを処理するための前処理
        # 各入力シーケンスの長さを記録
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.int64)
        # 最大長に合わせてパディング
        inputs = pad_sequence(inputs, batch_first=True)
        # RNN用にパック（計算効率のため）
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        return self.forward_packed(packed_inputs)

    def forward_packed(self, packed_inputs):
        # GRUで時系列を処理（出力は使わず、最終隠れ状態のみ使用）
        _, state = self.rnn(packed_inputs)  # state: (num_layers, batch, hidden_size)
        # 全層の隠れ状態を連結
        state = state.transpose(0, 1).flatten(1)  # (batch, num_layers * hidden_size)
        # 24通りの順位組み合わせに対するロジット
        logits = self.fc(state)  # (batch, 24)
        return logits

    # ロジットから各プレイヤーの各順位になる確率行列を計算
    # (N, 24) -> (N, player, rank_prob)
    def calc_matrix(self, logits: Tensor):
        batch_size = logits.shape[0]
        # ソフトマックスで確率に変換
        probs = logits.softmax(-1)  # (batch, 24)
        # 出力行列：matrix[b, p, r] = バッチbでプレイヤーpが順位rになる確率
        matrix = torch.zeros(batch_size, 4, 4, dtype=probs.dtype)
        
        # 各プレイヤー、各順位について確率を集計
        for player in range(4):
            for rank in range(4):
                # プレイヤーがその順位になる順列のインデックスを取得
                cond = self.perms_t[player] == rank  # (24,) のブール配列
                # 該当する順列の確率を合計
                matrix[:, player, rank] = probs[:, cond].sum(-1)
        return matrix

    # 実際の順位からラベル（順列インデックス）を取得
    # (N, 4) -> (N)
    def get_label(self, rank_by_player: Tensor):
        # rank_by_player[b, p] = バッチbでプレイヤーpの最終順位（0=1位）
        batch_size = rank_by_player.shape[0]
        # バッチサイズ分に拡張して比較準備
        perms = self.perms.expand(batch_size, -1, -1).transpose(0, 1)  # (24, batch, 4)
        # 各順列が実際の結果と一致するかチェック
        mappings = (perms == rank_by_player).all(-1).nonzero()  # 一致する(順列idx, バッチidx)のペア

        # 各バッチに対応する順列インデックスを格納
        labels = torch.zeros(batch_size, dtype=torch.int64, device=mappings.device)
        labels[mappings[:, 1]] = mappings[:, 0]  # バッチidxに対して順列idxを設定
        return labels
