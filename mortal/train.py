"""mortal/train.py - Mortal麻雀AIのメイン訓練スクリプト

概要:
  このファイルはMortal麻雀AIの深層強化学習訓練を実行するメインスクリプトです。
  以下の主要な機能を持ちます：
  
  1. DQN (Deep Q-Network) とCQL (Conservative Q-Learning) を使用した強化学習
  2. ResNetベースのニューラルネットワーク ("Brain") による特徴抽出
  3. 補助ネットワークによる順位予測タスク
  4. オンライン/オフラインモードでの訓練
  5. 定期的なモデル評価とチェックポイント保存
  6. TensorBoardによる訓練進捗の可視化
  
  訓練プロセス:
  - 麻雀の対局ログデータからバッチ単位で学習
  - Q値の推定と実際の報酬との誤差を最小化
  - CQLによる過大評価の抑制（オフラインモード）
  - 定期的なテストプレイによる性能評価
"""

def train():
    # preludeモジュールをインポート（ログ設定などの初期化処理を含む）
    import prelude

    # 標準ライブラリのインポート
    import logging  # ログ出力用
    import sys      # システム関連の操作（終了処理など）
    import os       # OS関連の操作（環境変数、ファイルパス操作）
    import gc       # ガベージコレクション（メモリ管理）
    import gzip     # gzip圧縮されたログファイルの読み込み用
    import json     # JSONフォーマットのデータ解析用
    import shutil   # ファイルコピー操作用（ベストモデル保存時）
    import random   # ランダムシャッフル用（データローダー）
    import torch    # PyTorchフレームワーク（深層学習の中核）
    from os import path                               # ファイルパス操作のヘルパー関数
    from glob import glob                             # ファイルパターンマッチング（データセット検索）
    from datetime import datetime                     # タイムスタンプ処理
    from itertools import chain                       # 複数のイテレータを連結
    from torch import optim, nn                       # 最適化アルゴリズムとニューラルネットワークモジュール
    from torch.amp import GradScaler                  # 自動混合精度訓練用のグラディエントスケーラー
    from torch.nn.utils import clip_grad_norm_        # 勾配クリッピング（勾配爆発防止）
    from torch.utils.data import DataLoader           # バッチデータローディング
    from torch.utils.tensorboard import SummaryWriter # TensorBoard統合（訓練の可視化）
    # プロジェクト固有モジュールのインポート
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm  # 共通ユーティリティ関数
    from player import TestPlayer                     # テストプレイ用のプレイヤークラス
    from dataloader import FileDatasetsIter, worker_init_fn  # データローディング関連
    from lr_scheduler import LinearWarmUpCosineAnnealingLR   # 学習率スケジューラー（ウォームアップ付きコサイン）
    from model import Brain, DQN, AuxNet              # モデル定義：Brain（特徴抽出）、DQN（Q値推定）、AuxNet（補助タスク）
    from libriichi.consts import obs_shape            # 麻雀の観測空間の形状定義
    from config import config                         # 設定ファイル（訓練パラメータなど）

    # 設定ファイルからモデルバージョンを取得
    version = config['control']['version']

    # 訓練制御パラメータの取得
    online = config['control']['online']                # オンラインモードフラグ（True: リアルタイム学習、False: オフライン学習）
    batch_size = config['control']['batch_size']        # バッチサイズ（一度に処理するサンプル数）
    opt_step_every = config['control']['opt_step_every']    # 最適化ステップの実行間隔
    save_every = config['control']['save_every']        # モデル保存間隔（ステップ数）
    test_every = config['control']['test_every']        # テストプレイ実行間隔（ステップ数）
    submit_every = config['control']['submit_every']    # オンラインモードでパラメータ送信間隔
    test_games = config['test_play']['games']           # テストプレイのゲーム数
    min_q_weight = config['cql']['min_q_weight']        # CQLの重み（Q値の過大評価を抑制）
    next_rank_weight = config['aux']['next_rank_weight']    # 補助タスク（順位予測）の重み
    assert save_every % opt_step_every == 0             # 保存間隔は最適化ステップ間隔の倍数であることを確認
    assert test_every % save_every == 0                 # テスト間隔は保存間隔の倍数であることを確認

    # PyTorchデバイス設定
    device = torch.device(config['control']['device'])  # 計算デバイス（GPU/CPU）の指定
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']  # cuDNNのベンチマークモード（高速化）
    enable_amp = config['control']['enable_amp']        # 自動混合精度訓練の有効化フラグ
    enable_compile = config['control']['enable_compile'] # PyTorch 2.0のコンパイル機能の有効化

    # 環境設定
    pts = config['env']['pts']                          # 麻雀の点数設定
    gamma = config['env']['gamma']                      # 割引率（強化学習の将来報酬の割引）
    
    # データセット設定
    file_batch_size = config['dataset']['file_batch_size']  # ファイル単位のバッチサイズ
    reserve_ratio = config['dataset']['reserve_ratio']       # データ予約率
    num_workers = config['dataset']['num_workers']          # データローダーのワーカープロセス数
    num_epochs = config['dataset']['num_epochs']            # エポック数（データセットの反復回数）
    enable_augmentation = config['dataset']['enable_augmentation']  # データ拡張の有効化
    augmented_first = config['dataset']['augmented_first']   # 拡張データを最初に使用するか
    
    # 最適化設定
    eps = config['optim']['eps']                        # Adamオプティマイザのエプシロン（数値安定性）
    betas = config['optim']['betas']                    # Adamのベータパラメータ（モーメンタム係数）
    weight_decay = config['optim']['weight_decay']      # 重み減衰（L2正則化）
    max_grad_norm = config['optim']['max_grad_norm']    # 勾配クリッピングの最大ノルム

    # モデルの初期化
    mortal = Brain(version=version, **config['resnet']).to(device)  # メインの特徴抽出ネットワーク（ResNetベース）
    dqn = DQN(version=version).to(device)               # Deep Q-Network（Q値推定用）
    aux_net = AuxNet((4,)).to(device)                   # 補助ネットワーク（4人麻雀の順位予測）
    all_models = (mortal, dqn, aux_net)                 # すべてのモデルをタプルにまとめる
    if enable_compile:                                  # PyTorch 2.0のコンパイル機能が有効な場合
        for m in all_models:
            m.compile()                                 # 各モデルをコンパイル（高速化）

    # モデル情報のログ出力
    logging.info(f'version: {version}')                 # モデルバージョン
    logging.info(f'obs shape: {obs_shape(version)}')    # 観測空間の形状（入力テンソルのサイズ）
    logging.info(f'mortal params: {parameter_count(mortal):,}')  # Brainモデルのパラメータ数
    logging.info(f'dqn params: {parameter_count(dqn):,}')        # DQNモデルのパラメータ数
    logging.info(f'aux params: {parameter_count(aux_net):,}')    # 補助ネットワークのパラメータ数

    # BatchNorm層の凍結設定（設定に応じてBatchNormのパラメータを固定）
    mortal.freeze_bn(config['freeze_bn']['mortal'])

    # パラメータグループの設定（重み減衰を適用するパラメータとしないパラメータを分離）
    decay_params = []       # 重み減衰を適用するパラメータのリスト
    no_decay_params = []    # 重み減衰を適用しないパラメータのリスト
    for model in all_models:                            # すべてのモデルに対して
        params_dict = {}                                # パラメータ名とパラメータの辞書
        to_decay = set()                                # 重み減衰を適用するパラメータ名のセット
        for mod_name, mod in model.named_modules():     # モデル内の各モジュールを走査
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):  # モジュールのパラメータを取得
                params_dict[name] = param               # パラメータ辞書に追加
                # Linear層やConv1d層のweightパラメータには重み減衰を適用
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))          # 重み減衰対象パラメータを追加
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))  # 非対象パラメータを追加
    # オプティマイザ用のパラメータグループを作成
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},  # 重み減衰ありグループ
        {'params': no_decay_params},                             # 重み減衰なしグループ
    ]
    # 最適化関連の初期化
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)  # AdamWオプティマイザ（lr=1はスケジューラが管理）
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])  # 学習率スケジューラ
    scaler = GradScaler(device.type, enabled=enable_amp)  # 自動混合精度用の勾配スケーラー
    test_player = TestPlayer()                          # テストプレイ用のプレイヤーインスタンス
    # 最高性能の記録用辞書
    best_perf = {
        'avg_rank': 4.,     # 平均順位（最悪値から開始）
        'avg_pt': -135.,    # 平均点数（最悪値から開始）
    }

    # 訓練ステップ数の初期化
    steps = 0
    # チェックポイントファイルのパスを取得
    state_file = config['control']['state_file']        # 通常のチェックポイント
    best_state_file = config['control']['best_state_file']  # 最良性能のチェックポイント
    # 既存のチェックポイントがある場合は読み込み
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)  # チェックポイントをロード
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')  # タイムスタンプをフォーマット
        logging.info(f'loaded: {timestamp}')            # ロード時刻をログ出力
        mortal.load_state_dict(state['mortal'])         # Brainモデルの重みをロード
        dqn.load_state_dict(state['current_dqn'])       # DQNモデルの重みをロード
        aux_net.load_state_dict(state['aux_net'])       # 補助ネットワークの重みをロード
        # オフラインモード、または以前もオンラインモードだった場合はオプティマイザーの状態もロード
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])   # オプティマイザーの状態をロード
            scheduler.load_state_dict(state['scheduler'])   # 学習率スケジューラーの状態をロード
        scaler.load_state_dict(state['scaler'])         # 勾配スケーラーの状態をロード
        best_perf = state['best_perf']                  # 最高性能の記録をロード
        steps = state['steps']                          # 訓練ステップ数をロード

    # 勾配の初期化（メモリ効率のためNoneに設定）
    optimizer.zero_grad(set_to_none=True)
    # 損失関数の初期化
    mse = nn.MSELoss()          # 平均二乗誤差（DQNのQ値予測用）
    ce = nn.CrossEntropyLoss()  # クロスエントロピー損失（順位予測用）

    # デバイス情報のログ出力
    if device.type == 'cuda':                           # GPUを使用する場合
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')  # GPU名も表示
    else:
        logging.info(f'device: {device}')               # CPUの場合

    # オンラインモードの場合、初回パラメータ送信
    if online:
        submit_param(mortal, dqn, is_idle=True)         # パラメータをサーバーに送信（アイドル状態）
        logging.info('param has been submitted')

    # TensorBoardのライター初期化
    writer = SummaryWriter(config['control']['tensorboard_dir'])
    # 訓練統計情報の辞書
    stats = {
        'dqn_loss': 0,          # DQNの損失値累積
        'cql_loss': 0,          # CQLの損失値累積（オフラインモードのみ）
        'next_rank_loss': 0,    # 順位予測の損失値累積ｎ
    }
    # Q値の記録用テンソル（ヒストグラム作成用）
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)        # 予測Q値
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32) # 目標Q値
    idx = 0  # 現在のバッチインデックス（save_everyまでカウント）

    def train_epoch():
        """エポック単位の訓練を実行する関数
        
        データセットの読み込み、バッチ単位の訓練、定期的な保存と評価を行う
        """
        nonlocal steps  # 外部変数の訓練ステップ数を参照
        nonlocal idx    # 外部変数のバッチインデックスを参照

        # プレイヤー名のリスト初期化
        player_names = []
        if online:  # オンラインモードの場合
            player_names = ['trainee']                  # 訓練用プレイヤー名を設定
            dirname = drain()                           # 新しいデータを取得（drain関数はデータディレクトリを返す）
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))  # ディレクトリ内のファイルパスを作成
        else:  # オフラインモードの場合
            player_names_set = set()                    # プレイヤー名のセット（重複除去）
            for filename in config['dataset']['player_names_files']:  # プレイヤー名ファイルを走査
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))  # ファイルからプレイヤー名を読み込み
            player_names = list(player_names_set)       # セットをリストに変換
            logging.info(f'loaded {len(player_names):,} players')  # ロードしたプレイヤー数をログ出力

            # ファイルインデックスの読み込みまたは作成
            file_index = config['dataset']['file_index']  # ファイルインデックスのパス
            if path.exists(file_index):                 # 既存のインデックスがある場合
                index = torch.load(file_index, weights_only=True)  # インデックスをロード
                file_list = index['file_list']          # ファイルリストを取得
            else:                                       # インデックスがない場合は新規作成
                logging.info('building file index...')
                file_list = []
                # 指定されたグロブパターンにマッチするファイルを収集
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))  # 再帰的にファイルを検索
                # プレイヤー名でフィルタリング
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):  # 進捗バー付きでファイルを処理
                        with gzip.open(filename, 'rt') as f:       # gzip圧縮ファイルをテキストモードで開く
                            start = json.loads(next(f))             # 最初の行（ゲーム開始情報）をパース
                            # 指定プレイヤーが含まれている場合のみファイルを保持
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)            # 新しいファイルが先頭に来るように逆順ソート
                torch.save({'file_list': file_list}, file_index)  # インデックスを保存
        logging.info(f'file list size: {len(file_list):,}')  # ファイルリストのサイズをログ出力

        # 次のテストプレイまでのステップ数を計算
        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')  # 現在のステップ数と次のテストまでの残り

        # マルチワーカーの場合はファイルリストをシャッフル（負荷分散）
        if num_workers > 1:
            random.shuffle(file_list)
        # ファイルデータセットイテレータの作成
        file_data = FileDatasetsIter(
            version = version,                          # モデルバージョン
            file_list = file_list,                      # 訓練データファイルのリスト
            pts = pts,                                  # 点数設定
            file_batch_size = file_batch_size,          # ファイル単位のバッチサイズ
            reserve_ratio = reserve_ratio,              # データ予約率
            player_names = player_names,                # 対象プレイヤー名
            num_epochs = num_epochs,                    # エポック数
            enable_augmentation = enable_augmentation,  # データ拡張の有効化
            augmented_first = augmented_first,          # 拡張データを最初に使用
        )
        # PyTorchデータローダーの作成
        data_loader = iter(DataLoader(
            dataset = file_data,                        # データセット
            batch_size = batch_size,                    # バッチサイズ
            drop_last = False,                          # 最後の不完全バッチを保持
            num_workers = num_workers,                  # ワーカープロセス数
            pin_memory = True,                          # GPU転送の高速化
            worker_init_fn = worker_init_fn,            # ワーカー初期化関数
        ))

        # 不完全バッチ用のバッファ（バッチサイズに満たないデータを一時保存）
        remaining_obs = []              # 観測データ
        remaining_actions = []          # 行動データ
        remaining_masks = []            # 有効行動マスク
        remaining_steps_to_done = []    # 終了までのステップ数
        remaining_kyoku_rewards = []    # 局の報酬
        remaining_player_ranks = []     # プレイヤーの順位
        remaining_bs = 0                # 保留中のサンプル数
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)  # 進捗バーの初期化

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            """バッチ単位の訓練を実行する内部関数
            
            Args:
                obs: 観測データ（ゲーム状態）
                actions: 実際に取られた行動
                masks: 有効行動のマスク
                steps_to_done: ゲーム終了までのステップ数
                kyoku_rewards: 局の報酬（最終的な点数差）
                player_ranks: プレイヤーの最終順位
            """
            nonlocal steps  # 外部変数：総訓練ステップ数
            nonlocal idx    # 外部変数：現在のバッチインデックス
            nonlocal pb     # 外部変数：進捗バー

            # データを適切なデバイスとデータ型に転送
            obs = obs.to(dtype=torch.float32, device=device)            # 観測データをfloat32でGPU/CPUへ
            actions = actions.to(dtype=torch.int64, device=device)      # 行動をint64でGPU/CPUへ
            masks = masks.to(dtype=torch.bool, device=device)           # マスクをboolでGPU/CPUへ
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)  # ステップ数をint64でGPU/CPUへ
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)  # 報酬をfloat64でGPU/CPUへ（高精度）
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)     # 順位をint64でGPU/CPUへ
            assert masks[range(batch_size), actions].all()  # すべての選択された行動が有効であることを確認

            # モンテカルロ推定による目標Q値の計算
            q_target_mc = gamma ** steps_to_done * kyoku_rewards  # 割引率^ステップ数 * 最終報酬
            q_target_mc = q_target_mc.to(torch.float32)           # float32に変換（計算効率のため）

            # 自動混合精度コンテキスト内で順伝播計算
            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)                               # Brainモデルで特徴抽出
                q_out = dqn(phi, masks)                         # DQNでQ値を推定（各行動のQ値）
                q = q_out[range(batch_size), actions]           # 実際に取られた行動のQ値を抽出
                dqn_loss = 0.5 * mse(q, q_target_mc)            # DQN損失：予測Q値と目標Q値のMSE
                cql_loss = 0                                    # CQL損失の初期化
                if not online:                                  # オフラインモードの場合のみ
                    # CQL損失：すべてのQ値の対数和の期待値 - 実際に取られた行動のQ値の期待値
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)               # 補助ネットワークで順位予測
                next_rank_loss = ce(next_rank_logits, player_ranks)  # 順位予測のクロスエントロピー損失

                # 総損失の計算（各損失に重みを付けて合計）
                loss = sum((
                    dqn_loss,                                   # DQNの基本損失
                    cql_loss * min_q_weight,                    # CQL損失（重み付き）
                    next_rank_loss * next_rank_weight,          # 順位予測損失（重み付き）
                ))
            # 勾配累積のために損失をopt_step_everyで割ってバックプロパゲーション
            scaler.scale(loss / opt_step_every).backward()

            # 統計情報の記録（勾配計算なし）
            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss                  # DQN損失を累積
                if not online:
                    stats['cql_loss'] += cql_loss               # CQL損失を累積（オフラインのみ）
                stats['next_rank_loss'] += next_rank_loss      # 順位予測損失を累積
                all_q[idx] = q                                 # 予測Q値を保存
                all_q_target[idx] = q_target_mc                # 目標Q値を保存

            steps += 1      # 総訓練ステップ数をインクリメント
            idx += 1        # バッチインデックスをインクリメント
            # 指定ステップ毎にオプティマイザーを更新
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:                           # 勾配クリッピングが有効な場合
                    scaler.unscale_(optimizer)                  # 勾配のスケールを元に戻す
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)  # すべてのパラメータを取得
                    clip_grad_norm_(params, max_grad_norm)      # 勾配のノルムをクリッピング
                scaler.step(optimizer)                          # オプティマイザーのステップ実行
                scaler.update()                                 # スケーラーの更新
                optimizer.zero_grad(set_to_none=True)           # 勾配をリセット
            scheduler.step()    # 学習率スケジューラーのステップ
            pb.update(1)        # 進捗バーを更新

            # オンラインモードで定期的にパラメータをサーバーに送信
            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)        # パラメータ送信（アクティブ状態）
                logging.info('param has been submitted')

            # 定期的な保存と統計記録
            if steps % save_every == 0:
                pb.close()      # 進捗バーを閉じる

                # TensorBoardのイベントサイズ削減のためダウンサンプリング
                all_q_1d = all_q.cpu().numpy().flatten()[::128]         # Q値を128個おきにサンプリング
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]  # 目標Q値を128個おきにサンプリング

                # TensorBoardに各種メトリクスを記録
                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)  # 平均DQN損失
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)  # 平均CQL損失
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)  # 平均順位予測損失
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)  # 現在の学習率
                writer.add_histogram('q_predicted', all_q_1d, steps)    # 予測Q値の分布
                writer.add_histogram('q_target', all_q_target_1d, steps)  # 目標Q値の分布
                writer.flush()  # TensorBoardに書き込み

                # 統計をリセット
                for k in stats:
                    stats[k] = 0
                idx = 0         # バッチインデックスをリセット

                # 次のテストプレイまでのステップ数を再計算
                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                # チェックポイントの作成
                state = {
                    'mortal': mortal.state_dict(),              # Brainモデルの状態
                    'current_dqn': dqn.state_dict(),            # DQNモデルの状態
                    'aux_net': aux_net.state_dict(),            # 補助ネットワークの状態
                    'optimizer': optimizer.state_dict(),        # オプティマイザーの状態
                    'scheduler': scheduler.state_dict(),        # 学習率スケジューラーの状態
                    'scaler': scaler.state_dict(),              # 勾配スケーラーの状態
                    'steps': steps,                             # 現在の訓練ステップ数
                    'timestamp': datetime.now().timestamp(),    # 現在のタイムスタンプ
                    'best_perf': best_perf,                     # 最高性能の記録
                    'config': config,                           # 設定情報
                }
                torch.save(state, state_file)                   # チェックポイントを保存

                # オンラインモードで保存時にパラメータ送信がまだの場合
                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)    # パラメータを送信
                    logging.info('param has been submitted')

                # 定期的なテストプレイによる性能評価
                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)  # 1/4のゲーム数でテスト
                    mortal.train()      # モデルを訓練モードに戻す
                    dqn.train()         # DQNを訓練モードに戻す

                    # 平均点数を計算（表示用、実際の訓練では使用しない）
                    avg_pt = stat.avg_pt([90, 45, 0, -135])     # 順位点数: 1位90、1位45、3位0、4位-135
                    # 最高性能の更新判定（平均点数が以上かつ平均順位が以下）
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()            # 過去の最高性能を保存
                        best_perf['avg_pt'] = avg_pt           # 新しい最高点数を記録
                        best_perf['avg_rank'] = stat.avg_rank  # 新しい最高順位を記録

                    # テスト結果のログ出力
                    logging.info(f'avg rank: {stat.avg_rank:.6}')   # 平均順位
                    logging.info(f'avg pt: {avg_pt:.6}')           # 平均点数
                    
                    # TensorBoardに詳細な統計を記録
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)    # 平均順位
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)               # 平均点数
                    writer.add_scalars('test_play/ranking', {     # 順位率の分布
                        '1st': stat.rank_1_rate,                  # 1位率
                        '2nd': stat.rank_2_rate,                  # 2位率
                        '3rd': stat.rank_3_rate,                  # 3位率
                        '4th': stat.rank_4_rate,                  # 4位率
                    }, steps)
                    writer.add_scalars('test_play/behavior', {    # プレイ行動の統計
                        'agari': stat.agari_rate,                 # 和了率
                        'houjuu': stat.houjuu_rate,               # 放銃率
                        'fuuro': stat.fuuro_rate,                 # 副露率（鳴き・ポン・チー）
                        'riichi': stat.riichi_rate,               # リーチ率
                    }, steps)
                    writer.add_scalars('test_play/agari_point', { # 和了時の平均点数
                        'overall': stat.avg_point_per_agari,      # 全体の平均和了点
                        'riichi': stat.avg_point_per_riichi_agari,   # リーチ和了の平均点
                        'fuuro': stat.avg_point_per_fuuro_agari,     # 副露和了の平均点
                        'dama': stat.avg_point_per_dama_agari,       # ダマ和了の平均点
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)  # 平均放銃点
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)  # 1局あたりの平均点数
                    writer.add_scalars('test_play/key_step', {    # 重要アクションの巡目
                        'agari_jun': stat.avg_agari_jun,         # 平均和了巡目
                        'houjuu_jun': stat.avg_houjuu_jun,       # 平均放銃巡目
                        'riichi_jun': stat.avg_riichi_jun,       # 平均リーチ巡目
                    }, steps)
                    writer.add_scalars('test_play/riichi', {      # リーチ関連の統計
                        'agari_after_riichi': stat.agari_rate_after_riichi,     # リーチ後の和了率
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,   # リーチ後の放銃率
                        'chasing_riichi': stat.chasing_riichi_rate,             # 追っかけリーチ率
                        'riichi_chased': stat.riichi_chased_rate,               # 追いかけられリーチ率
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)  # リーチ和了の平均点
                    writer.add_scalars('test_play/fuuro', {       # 副露関連の統計
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,       # 副露後の和了率
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,     # 副露後の放銃率
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)    # 平均副露回数
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)  # 副露和了の平均点
                    writer.flush()      # TensorBoardに書き込み

                    # 最高性能を更新した場合の処理
                    if better:
                        torch.save(state, state_file)           # チェックポイントを保存
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '     # 点数の改善
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '  # 順位の改善
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)  # 最良モデルファイルにコピー
                    if online:
                        # バグ: オンラインモードではここでプロセスがフリーズする未知の問題がある
                        # これがmain関数がオンラインモードでサブプロセスを生成して訓練する理由
                        sys.exit(0)                             # プロセスを終了
                pb = tqdm(total=save_every, desc='TRAIN')      # 新しい進捗バーを作成

        # データローダーからバッチデータを取得して訓練
        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]                           # バッチサイズを取得
            if bs != batch_size:                        # 不完全バッチの場合
                # バッファに追加
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs                      # 保留中のサンプル数を更新
                continue                                # 次のバッチへ
            # 完全なバッチの場合は訓練実行
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        # 保留していた不完全バッチを処理
        remaining_batches = remaining_bs // batch_size  # 作成可能な完全バッチ数
        if remaining_batches > 0:
            # すべての保留データを結合
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            # バッチサイズごとに分割して訓練
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],                     # 観測データのスライス
                    actions[start:end],                 # 行動データのスライス
                    masks[start:end],                   # マスクデータのスライス
                    steps_to_done[start:end],           # ステップ数データのスライス
                    kyoku_rewards[start:end],           # 報酬データのスライス
                    player_ranks[start:end],            # 順位データのスライス
                )
                start = end
                end += batch_size
        pb.close()      # 進捗バーを閉じる

        # オンラインモードの場合、エポック終了時にアイドル状態を送信
        if online:
            submit_param(mortal, dqn, is_idle=True)     # アイドル状態でパラメータ送信
            logging.info('param has been submitted')

    # メインの訓練ループ
    while True:
        train_epoch()   # 1エポック分の訓練を実行
        gc.collect()    # ガベージコレクションを実行（メモリ解放）
        # torch.cuda.empty_cache()    # GPUメモリキャッシュのクリア（コメントアウト）
        # torch.cuda.synchronize()    # GPU同期（コメントアウト）
        if not online:
            # オフラインモードでは1エポックのみ実行（制御しやすくするため）
            break

def main():
    """メイン関数 - オンラインモードでのプロセス管理を担当
    
    オンラインモードではサブプロセスを生成して訓練を実行し、
    プロセスが終了したら再起動する。これはオンラインモードでの
    未知のフリーズ問題を回避するための回避策。
    """
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # 環境変数キー（手動で設定しないこと）
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    # オフラインモード、または既にサブプロセス内の場合は直接訓練を実行
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    # サブプロセスとして実行するコマンド
    cmd = (sys.executable, __file__)    # Pythonインタプリタと現在のスクリプト
    env = {
        is_sub_proc_key: '1',           # サブプロセスであることを示すフラグ
        **os.environ.copy(),            # 現在の環境変数をコピー
    }
    # プロセスを繰り返し生成・実行
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,          # 標準入力を継承
            stdout = sys.stdout,        # 標準出力を継承
            stderr = sys.stderr,        # 標準エラー出力を継承
            env = env,                  # 環境変数を設定
        )
        if (code := child.wait()) != 0:    # プロセスの終了を待ち、終了コードを取得
            sys.exit(code)              # エラー終了の場合は親プロセスも終了
        time.sleep(3)                   # 3秒待機してから再起動

if __name__ == '__main__':
    # スクリプトが直接実行された場合のエントリーポイント
    try:
        main()                          # メイン関数を実行
    except KeyboardInterrupt:
        pass                            # Ctrl+Cによる中断を静かに処理
