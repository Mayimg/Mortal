"""
GRP (Game Result Prediction) モデルの訓練スクリプト

このスクリプトは、麻雀ゲームの各局の情報（通算局数、本場、供託、各プレイヤーの得点）を
時系列データとして扱い、GRU（Gated Recurrent Unit）を使用して最終的な順位を予測する
ニューラルネットワークモデルを訓練します。

主な機能：
1. 圧縮された麻雀ログファイルから学習データを生成
2. 各局の状態から最終順位（4人の順位の24通りの並び）を予測
3. AdamWオプティマイザを使用した学習
4. TensorBoardによる学習過程の可視化
5. 定期的なモデルの保存と検証

データの特徴：
- 入力: 各局の7次元ベクトル（局数、本場、供託、4人の得点）の時系列
- 出力: 24クラス分類（4人の順位の全順列）
"""

import prelude  # プロジェクト固有の初期化処理（ログ設定など）

import random  # データのシャッフルやサンプリングに使用
import torch  # PyTorchフレームワーク
import logging  # ログ出力用
from os import path  # ファイルパス操作
from glob import glob  # ファイルパターンマッチング
from datetime import datetime  # タイムスタンプ処理
from torch import optim  # 最適化アルゴリズム
from torch.nn import functional as F  # 損失関数などの関数群
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence  # 可変長系列の効率的な処理
from torch.utils.data import DataLoader, IterableDataset  # データローディング
from torch.utils.tensorboard import SummaryWriter  # 学習過程の可視化
from model import GRP  # Game Result Predictionモデル（GRUベース）
from libriichi.dataset import Grp  # Rustで実装された高速なデータローダー
from common import tqdm  # プログレスバー表示
from config import config  # 設定ファイルの読み込み

class GrpFileDatasetsIter(IterableDataset):
    """
    麻雀ログファイルから学習データを生成するIterableDataset
    
    大量のログファイルを効率的に処理するため、ファイルをバッチ単位で読み込み、
    メモリに保持するデータ量を制限しながら学習データを生成する。
    """
    def __init__(
        self,
        file_list,  # 処理するログファイルのパスリスト
        file_batch_size = 50,  # 一度にメモリに読み込むファイル数（メモリ使用量の制御）
        cycle = False  # Trueの場合、ファイルリストを無限に繰り返す（訓練用）
    ):
        super().__init__()  # IterableDatasetの初期化
        self.file_list = file_list  # 処理対象のファイルパスを保存
        self.file_batch_size = file_batch_size  # ファイルバッチサイズを保存
        self.cycle = cycle  # 循環フラグを保存
        self.buffer = []  # 現在のバッチのデータを一時的に保持するバッファ
        self.iterator = None  # イテレータのインスタンス（遅延初期化）

    def build_iter(self):
        """
        データイテレータを構築し、学習データを生成する
        
        処理の流れ：
        1. ファイルリストをシャッフル（エポックごとに順序を変える）
        2. file_batch_size単位でファイルを読み込む
        3. 各バッチ内のデータをランダムな順序で返す
        4. cycleがTrueの場合は無限ループ、Falseの場合は1エポックで終了
        """
        while True:  # 無限ループ（cycleフラグで制御）
            random.shuffle(self.file_list)  # ファイルリストをシャッフル（学習の偏りを防ぐ）
            for start_idx in range(0, len(self.file_list), self.file_batch_size):  # バッチ単位でファイルを処理
                self.populate_buffer(start_idx)  # 現在のバッチのファイルを読み込み、バッファに格納
                buffer_size = len(self.buffer)  # バッファ内のデータ数を取得
                for i in random.sample(range(buffer_size), buffer_size):  # バッファ内のデータをランダムな順序で取得
                    yield self.buffer[i]  # データを1つずつ返す
                self.buffer.clear()  # バッファをクリア（メモリ解放）
            if not self.cycle:  # cycleがFalseの場合、1エポック処理したら終了
                break

    def populate_buffer(self, start_idx):
        """
        指定されたインデックスから始まるファイルバッチを読み込み、バッファに学習データを格納
        
        各ゲームから時系列データを生成：
        - 各局の情報を累積的に含む系列を作成（1局目のみ、1-2局目、1-3局目...）
        - これにより、ゲームの途中経過から最終順位を予測する学習が可能になる
        """
        file_list = self.file_list[start_idx:start_idx + self.file_batch_size]  # 現在のバッチのファイルリストを取得
        data = Grp.load_gz_log_files(file_list)  # Rustで実装された高速ローダーでgzファイルを並列読み込み

        for game in data:  # 各ゲームを処理
            feature = game.take_feature()  # ゲームの特徴量を取得 [局数, 7]の2次元配列
            rank_by_player = game.take_rank_by_player()  # 最終順位を取得（0が1位、3が4位）

            for i in range(feature.shape[0]):  # 各局について処理
                # i+1局目までの情報を含む系列を作成（累積的な時系列データ）
                inputs_seq = torch.as_tensor(feature[:i + 1], dtype=torch.float64)  # float64精度で変換
                self.buffer.append((  # バッファにタプルとして追加
                    inputs_seq,  # 入力系列（1局目からi+1局目までの情報）
                    rank_by_player,  # 正解ラベル（最終順位）
                ))

    def __iter__(self):
        """
        イテレータプロトコルの実装
        初回呼び出し時にイテレータを作成し、以降は同じイテレータを返す
        """
        if self.iterator is None:  # イテレータが未作成の場合
            self.iterator = self.build_iter()  # イテレータを作成
        return self.iterator  # イテレータを返す

def collate(batch):
    """
    DataLoaderのバッチ処理用関数
    可変長の系列データを効率的にバッチ処理するための前処理を行う
    
    処理内容：
    1. バッチ内の各データを分解し、リストに格納
    2. 系列をパディングして同じ長さに揃える
    3. PackedSequenceに変換してRNNの効率的な処理を可能にする
    4. GPUメモリへの事前転送（pin_memory）で高速化
    """
    inputs = []  # 入力系列のリスト
    lengths = []  # 各系列の長さのリスト
    rank_by_players = []  # 各ゲームの最終順位のリスト
    
    for inputs_seq, rank_by_player in batch:  # バッチ内の各データを処理
        inputs.append(inputs_seq)  # 入力系列を追加
        lengths.append(len(inputs_seq))  # 系列の長さを記録（局数）
        rank_by_players.append(rank_by_player)  # 最終順位を追加

    lengths = torch.tensor(lengths)  # 長さのリストをテンソルに変換
    rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=True)  # 順位をint64テンソルに変換、GPUメモリに固定

    padded = pad_sequence(inputs, batch_first=True)  # 系列を最大長に合わせてパディング（短い系列には0を追加）
    packed_inputs = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)  # PackedSequenceに変換（パディング部分を無視して効率的に処理）
    packed_inputs.pin_memory()  # PackedSequenceもGPUメモリに固定（高速転送のため）

    return packed_inputs, rank_by_players  # パックされた入力と順位ラベルを返す

def train():
    """
    GRPモデルの訓練を実行するメイン関数
    
    処理の流れ：
    1. 設定の読み込みとデバイスの初期化
    2. モデルとオプティマイザの作成
    3. 保存されたモデルの読み込み（存在する場合）
    4. データローダーの作成
    5. 訓練ループの実行
    """
    cfg = config['grp']  # GRP用の設定を取得
    batch_size = cfg['control']['batch_size']  # バッチサイズ（通常512）
    save_every = cfg['control']['save_every']  # モデル保存間隔（通常2000ステップ）
    val_steps = cfg['control']['val_steps']  # 検証ステップ数（通常400）

    device = torch.device(cfg['control']['device'])  # 計算デバイス（'cuda:0'など）を設定
    torch.backends.cudnn.benchmark = cfg['control']['enable_cudnn_benchmark']  # cuDNNの最適化を有効化（畳み込みがない場合は影響小）
    if device.type == 'cuda':  # GPUを使用する場合
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')  # GPU名も表示
    else:
        logging.info(f'device: {device}')  # CPUの場合はデバイス名のみ

    grp = GRP(**cfg['network']).to(device)  # GRPモデルを作成し、指定デバイスに転送
    optimizer = optim.AdamW(grp.parameters())  # AdamWオプティマイザを作成（重み減衰付きAdam）

    state_file = cfg['state_file']  # モデルの保存ファイルパス
    if path.exists(state_file):  # 保存されたモデルが存在する場合
        state = torch.load(state_file, weights_only=True, map_location=device)  # 重みのみ読み込み（安全性のため）
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')  # 保存時刻を取得
        logging.info(f'loaded: {timestamp}')  # 保存時刻をログ出力
        grp.load_state_dict(state['model'])  # モデルの重みを復元
        optimizer.load_state_dict(state['optimizer'])  # オプティマイザの状態を復元
        steps = state['steps']  # 訓練ステップ数を復元
    else:
        steps = 0  # 新規訓練の場合は0から開始

    lr = cfg['optim']['lr']  # 学習率を取得（通常1e-5）
    optimizer.param_groups[0]['lr'] = lr  # オプティマイザの学習率を設定（load_state_dictで変わる可能性があるため）

    file_index = cfg['dataset']['file_index']  # ファイルインデックスの保存パス
    train_globs = cfg['dataset']['train_globs']  # 訓練用ファイルのglobパターン（例: 'data/train/**/*.gz'）
    val_globs = cfg['dataset']['val_globs']  # 検証用ファイルのglobパターン
    if path.exists(file_index):  # ファイルインデックスが既に存在する場合
        index = torch.load(file_index, weights_only=True)  # インデックスを読み込み
        train_file_list = index['train_file_list']  # 訓練用ファイルリストを取得
        val_file_list = index['val_file_list']  # 検証用ファイルリストを取得
    else:  # インデックスが存在しない場合は新規作成
        logging.info('building file index...')  # インデックス作成開始をログ出力
        train_file_list = []  # 訓練用ファイルリストを初期化
        val_file_list = []  # 検証用ファイルリストを初期化
        for pat in train_globs:  # 各訓練用globパターンを処理
            train_file_list.extend(glob(pat, recursive=True))  # 再帰的にファイルを検索して追加
        for pat in val_globs:  # 各検証用globパターンを処理
            val_file_list.extend(glob(pat, recursive=True))  # 再帰的にファイルを検索して追加
        train_file_list.sort(reverse=True)  # ファイルリストを逆順ソート（新しいファイルを先に）
        val_file_list.sort(reverse=True)  # ファイルリストを逆順ソート
        torch.save({'train_file_list': train_file_list, 'val_file_list': val_file_list}, file_index)  # インデックスを保存
    writer = SummaryWriter(cfg['control']['tensorboard_dir'])  # TensorBoardライターを作成（学習過程の可視化用）

    train_file_data = GrpFileDatasetsIter(  # 訓練用データセットを作成
        file_list = train_file_list,  # 訓練用ファイルリスト
        file_batch_size = cfg['dataset']['file_batch_size'],  # 一度に読み込むファイル数（メモリ制御）
        cycle = True,  # 無限に繰り返す（訓練用）
    )
    train_data_loader = iter(DataLoader(  # 訓練用DataLoaderを作成し、イテレータ化
        dataset = train_file_data,  # 訓練用データセット
        batch_size = batch_size,  # バッチサイズ（通常512）
        drop_last = True,  # 最後の不完全なバッチを削除（安定した学習のため）
        num_workers = 1,  # データ読み込み用のワーカープロセス数
        collate_fn = collate,  # バッチ処理用の関数（可変長系列の処理）
    ))

    val_file_data = GrpFileDatasetsIter(  # 検証用データセットを作成
        file_list = val_file_list,  # 検証用ファイルリスト
        file_batch_size = cfg['dataset']['file_batch_size'],  # 一度に読み込むファイル数
        cycle = True,  # 無限に繰り返す（検証用）
    )
    val_data_loader = iter(DataLoader(  # 検証用DataLoaderを作成し、イテレータ化
        dataset = val_file_data,  # 検証用データセット
        batch_size = batch_size,  # バッチサイズ（訓練と同じ）
        drop_last = True,  # 最後の不完全なバッチを削除
        num_workers = 1,  # データ読み込み用のワーカープロセス数
        collate_fn = collate,  # バッチ処理用の関数
    ))

    stats = {  # 統計情報を保持する辞書
        'train_loss': 0,  # 訓練損失の累積値
        'train_acc': 0,  # 訓練精度の累積値
        'val_loss': 0,  # 検証損失の累積値
        'val_acc': 0,  # 検証精度の累積値
    }
    logging.info(f'train file list size: {len(train_file_list):,}')  # 訓練用ファイル数をログ出力（カンマ区切り）
    logging.info(f'val file list size: {len(val_file_list):,}')  # 検証用ファイル数をログ出力

    # 進捗の推定：各ファイルに平均10ゲームが含まれると仮定
    approx_percent = steps * batch_size / (len(train_file_list) * 10) * 100
    logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')  # 現在の進捗率を表示

    pb = tqdm(total=save_every, desc='TRAIN')  # 訓練用プログレスバーを作成（save_everyステップで一周）
    for inputs, rank_by_players in train_data_loader:  # 訓練データをバッチごとに取得
        inputs = inputs.to(dtype=torch.float64, device=device)  # 入力データをGPUに転送（float64精度）
        rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)  # 順位ラベルをGPUに転送

        logits = grp.forward_packed(inputs)  # モデルの順伝播（PackedSequenceを入力）
        labels = grp.get_label(rank_by_players)  # 順位情報を24クラスのラベルに変換
        loss = F.cross_entropy(logits, labels)  # クロスエントロピー損失を計算

        optimizer.zero_grad(set_to_none=True)  # 勾配をゼロにリセット（メモリ効率のためNoneに設定）
        loss.backward()  # 誤差逆伝播で勾配を計算
        optimizer.step()  # パラメータを更新

        with torch.inference_mode():  # 推論モード（勾配計算を無効化して高速化）
            stats['train_loss'] += loss  # 損失を累積
            stats['train_acc'] += (logits.argmax(-1) == labels).to(torch.float64).mean()  # 精度を累積（予測と正解が一致する割合）

        steps += 1  # ステップ数をインクリメント
        pb.update(1)  # プログレスバーを更新

        if steps % save_every == 0:  # 保存間隔に達した場合
            pb.close()  # 訓練用プログレスバーを閉じる

            with torch.inference_mode():  # 推論モードで検証を実行（勾配計算を無効化）
                grp.eval()  # モデルを評価モードに設定（DropoutやBatchNormの動作を変更）
                pb = tqdm(total=val_steps, desc='VAL')  # 検証用プログレスバーを作成
                for idx, (inputs, rank_by_players) in enumerate(val_data_loader):  # 検証データを取得
                    if idx == val_steps:  # 指定された検証ステップ数に達したら終了
                        break
                    inputs = inputs.to(dtype=torch.float64, device=device)  # 入力データをGPUに転送
                    rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)  # 順位ラベルをGPUに転送

                    logits = grp.forward_packed(inputs)  # モデルの順伝播（検証データ）
                    labels = grp.get_label(rank_by_players)  # 順位情報をラベルに変換
                    loss = F.cross_entropy(logits, labels)  # 損失を計算

                    stats['val_loss'] += loss  # 検証損失を累積
                    stats['val_acc'] += (logits.argmax(-1) == labels).to(torch.float64).mean()  # 検証精度を累積
                    pb.update(1)  # プログレスバーを更新
                pb.close()  # 検証用プログレスバーを閉じる
                grp.train()  # モデルを訓練モードに戻す

            # TensorBoardに統計情報を記録
            writer.add_scalars('loss', {  # 損失のグラフを記録
                'train': stats['train_loss'] / save_every,  # 訓練損失の平均値
                'val': stats['val_loss'] / val_steps,  # 検証損失の平均値
            }, steps)
            writer.add_scalars('acc', {  # 精度のグラフを記録
                'train': stats['train_acc'] / save_every,  # 訓練精度の平均値
                'val': stats['val_acc'] / val_steps,  # 検証精度の平均値
            }, steps)
            writer.add_scalar('lr', lr, steps)  # 学習率を記録
            writer.flush()  # TensorBoardへの書き込みを強制フラッシュ

            for k in stats:  # 統計情報をリセット
                stats[k] = 0
            approx_percent = steps * batch_size / (len(train_file_list) * 10) * 100  # 進捗率を再計算
            logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')  # 進捗をログ出力

            state = {  # モデルの状態を保存する辞書を作成
                'model': grp.state_dict(),  # モデルの重み
                'optimizer': optimizer.state_dict(),  # オプティマイザの状態（モーメンタムなど）
                'steps': steps,  # 現在のステップ数
                'timestamp': datetime.now().timestamp(),  # 保存時刻
            }
            torch.save(state, state_file)  # モデルをファイルに保存
            pb = tqdm(total=save_every, desc='TRAIN')  # 新しい訓練用プログレスバーを作成
    pb.close()  # 最後のプログレスバーを閉じる

if __name__ == '__main__':  # スクリプトが直接実行された場合のみ実行
    try:
        train()  # 訓練関数を実行
    except KeyboardInterrupt:  # Ctrl+Cで中断された場合
        pass  # エラーメッセージを表示せずに正常終了
