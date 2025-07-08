"""
mortal/common.py - 分散学習システムのための共通ユーティリティモジュール

このモジュールは、Mortal（麻雀AI）の分散学習システムで使用される共通機能を提供します。
主な機能：
1. PyTorchモデルのパラメータ管理（パラメータ数のカウント、勾配の操作）
2. ネットワーク通信（ソケット通信によるモデルパラメータの送受信）
3. リモートパラメータサーバーとの通信（モデルの更新、データの排出）
4. ユーティリティ関数（文字列処理、プログレスバー設定）

このモジュールは、クライアント（自己対戦ワーカー）とパラメータサーバー間の
通信を効率的に処理し、分散学習を可能にします。
"""

import torch  # PyTorchライブラリ - ディープラーニングフレームワーク、モデルパラメータの保存・読み込みに使用
import socket  # ソケット通信ライブラリ - パラメータサーバーとのTCP/IP通信に使用
import struct  # バイナリデータのパック/アンパック - メッセージサイズの送受信に使用
import time  # 時間関連の機能 - リトライ処理での待機時間に使用
from typing import *  # 型ヒント用 - 関数の引数・戻り値の型を明示
from io import BytesIO  # バイト列のI/Oストリーム - PyTorchオブジェクトのシリアライゼーションに使用
from functools import partial  # 関数の部分適用 - tqdmのデフォルト引数設定に使用
from tqdm.auto import tqdm as orig_tqdm  # プログレスバー表示ライブラリ - 学習の進捗表示に使用
from config import config  # 設定ファイル読み込みモジュール - サーバーアドレスなどの設定値を取得

# tqdmプログレスバーのカスタマイズ設定
# unit='batch': 進捗の単位をバッチとして表示
# dynamic_ncols=True: ターミナルの幅に合わせて自動的にサイズ調整
# ascii=True: ASCII文字のみを使用（Unicode文字を使わない）
tqdm = partial(orig_tqdm, unit='batch', dynamic_ncols=True, ascii=True)

def parameter_count(module):
    """
    PyTorchモジュール内の学習可能なパラメータの総数をカウントする
    
    Args:
        module: PyTorchのnn.Module - パラメータ数をカウントするモデル
    
    Returns:
        int: 学習可能なパラメータの総数
    """
    # module.parameters(): モジュール内のすべてのパラメータを取得
    # p.requires_grad: そのパラメータが勾配計算対象（学習可能）かどうかをチェック
    # p.numel(): パラメータテンソルの要素数を取得（例：100x50の行列なら5000）
    # sum(): すべての学習可能パラメータの要素数を合計
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def filtered_trimmed_lines(lines):
    """
    文字列のリストから空白行を除去し、各行の前後の空白を取り除く
    
    Args:
        lines: 文字列のリスト - 処理対象の行のリスト
    
    Returns:
        filter object: 空でない、トリムされた行のイテレータ
    """
    # map(lambda l: l.strip(), lines): 各行の前後の空白（スペース、タブ、改行）を削除
    # filter(lambda l: l, ...): 空文字列（Falsy値）を除外し、内容のある行のみを残す
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def iter_grads(parameters, take=False):
    """
    モデルパラメータの勾配をイテレートする。オプションで勾配を取得後にゼロクリアできる。
    
    Args:
        parameters: イテラブルなパラメータ - 通常はmodel.parameters()の結果
        take (bool): True の場合、勾配をクローンして返し、元の勾配をゼロにする。
                     False の場合、勾配への参照を返すのみ。
    
    Yields:
        torch.Tensor: 各パラメータの勾配テンソル
                      take=True の場合はクローン、False の場合は参照
    
    用途:
        - 勾配の集約や分析
        - 分散学習での勾配の送信
        - カスタム最適化アルゴリズムの実装
    """
    # すべてのパラメータをループ
    for p in parameters:
        # 勾配が計算されている場合のみ処理（勾配がNoneでない）
        if p.grad is not None:
            if take:
                # take=Trueの場合：勾配を取得して元の勾配をクリア
                # Set to zero instead of None to preserve the layout and make it
                # easier to assign back later
                # 勾配のクローンを作成（元の勾配データのコピー）
                yield p.grad.clone()
                # 元の勾配をゼロで埋める（Noneにしない理由：レイアウトを保持し、後で値を割り当てやすくするため）
                p.grad.zero_()
            else:
                # take=Falseの場合：勾配への参照をそのまま返す
                yield p.grad

def drain():
    """
    パラメータサーバーから処理可能なデータ（ゲームログ）のディレクトリを取得する。
    データがない場合は、データが利用可能になるまで待機する。
    
    Returns:
        str: 処理すべきデータが格納されているディレクトリのパス
    
    動作:
        1. パラメータサーバーに'drain'リクエストを送信
        2. サーバーが処理可能なデータの数とディレクトリを返す
        3. データがない場合（count=0）は5秒待機して再試行
        4. データがある場合はそのディレクトリパスを返す
    
    用途:
        トレーナープロセスがサーバーから学習用データを取得する際に使用
    """
    # 設定ファイルからリモートサーバーのアドレスとポートを取得
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    # データが取得できるまで無限ループ
    while True:
        # withステートメントでソケットを作成（自動的にクローズされる）
        with socket.socket() as conn:
            # リモートサーバーに接続
            conn.connect(remote)
            # 'drain'タイプのメッセージを送信（データ取得リクエスト）
            send_msg(conn, {'type': 'drain'})
            # サーバーからの応答を受信
            msg = recv_msg(conn)
        # データがない場合（count=0）
        if msg['count'] == 0:
            # 5秒待機してから再試行
            time.sleep(5)
            continue
        # データがある場合、ディレクトリパスを返す
        return msg['drain_dir']

def submit_param(mortal, dqn, is_idle=False):
    """
    更新されたモデルパラメータをパラメータサーバーに送信する
    
    Args:
        mortal: メインのMortalモデル（麻雀AIの主要モデル）
        dqn: DQN（Deep Q-Network）モデル（強化学習の価値関数）
        is_idle (bool): アイドル状態かどうか。True の場合、モデルが更新されていないことを示す
    
    動作:
        1. 両方のモデルの state_dict（パラメータの辞書）を取得
        2. パラメータサーバーに送信
        3. サーバーはこれらのパラメータを保存し、他のクライアントに配布
    
    用途:
        トレーナーが学習後の新しいモデルパラメータをサーバーに送信する際に使用
    """
    # 設定ファイルからリモートサーバーのアドレスとポートを取得
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    # withステートメントでソケットを作成（自動的にクローズされる）
    with socket.socket() as conn:
        # リモートサーバーに接続
        conn.connect(remote)
        # パラメータ送信メッセージを構築して送信
        send_msg(conn, {
            'type': 'submit_param',  # メッセージタイプ：パラメータ送信
            'mortal': mortal.state_dict(),  # Mortalモデルの全パラメータ（重み、バイアスなど）
            'dqn': dqn.state_dict(),  # DQNモデルの全パラメータ
            'is_idle': is_idle,  # アイドル状態フラグ（更新がない場合True）
        })

def send_msg(conn: socket.socket, msg, packed=False):
    """
    ソケット経由でメッセージ（主にPyTorchオブジェクト）を送信する
    
    Args:
        conn: socket.socket - 通信に使用するソケット接続
        msg: 送信するメッセージ。通常は辞書形式のPythonオブジェクト
        packed (bool): True の場合、msg は既にシリアライズされたバイト列
                       False の場合、PyTorchのsave機能でシリアライズする
    
    プロトコル:
        1. 最初に8バイトでメッセージサイズを送信（リトルエンディアン）
        2. 続いて実際のメッセージデータを送信
    
    これにより受信側は最初にメッセージサイズを読み取り、
    正確なバイト数を受信できる
    """
    if packed:
        # 既にシリアライズされている場合はそのまま使用
        tx = msg
    else:
        # PyTorchオブジェクトをバイト列にシリアライズ
        buf = BytesIO()  # メモリ上のバイトストリームを作成
        torch.save(msg, buf)  # PyTorchのsave機能でオブジェクトをシリアライズ
        tx = buf.getbuffer()  # バッファの内容を取得
    # メッセージサイズを8バイトの符号なし整数（<Q）としてパック（リトルエンディアン）
    conn.sendall(struct.pack('<Q', len(tx)))
    # 実際のメッセージデータを送信
    conn.sendall(tx)

def recv_msg(conn: socket.socket, map_location=torch.device('cpu')):
    """
    ソケットからメッセージ（PyTorchオブジェクト）を受信してデシリアライズする
    
    Args:
        conn: socket.socket - 通信に使用するソケット接続
        map_location: torch.device - テンソルをロードするデバイス（CPU/GPU）
    
    Returns:
        受信したPythonオブジェクト（通常は辞書）
    
    動作:
        1. 最初の8バイトからメッセージサイズを読み取る
        2. 指定されたサイズ分のデータを受信
        3. PyTorchのload機能でデシリアライズ
    """
    # 最初の8バイトを受信（メッセージサイズ情報）
    rx = recv_binary(conn, 8)
    # リトルエンディアンの符号なし64ビット整数としてアンパック
    (size,) = struct.unpack('<Q', rx)
    # 実際のメッセージデータを受信
    rx = recv_binary(conn, size)
    # PyTorchのload機能でバイト列からオブジェクトを復元
    # weights_only=False: モデルの重み以外のオブジェクトも読み込み可能
    # TODO: weights_only=True にしてセキュリティを向上させる予定
    return torch.load(BytesIO(rx), weights_only=False, map_location=map_location) # TODO: weights_only=True

def recv_binary(conn: socket.socket, size):
    """
    ソケットから指定されたバイト数のデータを確実に受信する
    
    Args:
        conn: socket.socket - 通信に使用するソケット接続
        size: int - 受信するバイト数
    
    Returns:
        bytes: 受信したバイトデータ
    
    Raises:
        UnexpectedEOF: 接続が予期せず切断された場合
        AssertionError: sizeが0以下の場合
    
    重要:
        TCP/IPでは、一度のrecv()で要求した全データが受信できるとは限らない。
        この関数は、指定されたバイト数を完全に受信するまでループする。
    """
    # サイズは正の値でなければならない
    assert size > 0
    # 受信用のバッファを作成
    ret = bytearray(size)
    # memoryviewを使用して効率的なバッファ操作を可能にする
    buf = memoryview(ret)

    # すべてのデータを受信するまでループ
    while len(buf) > 0:
        # バッファに直接データを受信（コピーを避けて効率化）
        n = conn.recv_into(buf)
        # n=0は接続が閉じられたことを意味する
        if n == 0:
            raise UnexpectedEOF()
        # 受信済みの部分をスキップして、残りのバッファを更新
        buf = buf[n:]
    # bytearrayをbytesに変換して返す
    return bytes(ret)

class UnexpectedEOF(Exception):
    """
    ネットワーク通信中に予期しない接続切断が発生した場合の例外
    
    データ受信中に相手側が接続を切断した場合などに発生する。
    これにより、不完全なデータ受信を検出し、適切にエラー処理できる。
    """
    def __init__(self):
        # エラーメッセージを設定して基底クラスを初期化
        super().__init__('unexpected EOF')
