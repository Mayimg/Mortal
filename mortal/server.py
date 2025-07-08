"""
Mortal オンライン学習システムのパラメータサーバー

このファイルは、分散学習環境において中央サーバーとして機能し、以下の役割を持つ：
1. ワーカー（対戦を行うプロセス）への学習済みモデルパラメータの配布
2. ワーカーから送信されるゲームリプレイログの収集・バッファリング
3. トレーナー（学習を行うプロセス）へのリプレイデータの転送
4. トレーナーからの更新されたモデルパラメータの受け取りと管理

動作フロー：
- ワーカーは対戦前に最新のモデルパラメータを取得（get_param）
- ワーカーは対戦後にリプレイログを送信（submit_replay）
- トレーナーは学習後に更新したパラメータを送信（submit_param）
- トレーナーは学習用データを取得（drain）

スレッドセーフティ：
- dir_lock: ファイルシステム操作（バッファ管理）の排他制御
- param_lock: モデルパラメータへのアクセスの排他制御
"""

import prelude  # システム全体の初期設定（ロギング設定、エンコーディング設定など）

import logging  # ロギング機能
import shutil  # ファイル・ディレクトリ操作（移動、削除など）
import torch  # PyTorchライブラリ（モデルパラメータのシリアライズに使用）
import sys  # システム関連機能（例外情報の取得など）
import os  # OS関連機能（ファイルシステム操作）
from os import path  # パス操作のための便利な関数
from io import BytesIO  # メモリ上でバイナリデータを扱うためのバッファ
from typing import *  # 型ヒントのためのインポート
from collections import OrderedDict  # 順序付き辞書（モデルパラメータの保存に使用）
from dataclasses import dataclass  # データクラスデコレータ
from socketserver import ThreadingTCPServer, BaseRequestHandler  # TCPサーバー実装の基底クラス
from threading import Lock  # スレッド間の排他制御のためのロック
from common import send_msg, recv_msg, UnexpectedEOF  # 共通通信関数と例外クラス
from config import config  # 設定ファイルから読み込まれた設定情報

@dataclass
class State:
    """サーバーの状態を管理するデータクラス"""
    buffer_dir: str          # リプレイログを一時保存するバッファディレクトリのパス
    drain_dir: str           # トレーナーに転送するデータを置くドレインディレクトリのパス
    capacity: int            # バッファの最大容量（これを超えるとワーカーからの送信を拒否）
    force_sequential: bool   # シーケンシャルモード（True時はトレーナーがビジーならパラメータ配布を停止）
    dir_lock: Lock          # ディレクトリ操作用の排他制御ロック
    param_lock: Lock        # パラメータアクセス用の排他制御ロック
    # fields below are protected by dir_lock
    buffer_size: int        # 現在バッファに保存されているファイル数
    submission_id: int      # 次に受信するリプレイに付与するID（ファイル名の重複を防ぐ）
    # fields below are protected by param_lock
    mortal_param: Optional[OrderedDict]  # Mortalモデル（メイン戦略モデル）のパラメータ
    dqn_param: Optional[OrderedDict]     # DQNモデル（価値関数）のパラメータ
    param_version: int                   # パラメータのバージョン番号（更新のたびにインクリメント）
    idle_param_version: int              # トレーナーがアイドル状態になった時のパラメータバージョン
S = None  # グローバルなサーバー状態オブジェクト（main関数で初期化される）

class Handler(BaseRequestHandler):
    """各クライアント接続を処理するハンドラークラス"""
    def handle(self):
        """クライアントからの接続を処理するメインメソッド（ThreadingTCPServerから自動的に呼ばれる）"""
        msg = self.recv_msg()  # クライアントからメッセージを受信（torch.loadでデシリアライズ）
        match msg['type']:     # メッセージタイプに応じて適切なハンドラーメソッドを呼び出す
            # called by workers
            case 'get_param':      # ワーカーが最新のモデルパラメータを要求
                self.handle_get_param(msg)
            case 'submit_replay':  # ワーカーがゲームリプレイログを送信
                self.handle_submit_replay(msg)
            # called by trainer
            case 'submit_param':   # トレーナーが更新したモデルパラメータを送信
                self.handle_submit_param(msg)
            case 'drain':          # トレーナーがバッファからデータを取得
                self.handle_drain()

    def handle_get_param(self, msg):
        """ワーカーからのパラメータ取得要求を処理"""
        with S.dir_lock:  # バッファサイズの確認のためにdir_lockを取得
            overflow = S.buffer_size >= S.capacity  # バッファが満杯かどうかをチェック
            with S.param_lock:  # パラメータの存在確認のためにparam_lockも取得（デッドロック回避のため順序重要）
                has_param = S.mortal_param is not None and S.dqn_param is not None  # 両方のモデルパラメータが存在するか確認
        if overflow:  # バッファが満杯の場合
            self.send_msg({'status': 'samples overflow'})  # ワーカーに待機を促すメッセージを送信
            return
        if not has_param:  # パラメータがまだ設定されていない場合
            self.send_msg({'status': 'empty param'})  # パラメータが存在しないことを通知
            return

        client_param_version = msg['param_version']  # クライアントが現在持っているパラメータのバージョン
        buf = BytesIO()  # レスポンスをシリアライズするためのバッファ
        with S.param_lock:  # パラメータアクセスのためのロック
            if S.force_sequential and S.idle_param_version <= client_param_version:
                # シーケンシャルモードかつクライアントが最新のアイドルバージョン以上を持っている場合
                # （トレーナーがまだ学習中であることを示す）
                res = {'status': 'trainer is busy'}
            else:
                # 正常にパラメータを返す
                res = {
                    'status': 'ok',
                    'mortal': S.mortal_param,        # Mortalモデルのパラメータ
                    'dqn': S.dqn_param,              # DQNモデルのパラメータ
                    'param_version': S.param_version, # 現在のパラメータバージョン
                }
            torch.save(res, buf)  # レスポンスをシリアライズ
        self.send_msg(buf.getbuffer(), packed=True)  # packed=Trueで既にシリアライズ済みのデータを送信

    def handle_submit_replay(self, msg):
        """ワーカーからのリプレイログ送信を処理"""
        with S.dir_lock:  # ファイルシステム操作のためのロック
            for filename, content in msg['logs'].items():  # 送信されたログファイルを1つずつ処理
                # ファイル名の前にsubmission_idを付けて重複を防ぐ
                filepath = path.join(S.buffer_dir, f'{S.submission_id}_{filename}')
                with open(filepath, 'wb') as f:  # バイナリモードでファイルを開く
                    f.write(content)  # ログ内容をファイルに書き込む
            S.buffer_size += len(msg['logs'])  # バッファサイズを受信したファイル数分増やす
            S.submission_id += 1  # 次の送信用にIDをインクリメント
            logging.info(f'total buffer size: {S.buffer_size}')  # 現在のバッファサイズをログ出力

    def handle_submit_param(self, msg):
        """トレーナーからの更新されたモデルパラメータを受け取って保存"""
        with S.param_lock:  # パラメータ更新のための排他制御
            S.mortal_param = msg['mortal']  # Mortalモデルの新しいパラメータを設定
            S.dqn_param = msg['dqn']        # DQNモデルの新しいパラメータを設定
            S.param_version += 1            # パラメータバージョンをインクリメント
            if msg['is_idle']:              # トレーナーがアイドル状態（学習完了）の場合
                S.idle_param_version = S.param_version  # アイドルバージョンを更新（force_sequentialモードで使用）

    def handle_drain(self):
        """トレーナーからの要求でバッファ内のデータをドレインディレクトリに転送"""
        drained_size = 0  # 転送したファイル数
        with S.dir_lock:  # ディレクトリ操作のための排他制御
            buffer_list = os.listdir(S.buffer_dir)  # バッファディレクトリ内のファイルリストを取得
            raw_count = len(buffer_list)  # 実際のファイル数
            assert raw_count == S.buffer_size  # ファイル数とカウンタが一致することを確認（整合性チェック）
            # 転送条件：シーケンシャルモードでない、またはバッファが満杯、かつファイルが存在する
            if (not S.force_sequential or raw_count >= S.capacity) and raw_count > 0:
                old_drain_list = os.listdir(S.drain_dir)  # 既存のドレインディレクトリの内容を取得
                for filename in old_drain_list:  # 古いファイルをすべて削除
                    filepath = path.join(S.drain_dir, filename)
                    os.remove(filepath)
                for filename in buffer_list:  # バッファ内のすべてのファイルを移動
                    src = path.join(S.buffer_dir, filename)  # 移動元パス
                    dst = path.join(S.drain_dir, filename)   # 移動先パス
                    shutil.move(src, dst)  # ファイルを移動（元のファイルは削除される）
                drained_size = raw_count  # 転送したファイル数を記録
                S.buffer_size = 0  # バッファサイズをリセット
                logging.info(f'files transferred to trainer: {drained_size}')  # 転送完了をログ出力
                logging.info(f'total buffer size: {S.buffer_size}')  # 新しいバッファサイズをログ出力
        self.send_msg({  # トレーナーに転送結果を返す
            'count': drained_size,     # 転送したファイル数
            'drain_dir': S.drain_dir,  # ファイルが置かれたディレクトリパス
        })

    def send_msg(self, msg, packed=False):
        """共通のsend_msg関数をラップして、現在の接続に対してメッセージを送信"""
        return send_msg(self.request, msg, packed)  # self.requestはクライアントとのソケット接続

    def recv_msg(self):
        """共通のrecv_msg関数をラップして、現在の接続からメッセージを受信"""
        return recv_msg(self.request)  # self.requestはクライアントとのソケット接続

class Server(ThreadingTCPServer):
    """マルチスレッドTCPサーバーの実装（各接続を別スレッドで処理）"""
    def handle_error(self, request, client_address):
        """エラーハンドリングのカスタマイズ（特定のエラーを無視）"""
        typ, _, _ = sys.exc_info()  # 発生した例外の情報を取得
        if typ is BrokenPipeError or typ is UnexpectedEOF:  # パイプ破損や予期しないEOFは無視
            return  # これらは正常な切断として扱う
        return super().handle_error(request, client_address)  # その他のエラーは親クラスのハンドラに委譲

def main():
    """サーバーのエントリーポイント（初期化と起動）"""
    global S  # グローバル状態オブジェクトを宣言
    cfg = config['online']['server']  # 設定ファイルからサーバー設定を取得
    S = State(  # サーバー状態オブジェクトを初期化
        buffer_dir = path.abspath(cfg['buffer_dir']),       # バッファディレクトリの絶対パス
        drain_dir = path.abspath(cfg['drain_dir']),         # ドレインディレクトリの絶対パス
        capacity = cfg['capacity'],                         # バッファの最大容量
        force_sequential = cfg['force_sequential'],         # シーケンシャルモードフラグ
        dir_lock = Lock(),          # ディレクトリ操作用のロックを作成
        param_lock = Lock(),        # パラメータアクセス用のロックを作成
        buffer_size = 0,            # 初期バッファサイズは0
        submission_id = 0,          # 初期送信IDは0
        mortal_param = None,        # Mortalモデルパラメータは未設定
        dqn_param = None,           # DQNモデルパラメータは未設定
        param_version = 0,          # 初期パラメータバージョンは0
        idle_param_version = 0,     # 初期アイドルバージョンは0
    )

    bind_addr = (config['online']['remote']['host'], config['online']['remote']['port'])  # バインドするアドレスとポート
    if path.isdir(S.buffer_dir):       # バッファディレクトリが既に存在する場合
        shutil.rmtree(S.buffer_dir)    # 削除してクリーンな状態から開始
    if path.isdir(S.drain_dir):        # ドレインディレクトリが既に存在する場合
        shutil.rmtree(S.drain_dir)     # 削除してクリーンな状態から開始
    os.makedirs(S.buffer_dir)          # 新しいバッファディレクトリを作成
    os.makedirs(S.drain_dir)           # 新しいドレインディレクトリを作成

    with Server(bind_addr, Handler, bind_and_activate=False) as server:  # TCPサーバーを作成
        server.allow_reuse_address = True   # アドレスの再利用を許可（再起動時の待機時間を回避）
        server.daemon_threads = True        # スレッドをデーモンスレッドとして実行（メイン終了時に自動終了）
        server.server_bind()                # ソケットをアドレスにバインド
        server.server_activate()            # サーバーをアクティブ化（リスニング開始）
        host, port = bind_addr
        logging.info(f'listening on {host}:{port}')  # リスニング開始をログ出力
        server.serve_forever()              # 無限ループで接続を待ち受ける

if __name__ == '__main__':
    """スクリプトが直接実行された場合のみmain関数を呼び出す"""
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+Cによる中断をキャッチ
        pass  # 正常終了として扱う（エラーメッセージを表示しない）
