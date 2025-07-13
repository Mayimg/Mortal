#!/usr/bin/env python3
"""
麻雀牌画像合成スクリプト
Front.pngを背景として、各牌画像を前面に合成する
"""

import os
import shutil
from PIL import Image

# ディレクトリパス
source_dir = "/home/naoyuki/mahjong01/Mortal/log-viewer/files/Export/Regular"
output_dir = "/home/naoyuki/mahjong01/Mortal/log-viewer/files/Export/New_Regular"

# Front.pngを読み込む（背景画像）
front_path = os.path.join(source_dir, "Front.png")
front_img = Image.open(front_path).convert("RGBA")

# 処理する画像のリスト取得
image_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

# 合成対象外のファイル
exclude_from_merge = ["Back.png", "Front.png"]
# コピーのみ行うファイル
copy_only = ["Back.png", "Front.png"]

print(f"画像合成開始...")
print(f"ソースディレクトリ: {source_dir}")
print(f"出力ディレクトリ: {output_dir}")

# 各画像を処理
for filename in image_files:
    source_path = os.path.join(source_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    if filename in copy_only:
        # Back.pngとFront.pngはそのままコピー
        print(f"コピー中: {filename}")
        shutil.copy2(source_path, output_path)
    elif filename not in exclude_from_merge:
        # その他の画像は合成
        print(f"合成中: {filename}")
        
        # 牌画像を読み込む
        tile_img = Image.open(source_path).convert("RGBA")
        
        # Front.pngのコピーを作成
        result_img = front_img.copy()
        
        # 牌画像を前面に合成
        # paste()メソッドで透過を考慮した合成を行う
        result_img.paste(tile_img, (0, 0), tile_img)
        
        # 結果を保存
        result_img.save(output_path, "PNG")

print("画像合成完了！")
print(f"合成された画像は {output_dir} に保存されました。")