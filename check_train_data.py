import gzip
import json
from pathlib import Path

def validate_mjai_file(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        game_started = False
        kyoku_started = False
        
        for line in f:
            event = json.loads(line)
            event_type = event.get('type')
            
            if event_type == 'start_game':
                game_started = True
            elif event_type == 'start_kyoku':
                kyoku_started = True
            elif event_type == 'end_kyoku':
                kyoku_started = False
            elif event_type == 'end_game':
                game_started = False
            
            # 終端イベントでのdeltasをチェック
            if event_type in ['hora', 'ryukyoku']:
                if 'deltas' not in event:
                    print(f"警告: {event_type}イベントにdeltasがありません")
        
        return not (game_started or kyoku_started)

# すべてのファイルを検証
for file in Path('data/training').rglob('*.json.gz'):
    print(f"確認中: {file}")
    if not validate_mjai_file(file):
        print(f"無効なファイル: {file}")