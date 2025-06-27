#!/usr/bin/env python3
import sys
import gzip
import os

def create_viewer(log_gz_path, output_html_path):
    """Create an HTML viewer for a compressed mjai log file"""
    
    # Read the compressed log file
    with gzip.open(log_gz_path, 'rt', encoding='utf-8') as f:
        log_content = f.read().strip()
    
    # Read the template HTML
    with open('log-viewer/index.example.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Find the allActions variable and replace its content
    start_marker = 'allActions = `'
    end_marker = "`.trim().split('\\n')"
    
    start_idx = template.find(start_marker) + len(start_marker)
    end_idx = template.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find allActions variable in template")
        return False
    
    # Create new HTML with the actual log data
    new_html = template[:start_idx] + '\n' + log_content + '\n' + template[end_idx:]
    
    # Write the output HTML
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    
    print(f"Created viewer: {output_html_path}")
    print(f"Open this file in a web browser to view the game log")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_log_viewer.py <log_file.json.gz> [output.html]")
        print("Example: python create_log_viewer.py logs/test_play/10000_8192_a.json.gz")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename based on input
        base_name = os.path.basename(log_file).replace('.json.gz', '')
        output_file = f'log-viewer/{base_name}.html'
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    create_viewer(log_file, output_file)