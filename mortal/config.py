import toml
import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
# If config_file is relative path, look for it relative to the parent directory
if not os.path.isabs(config_file):
    # Get the directory containing this config.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory to the project root
    project_root = os.path.dirname(current_dir)
    config_file = os.path.join(project_root, config_file)

with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)
