from mortal.config import config
import os
import sys

# Use a safe multiprocessing start method to avoid CUDA/threading deadlocks
# when DataLoader workers are spawned. This must be set before creating workers.
def _set_mp_start_method():
    try:
        import torch.multiprocessing as mp
        current = mp.get_start_method(allow_none=True)
        if current is None:
            mp.set_start_method('spawn', force=True)
    except Exception:
        # Fallback to Python's multiprocessing if torch.multiprocessing is unavailable
        import multiprocessing as mp
        try:
            current = mp.get_start_method(allow_none=True)
            if current is None:
                mp.set_start_method('spawn', force=True)
        except Exception:
            # As a last resort, proceed without changing the method
            pass

if config['control'].get('algo', '').lower() == 'ach':
    from mortal.train_ach import train
else:
    from mortal.train import train

if __name__ == '__main__':
    _set_mp_start_method()
    try:
        train()
    except KeyboardInterrupt:
        pass
