import torch
import socket
import struct
import time
import os
import random
import shutil
from glob import glob
from typing import *
from io import BytesIO
from functools import partial
from tqdm.auto import tqdm as orig_tqdm
from .config import config

tqdm = partial(orig_tqdm, unit='batch', dynamic_ncols=True, ascii=True)

def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def iter_grads(parameters, take=False):
    for p in parameters:
        if p.grad is not None:
            if take:
                # Set to zero instead of None to preserve the layout and make it
                # easier to assign back later
                yield p.grad.clone()
                p.grad.zero_()
            else:
                yield p.grad

def drain():
    """Drain data and return only the directory for backward compatibility."""
    drain_dir, _ = drain_with_version()
    return drain_dir

def drain_with_version():
    """Drain one version batch and return (drain_dir, param_version).

    If the server doesn't provide a version, param_version will be None.
    """
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {'type': 'drain'})
            msg = recv_msg(conn)
        if msg['count'] == 0:
            time.sleep(5)
            continue
        drain_dir = msg['drain_dir']
        param_version = msg.get('param_version')

        # Optionally augment drained self-play with human offline data.
        # Ratio is in [0,1]; 0 -> no human data, 1 -> add the same number of human files
        # as the number of drained self-play files.
        try:
            ratio = float(config['online'].get('human_data_ratio', 0.0))
        except Exception:
            ratio = 0.0
        ratio = max(0.0, min(1.0, ratio))

        if ratio > 0.0:
            try:
                self_count = int(msg.get('count') or 0)
                if self_count <= 0:
                    # As a fallback, count files in the directory
                    self_count = len(os.listdir(drain_dir))
                add_count = int(self_count * ratio)
                if add_count > 0:
                    # Resolve offline dataset file list once
                    offline_files = _get_offline_file_list()
                    if offline_files:
                        # Sample without replacement up to available count
                        add_count = min(add_count, len(offline_files))
                        selected = random.sample(offline_files, add_count)
                        # Link or copy into the drained directory
                        os.makedirs(drain_dir, exist_ok=True)
                        for i, src in enumerate(selected):
                            base = os.path.basename(src)
                            dst = os.path.join(drain_dir, f'human_{i:06d}_{base}')
                            if not os.path.exists(dst):
                                try:
                                    os.symlink(os.path.abspath(src), dst)
                                except Exception:
                                    # Fallback: copy if symlink unavailable
                                    try:
                                        shutil.copyfile(src, dst)
                                    except Exception:
                                        pass
            except Exception:
                # Best-effort: if anything goes wrong, proceed with original drain
                pass

        return drain_dir, param_version


# Cache for offline file listing to avoid repeated scans/loads during training
_OFFLINE_FILE_LIST_CACHE: Optional[List[str]] = None

def _get_offline_file_list() -> List[str]:
    global _OFFLINE_FILE_LIST_CACHE
    if _OFFLINE_FILE_LIST_CACHE is not None:
        return _OFFLINE_FILE_LIST_CACHE

    files: List[str] = []
    try:
        file_index = config['dataset'].get('file_index')
        if file_index and os.path.exists(file_index):
            obj = torch.load(file_index, weights_only=True, map_location=torch.device('cpu'))
            files = list(obj.get('file_list', []))
        else:
            globs = config['dataset'].get('globs', [])
            for pat in globs:
                files.extend(glob(pat, recursive=True))
            files.sort()
    except Exception:
        files = []

    _OFFLINE_FILE_LIST_CACHE = files
    return files

def submit_param(mortal, dqn, is_idle=False):
    """Submit parameters to the server and return the new param_version.

    For legacy servers that do not respond with a version, returns None.
    """
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    with socket.socket() as conn:
        conn.connect(remote)
        send_msg(conn, {
            'type': 'submit_param',
            'mortal': mortal.state_dict(),
            'dqn': dqn.state_dict(),
            'is_idle': is_idle,
        })
        try:
            rsp = recv_msg(conn)
            return rsp.get('param_version')
        except Exception:
            return None

def send_msg(conn: socket.socket, msg, packed=False):
    if packed:
        tx = msg
    else:
        buf = BytesIO()
        torch.save(msg, buf)
        tx = buf.getbuffer()
    conn.sendall(struct.pack('<Q', len(tx)))
    conn.sendall(tx)

def recv_msg(conn: socket.socket, map_location=torch.device('cpu')):
    rx = recv_binary(conn, 8)
    (size,) = struct.unpack('<Q', rx)
    rx = recv_binary(conn, size)
    return torch.load(BytesIO(rx), weights_only=False, map_location=map_location) # TODO: weights_only=True

def recv_binary(conn: socket.socket, size):
    assert size > 0
    ret = bytearray(size)
    buf = memoryview(ret)

    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)

class UnexpectedEOF(Exception):
    def __init__(self):
        super().__init__('unexpected EOF')
