import torch
import socket
import struct
import time
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
        return msg['drain_dir'], msg.get('param_version')

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
