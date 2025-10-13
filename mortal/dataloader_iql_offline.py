from __future__ import annotations

import os
import random
from os import path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import config
from .model import GRP
from .reward_calculator import RewardCalculator
from .libriichi.dataset import GameplayLoader


class IQLOfflineDataset(IterableDataset):
    """Iterable dataset yielding transitions for offline IQL training."""

    def __init__(
        self,
        *,
        version: int,
        file_list: List[str],
        pts: List[float],
        file_batch_size: int = 20,
        reserve_ratio: float = 0.0,
        sample_ratio: float = 1.0,
        player_names: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        num_epochs: int = 1,
        enable_augmentation: bool = False,
        augmented_first: bool = False,
        gamma: float = 0.995,
    ) -> None:
        super().__init__()
        self.version = int(version)
        self.file_list = file_list
        self.pts = pts
        self.file_batch_size = int(file_batch_size)
        self.reserve_ratio = float(reserve_ratio)
        self.sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = int(num_epochs)
        self.enable_augmentation = bool(enable_augmentation)
        self.augmented_first = bool(augmented_first)
        self.gamma = float(gamma)

        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = self._build_iter()
        return self._iterator

    def _build_iter(self):
        # Avoid CPU oversubscription: each worker is a process, and the Rust
        # GameplayLoader uses rayon for parallel IO/parse. Limit rayon threads
        # per worker to 1 unless the user sets it explicitly.
        # os.environ.setdefault("RAYON_NUM_THREADS", "1")

        grp = GRP(**config['grp']['network'])
        grp_cfg = config['grp']
        grp_state_path = grp_cfg.get('best_state_file') if grp_cfg.get('best_state_file') and path.exists(grp_cfg.get('best_state_file')) else grp_cfg['state_file']
        grp_state = torch.load(
            grp_state_path,
            weights_only=True,
            map_location=torch.device('cpu'),
        )
        grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self._load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self._load_files(not self.augmented_first)

    def _load_files(self, augmented: bool):
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version=self.version,
            oracle=False,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer: list = []

        for start in range(0, len(self.file_list), self.file_batch_size):
            old_size = len(self.buffer)
            self._populate_buffer(self.file_list[start:start + self.file_batch_size])
            new_size = len(self.buffer)
            reserved = int((new_size - old_size) * self.reserve_ratio)
            if reserved > len(self.buffer):
                continue
            random.shuffle(self.buffer)
            for item in self.buffer[reserved:]:
                yield item
            del self.buffer[reserved:]

        random.shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.buffer.clear()

    def _populate_buffer(self, file_list: List[str]) -> None:
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = np.asarray(game.take_obs(), dtype=np.float32)
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                masks = np.asarray(game.take_masks(), dtype=np.bool_)

                at_kyoku = np.array([int(k) for k in game.take_at_kyoku()], dtype=np.int64)

                dones = np.asarray(game.take_dones(), dtype=np.bool_)

                apply_gamma = np.array([int(x) for x in game.take_apply_gamma()], dtype=np.int64)

                grp = game.take_grp()
                player_id = int(game.take_player_id())

                T = len(obs)
                if T == 0:
                    continue

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)

                assert len(kyoku_rewards) >= int(at_kyoku[-1]) + 1

                next_obs = np.concatenate((obs[1:], obs[-1:]), axis=0)
                next_masks = np.concatenate((masks[1:], masks[-1:]), axis=0)

                rand = random.random
                for i in range(T):
                    if self.sample_ratio < 1.0 and rand() >= self.sample_ratio:
                        continue

                    done_flag = bool(dones[i])
                    reward = float(kyoku_rewards[int(at_kyoku[i])]) if done_flag else 0.0
                    discount = self.gamma if apply_gamma[i] else 1.0

                    self.buffer.append([
                        torch.from_numpy(obs[i]).to(dtype=torch.float32),
                        torch.from_numpy(next_obs[i]).to(dtype=torch.float32),
                        torch.as_tensor(actions[i], dtype=torch.int64),
                        torch.from_numpy(masks[i]).to(dtype=torch.bool),
                        torch.from_numpy(next_masks[i]).to(dtype=torch.bool),
                        torch.tensor(reward, dtype=torch.float32),
                        torch.tensor(discount, dtype=torch.float32),
                        torch.tensor(done_flag, dtype=torch.bool),
                    ])


def worker_init_fn_iql_offline(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
