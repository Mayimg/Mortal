import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import config
from .model import GRP
from .reward_calculator import RewardCalculator
from .libriichi.dataset import GameplayLoader


class AchOfflineFileDatasetsIter(IterableDataset):
    """ACH offline dataset iterator for supervised policy + value learning.

    Yields per-move tuples:
      - obs: FloatTensor (C, L)
      - action: LongTensor ()
      - mask: BoolTensor (A)
      - G: FloatTensor () discounted return toward kyoku-end expected point

    Notes:
    - Follows the existing offline dataset style (see mortal/dataloader.py).
    - Computes kyoku-level expected points via GRP like ACH online iterator,
      then assigns per-step G = gamma^(steps_to_kyoku_end) * R_kyoku[at_kyoku].
    - Sampling can be applied via sample_ratio in [0,1].
    """

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
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.gamma = float(gamma)

        self.iterator = None

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self._build_iter()
        return self.iterator

    def _build_iter(self):
        # Build GRP + RewardCalculator on CPU for expected-pts per-kyoku.
        grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self._load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self._load_files(not self.augmented_first)

    def _load_files(self, augmented: bool):
        # Shuffle per epoch to decorrelate
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version=self.version,
            oracle=False,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer = []

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

    def _populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = game.take_obs()                  # (T, C, L) float (list/np)
                actions = game.take_actions()          # (T,) int
                masks = game.take_masks()              # (T, A) bool
                dones = game.take_dones()              # (T,) bool
                apply_gamma = game.take_apply_gamma()  # (T,) int
                at_kyoku = game.take_at_kyoku()        # (T,)

                grp = game.take_grp()
                player_id = int(game.take_player_id())

                obs = np.asarray(obs, dtype=np.float32)
                actions = np.asarray(actions, dtype=np.int64)
                masks = np.asarray(masks, dtype=np.bool_)
                dones = np.asarray(dones, dtype=np.bool_)
                apply_gamma = np.asarray(apply_gamma, dtype=np.int64)
                T = len(obs)
                if T == 0:
                    continue

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_exp_pt(player_id, grp_feature, rank_by_player)  # (num_kyoku,)
                assert len(kyoku_rewards) >= int(at_kyoku[-1]) + 1

                # Steps to kyoku end; reset on kyoku boundary
                steps_to_kyoku_end = np.zeros(T, dtype=np.int64)
                for i in range(T - 1, -1, -1):
                    if i < T - 1 and not dones[i]:
                        steps_to_kyoku_end[i] = steps_to_kyoku_end[i + 1] + int(apply_gamma[i])
                    else:
                        steps_to_kyoku_end[i] = 0

                R_per_step = np.fromiter((float(kyoku_rewards[int(k)]) for k in at_kyoku), dtype=np.float64, count=T)
                G = (self.gamma ** steps_to_kyoku_end.astype(np.float64)) * R_per_step

                if self.sample_ratio >= 1.0:
                    for i in range(T):
                        self.buffer.append([
                            torch.from_numpy(obs[i]).to(dtype=torch.float32),            # obs (C, L)
                            torch.as_tensor(actions[i], dtype=torch.int64),              # action
                            torch.from_numpy(masks[i]).to(dtype=torch.bool),             # mask (A)
                            torch.as_tensor(G[i], dtype=torch.float32),                  # return
                        ])
                elif self.sample_ratio > 0.0:
                    rand = random.random
                    for i in range(T):
                        if rand() < self.sample_ratio:
                            self.buffer.append([
                                torch.from_numpy(obs[i]).to(dtype=torch.float32),        # obs (C, L)
                                torch.as_tensor(actions[i], dtype=torch.int64),          # action
                                torch.from_numpy(masks[i]).to(dtype=torch.bool),         # mask (A)
                                torch.as_tensor(G[i], dtype=torch.float32),              # return
                            ])


def worker_init_fn_ach_offline(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]

