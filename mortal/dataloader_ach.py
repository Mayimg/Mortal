import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import config
from .model import Brain
from .policy_ach import PolicyNet, ValueHead
from .libriichi.dataset import GameplayLoader


class AchFileDatasetsIter(IterableDataset):
    """ACH-specific dataset iterator that performs snapshot inference to produce
    per-state (A, G, policy_logits) along with (obs, actions, masks).

    Notes:
    - Designed for num_workers=0. Snapshot models are moved to GPU and used
      inside the iterator under inference_mode().
    - Returns CPU tensors; training loop moves them to device as needed.
    - π_old reconstruction (centering/clipping/eta/softmax) is done on the
      training side; we return raw masked logits from the snapshot policy.
    - Advantage A is computed via GAE(λ) with λ from config (default 0.95),
      using snapshot ValueHead predictions, and is applied per-kyoku by
      resetting at kyoku boundaries given by `dones`.
    - Returns are computed from final hanchan rank and `pts`, discounted by γ.
    """

    def __init__(
        self,
        *,
        version: int,
        file_list: List[str],
        pts: List[float],
        snapshot_states: Tuple[dict, dict, dict],  # (mortal, policy, value)
        device: torch.device,
        file_batch_size: int = 20,
        reserve_ratio: float = 0.0,
        sample_ratio: float = 1.0,
        player_names: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        num_epochs: int = 1,
        enable_augmentation: bool = False,
        augmented_first: bool = False,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = np.asarray(pts, dtype=np.float64)
        self.snapshot_states = snapshot_states
        self.device = device
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        # Clamp to [0, 1] for safety
        self.sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        self.iterator = None

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self._build_iter()
        return self.iterator

    def _build_iter(self):
        # Create snapshot models on the specified device (GPU expected).
        res_cfg = config['resnet']
        mortal_old = Brain(version=self.version, **res_cfg).to(self.device).eval()
        policy_old = PolicyNet().to(self.device).eval()
        value_old = ValueHead().to(self.device).eval()
        m_state, p_state, v_state = self.snapshot_states
        mortal_old.load_state_dict(m_state)
        policy_old.load_state_dict(p_state)
        value_old.load_state_dict(v_state)

        # Optional compile per config; safe since we only run forward.
        if config['control'].get('enable_compile', False):
            try:
                mortal_old.compile(); policy_old.compile(); value_old.compile()
            except Exception:
                pass

        self.loader = GameplayLoader(
            version=self.version,
            oracle=False,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=self.augmented_first,
        )

        for _ in range(self.num_epochs):
            yield from self._load_files(mortal_old, policy_old, value_old, self.augmented_first)
            if self.enable_augmentation:
                yield from self._load_files(mortal_old, policy_old, value_old, not self.augmented_first)

    def _load_files(self, mortal_old, policy_old, value_old, augmented: bool):
        # Shuffle per epoch to decorrelate
        random.shuffle(self.file_list)
        # Recreate loader with augmented flag to avoid mixing augmented sequences
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
            self._populate_buffer(self.file_list[start:start + self.file_batch_size], mortal_old, policy_old, value_old)
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

    def _populate_buffer(self, file_list, mortal_old, policy_old, value_old):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = game.take_obs()                  # (T, C, L) float (list/np)
                actions = np.asarray(game.take_actions(), dtype=np.int64)  # (T,) int
                masks = np.asarray(game.take_masks(), dtype=np.bool_)      # (T, A)
                dones = np.asarray(game.take_dones(), dtype=np.bool_)
                apply_gamma = np.asarray(game.take_apply_gamma(), dtype=np.int64)  # (T,)

                grp = game.take_grp()
                player_id = int(game.take_player_id())

                # Normalize obs into a single contiguous array for fast tensor conversion
                obs = np.asarray(obs, dtype=np.float32)
                T = len(obs)
                if T == 0:
                    continue

                # Compute final reward R from final scores and pts
                final_scores = np.asarray(grp.take_final_scores(), dtype=np.int64)  # (4,)
                # rank_by_player_final: rank 0..3 for each player id
                order = np.argsort(-final_scores, kind='stable')
                rank_by_player_final = np.empty(4, dtype=np.int64)
                for r, pid in enumerate(order):
                    rank_by_player_final[pid] = r
                final_rank = int(rank_by_player_final[player_id])
                R = float(self.pts[final_rank])

                # Steps-to-done from the end of the hanchan (ignore per-kyoku dones)
                # Accumulate discounting steps using apply_gamma only.
                steps_to_done = np.zeros(T, dtype=np.int64)
                for i in range(T - 2, -1, -1):
                    steps_to_done[i] = steps_to_done[i + 1] + 1

                # Compute discounted returns G_t = gamma^{steps_to_done} * R
                G = (self.gamma ** steps_to_done.astype(np.float64)) * R

                # Snapshot inference for V_old and policy logits on GPU, chunked
                # Convert obs/masks to tensors in chunks to limit memory usage
                max_chunk = 4096
                V_list: List[torch.Tensor] = []
                logits_list: List[torch.Tensor] = []

                with torch.inference_mode():
                    for start in range(0, T, max_chunk):
                        end = min(T, start + max_chunk)
                        obs_t = torch.from_numpy(obs[start:end]).to(self.device)
                        masks_t = torch.from_numpy(masks[start:end])
                        masks_t = masks_t.to(self.device, dtype=torch.bool)
                        phi_t = mortal_old(obs_t)
                        logits_t = policy_old(phi_t, masks_t)  # masked logits (-inf for invalid)
                        V_t = value_old(phi_t)
                        logits_list.append(logits_t.detach().to('cpu'))
                        V_list.append(V_t.detach().to('cpu'))

                logits_old = torch.cat(logits_list, dim=0)  # (T, A) float32 CPU
                V = torch.cat(V_list, dim=0).to(dtype=torch.float64)  # (T,) float64 CPU for precision

                # Compute GAE(λ) per-kyoku using `dones` as kyoku boundaries.
                # Definition:
                #   δ_t = r_t + γ V_{t+1} - V_t,
                #   where r_t = 0 for all t except t = T-1 with r_{T-1} = R.
                # For non-final kyoku ends (dones[t] is True and t < T-1), we
                # bootstrap with V_{t+1} (the first state of the next kyoku),
                # but reset the GAE accumulator at the boundary so advantages
                # are computed within each kyoku only.
                deltas = torch.zeros(T, dtype=torch.float64)
                for i in range(T - 1, -1, -1):
                    r = torch.tensor(R, dtype=torch.float64) if i == T - 1 else torch.tensor(0.0, dtype=torch.float64)
                    next_v = torch.tensor(0.0, dtype=torch.float64) if i == T - 1 else V[i + 1]
                    deltas[i] = r + self.gamma * next_v - V[i]

                A = torch.zeros(T, dtype=torch.float64)
                gae = torch.tensor(0.0, dtype=torch.float64)
                # Reset before accumulation at kyoku boundaries so that the
                # boundary step's advantage does not include next-kyoku terms.
                for i in range(T - 1, -1, -1):
                    if dones[i]:
                        gae = torch.tensor(0.0, dtype=torch.float64)
                    gae = deltas[i] + self.gamma * self.gae_lambda * gae
                    A[i] = gae

                # Cast to float32 for training, keep on CPU
                A = A.to(dtype=torch.float32)
                G = torch.as_tensor(G, dtype=torch.float32)

                # Append entries per step with sampling
                if self.sample_ratio >= 1.0:
                    for i in range(T):
                        self.buffer.append([
                            torch.from_numpy(obs[i]).to(dtype=torch.float32),            # obs (C, L)
                            torch.as_tensor(actions[i], dtype=torch.int64),              # action
                            torch.from_numpy(masks[i]).to(dtype=torch.bool),             # mask (A)
                            A[i].clone(),                                               # advantage
                            G[i].clone(),                                               # return
                            logits_old[i].clone(),                                      # policy logits (A)
                        ])
                elif self.sample_ratio > 0.0:
                    rand = random.random
                    for i in range(T):
                        if rand() < self.sample_ratio:
                            self.buffer.append([
                                torch.from_numpy(obs[i]).to(dtype=torch.float32),        # obs (C, L)
                                torch.as_tensor(actions[i], dtype=torch.int64),          # action
                                torch.from_numpy(masks[i]).to(dtype=torch.bool),         # mask (A)
                                A[i].clone(),                                           # advantage
                                G[i].clone(),                                           # return
                                logits_old[i].clone(),                                  # policy logits (A)
                            ])
                else:
                    # sample_ratio == 0.0 -> drop all
                    pass


def worker_init_fn_ach(*args, **kwargs):
    # ACH iterator is intended for num_workers=0. No sharding.
    pass
