import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.checkpoint import checkpoint
from typing import *
from functools import partial
from itertools import permutations
from .libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        x = weight.unsqueeze(-1) * x
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
        use_checkpoint = False,
        checkpoint_segments = 8,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments

        # Create blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResBlock(
                conv_channels,
                norm_builder = norm_builder,
                actv_builder = actv_builder,
                pre_actv = pre_actv,
            ))

        # Initial convolution
        self.initial_conv = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        
        # Pre/post processing layers
        if pre_actv:
            self.pre_block_layers = nn.Sequential(norm_builder(), actv_builder())
            self.post_block_layers = nn.Sequential(norm_builder(), actv_builder())
        else:
            self.pre_block_layers = nn.Sequential(norm_builder(), actv_builder())
            self.post_block_layers = nn.Identity()
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )

        # Calculate blocks per segment for checkpointing
        if use_checkpoint:
            self.blocks_per_segment = max(1, num_blocks // checkpoint_segments)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Pre-block processing
        x = self.pre_block_layers(x)
        
        # Apply blocks with optional checkpointing
        if self.use_checkpoint and self.training:
            # Divide blocks into segments and apply checkpointing
            for i in range(0, len(self.blocks), self.blocks_per_segment):
                segment_end = min(i + self.blocks_per_segment, len(self.blocks))
                # Create a function that applies a segment of blocks
                def segment_forward(x, start_idx, end_idx):
                    for j in range(start_idx, end_idx):
                        x = self.blocks[j](x)
                    return x
                # Apply checkpoint to this segment
                x = checkpoint(segment_forward, x, i, segment_end, use_reentrant=False)
        else:
            # Normal forward pass without checkpointing
            for block in self.blocks:
                x = block(x)
        
        # Post-block processing
        x = self.post_block_layers(x)
        
        # Final layers
        x = self.final_layers(x)
        
        return x

class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1, use_checkpoint=False, checkpoint_segments=8):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01)
        actv_builder = partial(nn.Mish, inplace=True)
        pre_actv = True

        match version:
            case 1:
                actv_builder = partial(nn.ReLU, inplace=True)
                pre_actv = False
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                )
                self.mu_head = nn.Linear(512, 512)
                self.logsig_head = nn.Linear(512, 512)
            case 2:
                pass
            case 3 | 4:
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

        self.encoder = ResNet(
            in_channels = in_channels,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = norm_builder,
            actv_builder = actv_builder,
            pre_actv = pre_actv,
            use_checkpoint = use_checkpoint,
            checkpoint_segments = checkpoint_segments,
        )
        self.actv = actv_builder()

        # always use EMA or CMA when True
        self._freeze_bn = False

    def forward(self, obs: Tensor, invisible_obs: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        phi = self.encoder(obs)

        match self.version:
            case 1:
                latent_out = self.latent_net(phi)
                mu = self.mu_head(latent_out)
                logsig = self.logsig_head(latent_out)
                return mu, logsig
            case 2 | 3 | 4:
                return self.actv(phi)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
                    # I don't think this benefits
                    # module.requires_grad_(False)
        return self

    def reset_running_stats(self):
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)

class AuxNet(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(1024, sum(dims), bias=False)

    def forward(self, x):
        return self.net(x).split(self.dims, dim=-1)

class DQN(nn.Module):
    def __init__(self, *, version=1):
        super().__init__()
        self.version = version
        match version:
            case 1:
                self.v_head = nn.Linear(512, 1)
                self.a_head = nn.Linear(512, ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                self.net = nn.Linear(1024, 1 + ACTION_SPACE)
                nn.init.constant_(self.net.bias, 0)

    def forward(self, phi, mask):
        if self.version == 4:
            v, a = self.net(phi).split((1, ACTION_SPACE), dim=-1)
        else:
            v = self.v_head(phi)
            a = self.a_head(phi)
        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
        mask_sum = mask.sum(-1, keepdim=True)
        a_mean = a_sum / mask_sum
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q

class GRP(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        # 4-layer MLP with Mish; `num_layers` kept for config compatibility (unused)
        width = int(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(GRP_SIZE, width),
            nn.Mish(inplace=True),
            nn.Linear(width, width),
            nn.Mish(inplace=True),
            nn.Linear(width, width),
            nn.Mish(inplace=True),
            nn.Linear(width, width),
            nn.Mish(inplace=True),
            nn.Linear(width, 24),
        )
        for mod in self.modules():
            mod.to(torch.float64)

        # perms are the permutations of all possible rank-by-player result
        perms = torch.tensor(list(permutations(range(4))))
        perms_t = perms.transpose(0, 1)
        self.register_buffer('perms', perms)     # (24, 4)
        self.register_buffer('perms_t', perms_t) # (4, 24)

    # input features per step:
    #   base 7 dims + 16 rank one-hot + 168 agari improvement (total 191)
    # Consumes only the last step per sequence and outputs 24-class logits.
    def forward(self, inputs: List[Tensor]):
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.int64)
        inputs = pad_sequence(inputs, batch_first=True)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        return self.forward_packed(packed_inputs)

    def forward_packed(self, packed_inputs):
        # Unpack and take the last step features of each sequence
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_inputs, batch_first=True)
        batch_indices = torch.arange(padded.shape[0], device=padded.device)
        last_idx = lengths.to(device=padded.device) - 1
        last_feat = padded[batch_indices, last_idx]  # (B, GRP_SIZE)
        logits = self.mlp(last_feat)
        return logits

    # (N, 24) -> (N, player, rank_prob)
    def calc_matrix(self, logits: Tensor):
        batch_size = logits.shape[0]
        probs = logits.softmax(-1)
        matrix = torch.zeros(batch_size, 4, 4, dtype=probs.dtype)
        for player in range(4):
            for rank in range(4):
                cond = self.perms_t[player] == rank
                matrix[:, player, rank] = probs[:, cond].sum(-1)
        return matrix

    # (N, 4) -> (N)
    def get_label(self, rank_by_player: Tensor):
        batch_size = rank_by_player.shape[0]
        perms = self.perms.expand(batch_size, -1, -1).transpose(0, 1)
        mappings = (perms == rank_by_player).all(-1).nonzero()

        labels = torch.zeros(batch_size, dtype=torch.int64, device=mappings.device)
        labels[mappings[:, 1]] = mappings[:, 0]
        return labels
