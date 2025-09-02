def train(is_first_run: bool = False):
    from . import prelude

    import logging
    import sys
    import os
    import gc
    import shutil
    import copy
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from .common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from .player import TestPlayer
    from .dataloader_ach import AchFileDatasetsIter
    from .model import Brain
    from .policy_ach import PolicyNet, ValueHead
    from .libriichi.consts import obs_shape
    from .config import config

    assert config['control']['online'], 'ACH trainer currently supports online=true only'

    version = config['control']['version']
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    # ACH hyperparameters (with sane defaults)
    ach_cfg = config.get('ach', {})
    eta = float(ach_cfg.get('eta', 1.0))
    logit_threshold = float(ach_cfg.get('logit_threshold', 6.0))
    ratio_clip = float(ach_cfg.get('ratio_clip', 0.5))
    # Advantage normalization/clipping width (per minibatch)
    adv_clip = float(ach_cfg.get('adv_clip', 5.0))
    gae_lambda = float(ach_cfg.get('gae_lambda', 0.95))  # kept for completeness
    entropy_coef = float(ach_cfg.get('entropy_coef', 1e-2))
    value_coef = float(ach_cfg.get('value_coef', 0.5))

    batch_size = int(ach_cfg.get('batch_size', config['control']['batch_size']))
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']

    pts = config['env']['pts']
    gamma = float(config['env'].get('gamma', 0.995))
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']

    lr = float(config['optim'].get('lr', 2.5e-4))
    weight_decay = float(config['optim'].get('weight_decay', 0.0))
    eps = float(config['optim'].get('eps', 1e-8))
    betas = tuple(config['optim'].get('betas', [0.9, 0.999]))
    max_grad_norm = float(config['optim'].get('max_grad_norm', 0.0))

    mortal = Brain(version=version, **config['resnet']).to(device)
    policy = PolicyNet().to(device)
    value = ValueHead().to(device)

    if enable_compile:
        mortal.compile()
        policy.compile()
        value.compile()

    logging.info(f'[ACH] version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy):,}')
    logging.info(f'value params: {parameter_count(value):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    # Build parameter groups
    def split_decay_params(model):
        decay_params, no_decay_params = [], []
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        return decay_params, no_decay_params

    decay, no_decay = [], []
    for m in (mortal, policy, value):
        d, nd = split_decay_params(m)
        decay.extend(d); no_decay.extend(nd)

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay},
    ]
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.0, betas=betas, eps=eps)
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    best_perf = {'avg_rank': 4., 'avg_pt': -135.}

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        # For compatibility, load policy from 'current_dqn'
        if 'policy' in state:
            policy.load_state_dict(state['policy'])
        elif 'current_dqn' in state:
            policy.load_state_dict(state['current_dqn'])
        if 'value' in state:
            value.load_state_dict(state['value'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        steps = state.get('steps', 0)
        best_perf = state.get('best_perf', best_perf)

    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    # Initial param push so actors can fetch ACH params immediately
    init_ver = submit_param(mortal, policy, is_idle=True)
    logging.info('[ACH] initial param has been submitted')

    # Track deployed snapshots by param_version to reconstruct snapshot in iterator.
    # Key: param_version (int or None), Value: (mortal_state, policy_state, value_state)
    deployed_snapshots = {}
    if init_ver is None:
        # Legacy fallback: use a single None key
        init_ver = None
    deployed_snapshots[init_ver] = (
        copy.deepcopy(mortal.state_dict()),
        copy.deepcopy(policy.state_dict()),
        copy.deepcopy(value.state_dict()),
    )

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'ratio_mean': 0.0,
        'ratio_clip_rate': 0.0,
        # A (advantage) monitoring (normalized, pre-clip)
        'adv_norm_mean': 0.0,
        'adv_norm_std': 0.0,
        'adv_clip_frac': 0.0,
    }
    last_adv_norm = None  # for histogram

    def softmax_from_logits(logits: torch.Tensor, mask: torch.Tensor, eta_: float) -> torch.Tensor:
        logits = logits.masked_fill(~mask, -torch.inf)
        return torch.softmax(eta_ * logits, dim=-1)

    def center_and_clip(logits: torch.Tensor, mask: torch.Tensor, thresh: float) -> torch.Tensor:
        # center per row over valid actions
        valid_counts = mask.sum(-1).clamp_min(1).unsqueeze(-1)
        masked_logits = torch.where(mask, logits, torch.zeros_like(logits))
        means = (masked_logits.sum(-1, keepdim=True) / valid_counts)
        centered = logits - means
        return centered.clamp(-thresh, thresh)

    def train_epoch():
        nonlocal steps
        nonlocal deployed_snapshots

        # Build online file list from drain buffer and select snapshot states
        from .common import drain_with_version
        dirname, data_ver = drain_with_version()
        if data_ver not in deployed_snapshots:
            logging.warning(
                f'[ACH] drained version v{data_ver} not found in local snapshots; '
                f'falling back to latest available'
            )
        file_list = [path.join(dirname, p) for p in os.listdir(dirname)]
        logging.info(f'[ACH] drained files: {len(file_list):,} (v{data_ver})')

        # Choose the matching snapshot by data_ver; if missing, pick the most recent
        if data_ver in deployed_snapshots:
            snapshot_states = deployed_snapshots[data_ver]
        else:
            sel_ver = max(deployed_snapshots.keys(), key=lambda v: (-1 if v is None else v))
            snapshot_states = deployed_snapshots[sel_ver]

        data_iter = AchFileDatasetsIter(
            version=version,
            file_list=file_list,
            pts=pts,
            snapshot_states=snapshot_states,
            device=device,
            file_batch_size=file_batch_size,
            reserve_ratio=reserve_ratio,
            player_names=['trainee'],
            num_epochs=num_epochs,
            enable_augmentation=enable_augmentation,
            augmented_first=augmented_first,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        # ACH iterator requires num_workers=0 to safely use GPU inside iterator
        loader = iter(DataLoader(
            dataset=data_iter,
            batch_size=batch_size,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        ))

        pb = tqdm(total=save_every, desc='ACH', initial=steps % save_every)

        def train_batch(obs, actions, masks, A_offline, G, logits_old):
            nonlocal steps, pb
            nonlocal last_adv_norm

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            A_offline = A_offline.to(dtype=torch.float32, device=device)
            G = G.to(dtype=torch.float32, device=device)
            logits_old = logits_old.to(dtype=torch.float32, device=device)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                # current logits and probs
                logits = policy(phi, masks)
                logits_centered = center_and_clip(logits, masks, logit_threshold)
                pi = softmax_from_logits(logits_centered, masks, eta)

                # old policy from provided logits
                logits_old_centered = center_and_clip(logits_old, masks, logit_threshold)
                pi_old = softmax_from_logits(logits_old_centered, masks, eta)

                pi_a = pi[torch.arange(pi.shape[0]), actions]
                pi_old_a = pi_old[torch.arange(pi_old.shape[0]), actions]
                ratio = (pi_a / (pi_old_a + 1e-8))

                # value and advantage
                V = value(phi)
                # Advantage from snapshot (GAE/MC): normalize per minibatch and clip
                A_raw = A_offline.detach()
                A_mean = A_raw.mean()
                A_std = A_raw.std(unbiased=False) + 1e-8
                A_norm = (A_raw - A_mean) / A_std
                clip_mask = (A_norm.abs() > adv_clip)
                A = A_norm.clamp(-adv_clip, adv_clip)

                # gating: ratio is symmetric; logit is directional by sign(A)
                logits_centered_a = logits_centered[torch.arange(logits_centered.shape[0]), actions]
                ratio_ok = (ratio >= (1.0 - ratio_clip)) & (ratio <= (1.0 + ratio_clip))
                # If A>=0, block only when logits are too high; if A<0, block only when too low
                logit_ok = torch.where(A >= 0,
                                       logits_centered_a < logit_threshold,
                                       logits_centered_a > -logit_threshold)
                gate = ratio_ok & logit_ok
                c = gate.to(logits.dtype)

                # losses
                pi_old_a_clipped = pi_old_a.clamp_min(1e-3)
                policy_loss = -(c * eta * (logits_centered_a / pi_old_a_clipped) * A).mean()
                value_loss = 0.5 * (V - G).pow(2).mean()
                # entropy over valid actions
                ent = -(pi.masked_fill(~masks, 0.0) * (pi.masked_fill(~masks, 1e-8)).log()).sum(-1).mean()
                # Follow ACH paper: add +beta * sum pi log pi  <=>  -beta * H(pi)
                loss = policy_loss + value_coef * value_loss - entropy_coef * ent

            scaler.scale(loss / max(1, opt_step_every)).backward()

            with torch.inference_mode():
                stats['policy_loss'] += policy_loss.detach()
                stats['value_loss'] += value_loss.detach()
                stats['entropy'] += ent.detach()
                stats['ratio_mean'] += ratio.mean().detach()
                stats['ratio_clip_rate'] += (1.0 - gate.float().mean()).detach()
                # Monitor normalized A (pre-clip)
                stats['adv_norm_mean'] += A_norm.mean().detach()
                stats['adv_norm_std'] += A_norm.std(unbiased=False).detach()
                stats['adv_clip_frac'] += clip_mask.float().mean().detach()
                # Keep the last normalized A for histogram logging
                last_adv_norm = A_norm.detach().to('cpu')

            steps += 1
            if steps % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pb.update(1)

            if steps % save_every == 0:
                pb.close()

                writer.add_scalar('ach/policy_loss', stats['policy_loss'] / save_every, steps)
                writer.add_scalar('ach/value_loss', stats['value_loss'] / save_every, steps)
                writer.add_scalar('ach/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('ach/ratio_mean', stats['ratio_mean'] / save_every, steps)
                writer.add_scalar('ach/ratio_clip_rate', stats['ratio_clip_rate'] / save_every, steps)
                # A (advantage) distribution monitoring (normalized, pre-clip)
                writer.add_scalar('ach/adv_norm_mean', stats['adv_norm_mean'] / save_every, steps)
                writer.add_scalar('ach/adv_norm_std', stats['adv_norm_std'] / save_every, steps)
                writer.add_scalar('ach/adv_clip_frac', stats['adv_clip_frac'] / save_every, steps)
                if last_adv_norm is not None:
                    writer.add_histogram('ach/adv_norm', last_adv_norm, steps)
                writer.flush()
                for k in stats:
                    stats[k] = 0.0

                # Save state
                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': policy.state_dict(),  # keep key for compatibility
                    'policy': policy.state_dict(),
                    'value': value.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                torch.save(state, state_file)

                # Evaluate
                if steps % test_every == 0 :
                    torch.cuda.empty_cache()
                    stat = test_player.test_play(test_games // 4, mortal, policy, device)
                    mortal.train(); policy.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135])
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                        # submit_param(mortal, policy, is_idle=False)
                        # logging.info('[ACH] param has been submitted (best)')

                # Continue training after test play; recreate progress bar
                pb = tqdm(total=save_every, desc='ACH')

        # iterate batches
        remaining = []
        total_seen = 0
        for batch in loader:
            bs = batch[0].shape[0]
            total_seen += bs
            if bs != batch_size:
                remaining.append(batch)
                continue
            train_batch(*batch)

        # process leftover
        # if remaining:
        if False:
            # concat and slice by batch_size
            obs = torch.cat([b[0] for b in remaining], 0)
            actions = torch.cat([b[1] for b in remaining], 0)
            masks = torch.cat([b[2] for b in remaining], 0)
            A_offline = torch.cat([b[3] for b in remaining], 0)
            G = torch.cat([b[4] for b in remaining], 0)
            logits_old = torch.cat([b[5] for b in remaining], 0)
            start = 0
            while start + batch_size <= obs.shape[0]:
                sl = slice(start, start + batch_size)
                train_batch(obs[sl], actions[sl], masks[sl], A_offline[sl], G[sl], logits_old[sl])
                start += batch_size
        pb.close()

        # End of epoch: submit the latest params for the next data collection window
        new_ver = submit_param(mortal, policy, is_idle=False)
        # Store snapshot keyed by the returned server version for future drains
        if new_ver not in deployed_snapshots:
            deployed_snapshots[new_ver] = (
                copy.deepcopy(mortal.state_dict()),
                copy.deepcopy(policy.state_dict()),
                copy.deepcopy(value.state_dict()),
            )
            # Optionally prune very old versions to limit memory
            if len(deployed_snapshots) > 8:
                # Keep the 8 most recent versions (None is treated as oldest)
                keys_sorted = sorted(
                    deployed_snapshots.keys(),
                    key=lambda v: (-1 if v is None else v)
                )
                for k in keys_sorted[:-8]:
                    if k in deployed_snapshots:
                        del deployed_snapshots[k]
        logging.info('[ACH] param has been submitted (epoch end); deployed snapshot updated')

    while True:
        train_epoch()
        gc.collect()
