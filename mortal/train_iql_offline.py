def train(is_first_run: bool = False):
    from . import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    from copy import deepcopy
    from datetime import datetime
    from glob import glob
    from itertools import chain
    from os import path

    import torch
    from torch import nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from .common import filtered_trimmed_lines, parameter_count, tqdm
    from .config import config
    from .dataloader_iql_offline import IQLOfflineDataset, worker_init_fn_iql_offline
    from .iql_heads import IQLQHead, IQLValueHead
    from .model import Brain
    from .player import TestPlayer
    from .policy_ach import PolicyNet
    from .libriichi.consts import obs_shape

    online = config['control']['online']
    assert not online, 'IQL offline trainer expects control.online = false'

    version = config['control']['version']
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    iql_cfg = config.get('iql', {})
    batch_size = int(iql_cfg.get('batch_size', config['control']['batch_size']))
    expectile = float(iql_cfg.get('expectile', 0.7))
    beta = float(iql_cfg.get('beta', 3.0))
    adv_clip = float(iql_cfg.get('adv_clip', 5.0))
    weight_clip = float(iql_cfg.get('weight_clip', 50.0))
    policy_coef = float(iql_cfg.get('policy_coef', 1.0))
    value_coef = float(iql_cfg.get('value_coef', 1.0))
    q_coef = float(iql_cfg.get('q_coef', 1.0))
    entropy_coef = float(iql_cfg.get('entropy_coef', 0.0))
    target_tau = float(iql_cfg.get('target_update_tau', 0.005))
    target_period = int(iql_cfg.get('target_update_period', 1))

    opt_step_every = int(config['control']['opt_step_every'])
    save_every = int(config['control']['save_every'])
    test_every = int(config['control']['test_every'])
    test_games = int(config['test_play']['games'])

    pts = config['env']['pts']
    gamma = float(config['env'].get('gamma', 0.995))
    file_batch_size = int(config['dataset']['file_batch_size'])
    reserve_ratio = float(config['dataset']['reserve_ratio'])
    sample_ratio = float(config['dataset'].get('sample_ratio', 1.0))
    num_workers = int(config['dataset']['num_workers'])
    num_epochs = int(config['dataset']['num_epochs'])
    enable_augmentation = bool(config['dataset']['enable_augmentation'])
    augmented_first = bool(config['dataset']['augmented_first'])

    lr = float(config['optim'].get('lr', 2.5e-4))
    weight_decay = float(config['optim'].get('weight_decay', 0.0))
    eps = float(config['optim'].get('eps', 1e-8))
    betas = tuple(config['optim'].get('betas', [0.9, 0.999]))
    max_grad_norm = float(config['optim'].get('max_grad_norm', 0.0))

    mortal = Brain(version=version, **config['resnet']).to(device)
    policy = PolicyNet().to(device)
    value_head = IQLValueHead().to(device)
    q_head = IQLQHead().to(device)

    mortal_target = deepcopy(mortal).to(device)
    value_target = deepcopy(value_head).to(device)

    for target in (mortal_target, value_target):
        target.eval()
        for param in target.parameters():
            param.requires_grad_(False)

    if enable_compile:
        mortal.compile()
        policy.compile()
        value_head.compile()
        q_head.compile()

    logging.info(f'[IQL] version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy):,}')
    logging.info(f'value params: {parameter_count(value_head):,}')
    logging.info(f'q params: {parameter_count(q_head):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    def split_decay_params(model: nn.Module):
        decay_params, no_decay_params = [], []
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        return decay_params, no_decay_params

    decay, no_decay = [], []
    for module in (mortal, policy, value_head, q_head):
        d, nd = split_decay_params(module)
        decay.extend(d)
        no_decay.extend(nd)

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.0, betas=betas, eps=eps)
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        if 'policy' in state:
            policy.load_state_dict(state['policy'])
        if 'value_head' in state:
            value_head.load_state_dict(state['value_head'])
        if 'q_head' in state:
            q_head.load_state_dict(state['q_head'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        steps = state.get('steps', 0)
        best_perf = state.get('best_perf', best_perf)

        mortal_target.load_state_dict(mortal.state_dict())
        value_target.load_state_dict(value_head.state_dict())

    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'q_loss': 0.0,
        'entropy': 0.0,
        'adv': 0.0,
    }

    policy.policy_mode = 'iql'

    def polyak_update(source: nn.Module, target: nn.Module, tau: float):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.lerp_(src_p.data, tau)

    def encode_features(model: Brain, obs: torch.Tensor):
        out = model(obs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def build_file_list():
        player_names = []
        player_names_set = set()
        for filename in config['dataset']['player_names_files']:
            with open(filename) as f:
                player_names_set.update(filtered_trimmed_lines(f))
        player_names = list(player_names_set)
        if len(player_names) > 0:
            logging.info(f'loaded {len(player_names):,} players for filtering')

        file_index = config['dataset']['file_index']
        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            file_list = index['file_list']
        else:
            logging.info('building file index...')
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            if len(player_names_set) > 0:
                filtered = []
                for filename in tqdm(file_list, unit='file'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)

        logging.info(f'[IQL] file list size: {len(file_list):,}')
        return player_names, file_list

    def train_epoch():
        nonlocal steps

        player_names, file_list = build_file_list()
        if num_workers > 1:
            import random as pyrandom
            pyrandom.shuffle(file_list)

        dataset = IQLOfflineDataset(
            version=version,
            file_list=file_list,
            pts=pts,
            file_batch_size=file_batch_size,
            reserve_ratio=reserve_ratio,
            sample_ratio=sample_ratio,
            player_names=player_names,
            num_epochs=num_epochs,
            enable_augmentation=enable_augmentation,
            augmented_first=augmented_first,
            gamma=gamma,
        )

        loader = iter(DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn_iql_offline,
        ))

        pb = tqdm(total=save_every, desc='IQL', initial=steps % save_every)

        def train_batch(obs, next_obs, actions, masks, next_masks, rewards, discounts, dones):
            nonlocal steps
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            next_obs = next_obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            next_masks = next_masks.to(dtype=torch.bool, device=device)
            rewards = rewards.to(dtype=torch.float32, device=device)
            discounts = discounts.to(dtype=torch.float32, device=device)
            dones = dones.to(dtype=torch.bool, device=device)

            not_done = (~dones).to(dtype=torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = encode_features(mortal, obs)
                phi_next = encode_features(mortal_target, next_obs)

                V = value_head(phi)
                Q_all = q_head(phi)
                Q_sa = Q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    V_next = value_target(phi_next)
                    q_target = rewards + discounts * not_done * V_next

                delta = q_target.detach() - V
                weight = torch.where(delta.detach() < 0, 1 - expectile, expectile)
                value_loss = (weight * delta.pow(2)).mean()

                q_loss = 0.5 * (Q_sa - q_target.detach()).pow(2).mean()

                logits = policy(phi, masks)
                safe_fill = torch.full_like(logits, -1e4)
                safe_logits = torch.where(masks, logits, safe_fill)
                log_probs = torch.log_softmax(safe_logits, dim=-1)
                probs = torch.softmax(safe_logits, dim=-1)
                log_pi_a = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                adv = (Q_sa.detach() - V.detach()).clamp(-adv_clip, adv_clip)
                weights = torch.exp(beta * adv).clamp(max=weight_clip)
                policy_loss = -(weights * log_pi_a).mean()

                entropy = -(probs * log_probs).sum(-1).mean()

                loss = (
                    policy_coef * policy_loss
                    + value_coef * value_loss
                    + q_coef * q_loss
                    - entropy_coef * entropy
                )

            scaler.scale(loss / max(1, opt_step_every)).backward()

            with torch.inference_mode():
                stats['policy_loss'] += policy_loss.detach()
                stats['value_loss'] += value_loss.detach()
                stats['q_loss'] += q_loss.detach()
                stats['entropy'] += entropy.detach()
                stats['adv'] += adv.mean().detach()

            steps += 1
            if steps % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if steps % target_period == 0:
                    polyak_update(mortal, mortal_target, target_tau)
                    polyak_update(value_head, value_target, target_tau)

            pb.update(1)

            if steps % save_every == 0:
                pb.close()

                writer.add_scalar('iql/policy_loss', stats['policy_loss'] / save_every, steps)
                writer.add_scalar('iql/value_loss', stats['value_loss'] / save_every, steps)
                writer.add_scalar('iql/q_loss', stats['q_loss'] / save_every, steps)
                writer.add_scalar('iql/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('iql/advantage', stats['adv'] / save_every, steps)
                writer.add_scalar('hparam/lr', optimizer.param_groups[0]['lr'], steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0.0

                state = {
                    'mortal': mortal.state_dict(),
                    'policy': policy.state_dict(),
                    'value_head': value_head.state_dict(),
                    'q_head': q_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                torch.save(state, state_file)

                if steps % test_every == 0:
                    torch.cuda.empty_cache()
                    stat = test_player.test_play(test_games // 4, mortal, policy, device)
                    mortal.train(); policy.train(); value_head.train(); q_head.train()

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

                pb = tqdm(total=save_every, desc='IQL')

        for batch in loader:
            if batch[0].shape[0] != batch_size:
                continue
            train_batch(*batch)

        pb.close()

    while True:
        train_epoch()
        gc.collect()
        break


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
