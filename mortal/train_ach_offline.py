def train(is_first_run: bool = False):
    from . import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
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

    from .common import submit_param, parameter_count, filtered_trimmed_lines, tqdm
    from .player import TestPlayer
    from .dataloader_ach_offline import AchOfflineFileDatasetsIter, worker_init_fn_ach_offline
    from .model import Brain
    from .policy_ach import PolicyNet, ValueHead
    from .libriichi.consts import obs_shape
    from .config import config

    online = config['control']['online']
    assert not online, 'ACH offline trainer expects control.online = false'

    version = config['control']['version']
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    ach_cfg = config.get('ach', {})
    eta = float(ach_cfg.get('eta', 1.0))
    logit_threshold = float(ach_cfg.get('logit_threshold', 6.0))
    entropy_coef = float(ach_cfg.get('entropy_coef', 1e-2))
    value_coef = float(ach_cfg.get('value_coef', 0.5))
    batch_size = int(ach_cfg.get('batch_size', config['control']['batch_size']))

    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    test_games = config['test_play']['games']

    pts = config['env']['pts']
    gamma = float(config['env'].get('gamma', 0.995))
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    sample_ratio = float(config['dataset'].get('sample_ratio', 1.0))
    num_workers = int(config['dataset']['num_workers'])
    num_epochs = int(config['dataset']['num_epochs'])
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

    logging.info(f'[ACH-OFFLINE] version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy):,}')
    logging.info(f'value params: {parameter_count(value):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

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
        decay.extend(d)
        no_decay.extend(nd)

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
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        # Compatibility with DQN key used elsewhere
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

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'acc': 0.0,
    }

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

    def build_file_list_offline():
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

        logging.info(f'[ACH-OFFLINE] file list size: {len(file_list):,}')
        return player_names, file_list

    def train_epoch():
        nonlocal steps

        player_names, file_list = build_file_list_offline()
        if num_workers > 1:
            random_state = torch.randint(0, 1 << 31, (1,)).item()  # just for logging variety
            logging.info(f'[ACH-OFFLINE] shuffling files (seed hint: {random_state})')
            import random
            random.shuffle(file_list)

        data_iter = AchOfflineFileDatasetsIter(
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
            dataset=data_iter,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn_ach_offline,
        ))

        pb = tqdm(total=save_every, desc='ACH-OFFLINE', initial=steps % save_every)

        def train_batch(obs, actions, masks, G):
            nonlocal steps, pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            G = G.to(dtype=torch.float32, device=device)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                logits = policy(phi, masks)
                logits_centered = center_and_clip(logits, masks, logit_threshold)
                pi = softmax_from_logits(logits_centered, masks, eta)

                # Cross-entropy with one-hot action over valid actions only
                pi_a = pi[torch.arange(pi.shape[0]), actions]
                policy_loss = -(pi_a.clamp_min(1e-8).log()).mean()

                V = value(phi)
                value_loss = 0.5 * (V - G).pow(2).mean()

                # Entropy over valid actions
                ent = -(pi.masked_fill(~masks, 0.0) * (pi.masked_fill(~masks, 1e-8).log())).sum(-1).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * ent

            scaler.scale(loss / max(1, opt_step_every)).backward()

            with torch.inference_mode():
                stats['policy_loss'] += policy_loss.detach()
                stats['value_loss'] += value_loss.detach()
                stats['entropy'] += ent.detach()
                # Simple accuracy on valid action argmax
                pred = pi.argmax(dim=-1)
                stats['acc'] += (pred == actions).float().mean()

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

                writer.add_scalar('ach_offline/policy_loss', stats['policy_loss'] / save_every, steps)
                writer.add_scalar('ach_offline/value_loss', stats['value_loss'] / save_every, steps)
                writer.add_scalar('ach_offline/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('ach_offline/acc', stats['acc'] / save_every, steps)
                writer.add_scalar('hparam/lr', optimizer.param_groups[0]['lr'], steps)
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
                if steps % test_every == 0:
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

                pb = tqdm(total=save_every, desc='ACH-OFFLINE')

        remaining = []
        for batch in loader:
            bs = batch[0].shape[0]
            if bs != batch_size:
                remaining.append(batch)
                continue
            train_batch(*batch)

        # Optionally process leftover by slicing if desired; keep disabled by default
        if False and remaining:
            obs = torch.cat([b[0] for b in remaining], 0)
            actions = torch.cat([b[1] for b in remaining], 0)
            masks = torch.cat([b[2] for b in remaining], 0)
            G = torch.cat([b[3] for b in remaining], 0)
            start = 0
            while start + batch_size <= obs.shape[0]:
                sl = slice(start, start + batch_size)
                train_batch(obs[sl], actions[sl], masks[sl], G[sl])
                start += batch_size
        pb.close()

    while True:
        train_epoch()
        gc.collect()
        # Offline: run one epoch for easier control
        break


if __name__ == '__main__':
    import os
    import sys
    from subprocess import Popen

    # When run as a module, just invoke train() directly
    try:
        train()
    except KeyboardInterrupt:
        pass

