#!/usr/bin/env python
"""
Create a .pth model file with gradient checkpointing enabled for the ResNet model.
This creates models/checkpoint_baseline.pth with initial (untrained) weights.
"""

import torch
import os
from datetime import datetime
from mortal.model import Brain, DQN, AuxNet
from mortal.config import config

def main():
    # Model configuration from config.toml
    version = config['control']['version']
    resnet_config = config['resnet']
    
    # Create models with checkpoint parameters
    mortal = Brain(
        version=version,
        conv_channels=resnet_config['conv_channels'],
        num_blocks=resnet_config['num_blocks'],
        use_checkpoint=resnet_config['use_checkpoint'],
        checkpoint_segments=resnet_config['checkpoint_segments']
    )
    
    # Create DQN and AuxNet models
    dqn = DQN(version=version)
    aux_net = AuxNet((4,))
    
    # Create the state dictionary in the same format as used in training
    state = {
        'mortal': mortal.state_dict(),
        'current_dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'steps': 0,
        'timestamp': datetime.now().timestamp(),
        'config': config,
        'best_perf': {
            'avg_rank': 4.,
            'avg_pt': -135.,
            'stable_rank': 4.,
        }
    }
    
    # Save the model
    output_path = 'models/checkpoint_baseline.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(state, output_path)
    
    print(f"Model saved to: {output_path}")
    print(f"ResNet configuration:")
    print(f"  - version: {version}")
    print(f"  - conv_channels: {resnet_config['conv_channels']}")
    print(f"  - num_blocks: {resnet_config['num_blocks']}")
    print(f"  - use_checkpoint: {resnet_config['use_checkpoint']}")
    print(f"  - checkpoint_segments: {resnet_config['checkpoint_segments']}")
    print(f"Model contains:")
    print(f"  - mortal (Brain) parameters: {sum(p.numel() for p in mortal.parameters()):,}")
    print(f"  - dqn parameters: {sum(p.numel() for p in dqn.parameters()):,}")
    print(f"  - aux_net parameters: {sum(p.numel() for p in aux_net.parameters()):,}")

if __name__ == '__main__':
    main()