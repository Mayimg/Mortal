"""
Mortal Training Client

This client is part of a distributed training system for the Mortal mahjong AI.
It operates in an infinite loop performing the following tasks:

1. Connects to a remote parameter server to fetch the latest neural network parameters
2. Runs self-play games using the updated models (Brain and DQN networks)
3. Collects game statistics (rankings, points) and tracks performance over time
4. Submits generated game logs back to the server for centralized training

The client enables scalable, distributed training where multiple clients can generate
training data in parallel while a central server coordinates parameter updates.

Key components:
- Brain: ResNet-based feature extraction network for encoding game states
- DQN: Deep Q-Network for action-value estimation and decision making
- TrainPlayer: Handles the actual game playing with exploration strategies
"""

# Import prelude module - sets up logging and other initialization
import prelude

# Standard library imports
import logging  # For logging info and debug messages
import socket   # For TCP socket communication with parameter server
import torch    # PyTorch for neural network operations
import numpy as np  # For numerical arrays and calculations
import time     # For sleep delays between connection attempts
import gc       # For manual garbage collection
from os import path  # For file path operations

# Local module imports
from model import Brain, DQN  # Neural network models: Brain (encoder) and DQN (action values)
from player import TrainPlayer  # Manages self-play game execution
from common import send_msg, recv_msg  # Socket communication utilities
from config import config  # Configuration loaded from config.toml

def main():
    """
    Main function that runs the training client in an infinite loop.
    Manages the cycle of fetching parameters, playing games, and submitting results.
    """
    # Extract server connection details from config (host and port for parameter server)
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    
    # Set the compute device (CPU or CUDA GPU) based on config
    device = torch.device(config['control']['device'])
    
    # Model architecture parameters from config
    version = config['control']['version']  # Model version (e.g., '3.0')
    num_blocks = config['resnet']['num_blocks']  # Number of ResNet blocks in Brain
    conv_channels = config['resnet']['conv_channels']  # Number of convolutional channels

    # Initialize the Brain model (feature encoder)
    # - version: determines input/output format
    # - num_blocks & conv_channels: control model capacity
    # - .to(device): move to GPU/CPU
    # - .eval(): set to evaluation mode (no dropout, fixed batch norm)
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    
    # Initialize the DQN model (action-value estimator)
    # - version: must match Brain version for compatibility
    # - .to(device): move to same device as Brain
    dqn = DQN(version=version).to(device)
    
    # Optional: Compile models using torch.compile for faster inference
    # This can improve performance but may increase startup time
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    # Create TrainPlayer instance to manage game execution
    train_player = TrainPlayer()
    
    # Track the version of parameters we currently have
    # -1 indicates we haven't received any parameters yet
    param_version = -1

    # Points awarded for each ranking position in mahjong
    # 1st: +90, 2nd: +45, 3rd: 0, 4th: -135
    # These are typical tournament scoring values
    pts = np.array([90, 45, 0, -135])
    
    # Number of recent sessions to include in moving average
    history_window = config['online']['history_window']
    
    # List to store ranking distributions from recent sessions
    # Used to calculate moving averages for stable performance tracking
    history = []

    # Main training loop - runs indefinitely
    while True:
        # Inner loop to fetch latest parameters from server
        # Retries until successful parameter update is received
        while True:
            # Create a new TCP socket connection (auto-closed by context manager)
            with socket.socket() as conn:
                # Connect to the parameter server
                conn.connect(remote)
                
                # Prepare request message
                # - type: 'get_param' indicates we want to fetch parameters
                # - param_version: current version we have (server only sends if newer available)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                
                # Send request using custom protocol (8-byte header + torch serialized data)
                send_msg(conn, msg)
                
                # Receive response from server
                # - map_location=device ensures tensors are loaded to correct device
                rsp = recv_msg(conn, map_location=device)
                
                # Check if server has newer parameters
                if rsp['status'] == 'ok':
                    # Update our version number to match server
                    param_version = rsp['param_version']
                    # Exit retry loop - we have new parameters
                    break
                
                # No new parameters available - wait 3 seconds before retrying
                # This prevents overwhelming the server with requests
                time.sleep(3)
        
        # Load the new parameters into our models
        # state_dict contains all weights and biases for the neural network
        mortal.load_state_dict(rsp['mortal'])  # Load Brain model parameters
        dqn.load_state_dict(rsp['dqn'])        # Load DQN model parameters
        
        # Log that parameters have been successfully updated
        logging.info('param has been updated')

        # Run self-play games using the updated models
        # train_play runs multiple games where the trainee model plays against baseline models
        # Returns:
        # - rankings: array of counts [1st_place_count, 2nd_place_count, 3rd_place_count, 4th_place_count]
        # - file_list: list of file paths containing game logs
        rankings, file_list = train_player.train_play(mortal, dqn, device)
        
        # Calculate average rank for this session
        # @ is matrix multiplication: [count1, count2, count3, count4] @ [1, 2, 3, 4]
        # Gives weighted average rank (1.0 = always first, 4.0 = always last)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        
        # Calculate average points for this session
        # Similar calculation but using point values instead of ranks
        avg_pt = rankings @ pts / rankings.sum()

        # Add current session's rankings to history for moving average
        history.append(np.array(rankings))
        
        # Maintain sliding window of recent sessions
        # Remove oldest entry if we exceed the window size
        if len(history) > history_window:
            del history[0]
        
        # Calculate cumulative rankings over all sessions in history
        # axis=0 sums element-wise across all ranking arrays
        sum_rankings = np.sum(history, axis=0)
        
        # Calculate moving average rank over history window
        # This provides a more stable performance metric
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        
        # Calculate moving average points over history window
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        # Log performance metrics for this session
        # Format: [1st, 2nd, 3rd, 4th] counts (average_rank, average_points)
        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        
        # Log moving average performance over recent sessions
        # Provides insight into long-term performance trends
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        # Prepare game logs for submission to server
        # Dictionary mapping filename -> file contents
        logs = {}
        
        # Read all generated game log files
        for filename in file_list:
            # Open file in binary mode (logs are likely compressed or binary format)
            with open(filename, 'rb') as f:
                # Use just the filename (not full path) as the key
                # This makes the logs portable across different client machines
                logs[path.basename(filename)] = f.read()

        # Submit game logs back to the parameter server
        with socket.socket() as conn:
            # Connect to the parameter server again
            conn.connect(remote)
            
            # Send logs with submission message
            send_msg(conn, {
                'type': 'submit_replay',  # Message type for log submission
                'logs': logs,             # Dictionary of filename -> log data
                'param_version': param_version,  # Version of params used to generate these logs
            })
            
            # Confirm logs were sent successfully
            logging.info('logs have been submitted')
        
        # Memory cleanup operations
        # Important for long-running processes to prevent memory leaks
        
        # Force Python garbage collection
        # Ensures temporary objects are freed promptly
        gc.collect()
        
        # Clear PyTorch's CUDA memory cache
        # Releases GPU memory that's no longer needed
        torch.cuda.empty_cache()
        
        # Wait for all CUDA operations to complete
        # Ensures GPU is ready for next iteration
        torch.cuda.synchronize()

# Entry point: Only execute if this file is run directly (not imported)
if __name__ == '__main__':
    try:
        # Run the main training loop
        main()
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C interruption
        # Pass silently without error message - expected way to stop the client
        pass
