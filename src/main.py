#!/usr/bin/env python3
"""
Main execution file for Quantum vs Classical Random Walk Simulation
"""

import argparse
from quantum_walk import QuantumWalk, ClassicalWalk
from visualization import animate_quantum_walk

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantum vs Classical Random Walk Simulation')
    parser.add_argument('--grid-size', type=int, default=4,
                       help='Size of the grid (default: 4)')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of steps to simulate (default: 20)')
    parser.add_argument('--interval', type=int, default=800,
                       help='Animation interval in milliseconds (default: 800)')
    parser.add_argument('--stop-at-completion', action='store_true',
                       help='Stop animation when both walks complete')
    return parser.parse_args()

def print_simulation_info(args):
    """Print information about the simulation configuration."""
    print("\n=== Quantum vs Classical Random Walk Simulation ===")
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Maximum Steps: {args.steps}")
    print(f"Animation Interval: {args.interval}ms")
    print(f"Stop at Completion: {'Yes' if args.stop_at_completion else 'No'}")
    print("=" * 45 + "\n")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Print simulation configuration
    print_simulation_info(args)
    
    # Create quantum and classical walk instances
    print("Initializing quantum and classical walks...")
    qw = QuantumWalk(args.grid_size)
    cw = ClassicalWalk(args.grid_size)
    
    print("Creating animation...")
    try:
        # Create and display animation
        animation = animate_quantum_walk(
            qw, cw,
            steps=args.steps,
            grid_size=args.grid_size,
            interval=args.interval,
            stop_at_completion=args.stop_at_completion
        )
        
        print("\nAnimation controls:")
        print("- Play/Pause: Toggle animation")
        print("- << : Go to previous step")
        print("- >> : Go to next step")
        print("- O  : Reset to beginning")
        print("\nClose the window to exit the simulation.")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
    
if __name__ == '__main__':
    main() 