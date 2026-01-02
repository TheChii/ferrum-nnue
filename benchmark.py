#!/usr/bin/env python3
"""
NNUE Inference Benchmark Script
Tests evaluations/second for the nnue-rs library.

Usage:
    python benchmark.py                    # Run release benchmark
    python benchmark.py --iterations 100000  # Custom iteration count
    python benchmark.py --debug            # Debug build
"""

import subprocess
import os
import sys
import argparse


def run_benchmark(iterations: int = 50000, release: bool = True):
    """Build and run the Rust NNUE benchmark."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build first
    print("Building benchmark...")
    build_cmd = ["cargo", "build", "--test", "benchmark"]
    if release:
        build_cmd.append("--release")
    
    result = subprocess.run(build_cmd, cwd=script_dir)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)
    
    # Set environment and run
    env = os.environ.copy()
    env["BENCH_ITERATIONS"] = str(iterations)
    
    run_cmd = ["cargo", "test", "--test", "benchmark"]
    if release:
        run_cmd.append("--release")
    run_cmd.extend(["--", "--ignored", "--nocapture"])
    
    subprocess.run(run_cmd, cwd=script_dir, env=env)


def main():
    parser = argparse.ArgumentParser(description="Benchmark NNUE inference speed")
    parser.add_argument("-n", "--iterations", type=int, default=50000,
                        help="Evaluations per position (default: 50000)")
    parser.add_argument("--debug", action="store_true", 
                        help="Use debug build instead of release")
    
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, release=not args.debug)


if __name__ == "__main__":
    main()
