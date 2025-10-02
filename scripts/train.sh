#!/bin/bash
# Simple shell script to launch training with configs.

# Example usage:
# bash scripts/train.sh data=sisfall model=lstm_ae trainer=base

python main.py "$@"
