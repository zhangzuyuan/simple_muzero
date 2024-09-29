


import json
import sys

import ray
from algorithm.muzero.muzero import MuZero







if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Train directly with: python main.py muzero cartpole
        if sys.argv[1] == 'muzero':
            muzero = MuZero(sys.argv[2])
            muzero.train()
    elif len(sys.argv) == 4:
        # Resume training with: python main.py muzero cartpole '{"lr_init": 0.01}'
        if sys.argv[1] == 'muzero':
            config = json.loads(sys.argv[3])
            muzero = MuZero(sys.argv[2], config)
            muzero.train()
    ray.shutdown()
