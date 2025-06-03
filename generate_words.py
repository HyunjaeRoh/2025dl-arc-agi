import numpy as np
import json
from typing import List, Dict
import random

MIN_SIZE=1
MAX_SIZE=10


def generate_center_dot(h, w, background, color):

    grid = np.full((h, w), background, dtype=int)

    center = ((h//2), (w//2))

    input_grid = grid.copy()
    output_grid = grid.copy()
    output_grid[center[0], center[1]] = color

    return {
        "input": input_grid.tolist(),
        "output": output_grid.tolist()
    }




def main():
    ...

if __name__ == "__main__":
    main()