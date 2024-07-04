import numpy as np
import torch

def coords2heatmap(coords, img_size=256):
    """
    Convert coordinates to heatmap
    Args:
        coords: (N, 2) numpy array
    Returns:
        heatmap: (H, W) numpy array
    """
    bs = coords.shape[0]
    heatmap = torch.zeros(bs, img_size, img_size, dtype=torch.float64)
    for i in range(bs):
        for coord in coords[i]:
            x, y = coord*img_size
            heatmap[i, round(x.item()), round(y.item())] = 1
    return heatmap
