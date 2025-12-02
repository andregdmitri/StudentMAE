import torch
from ptflops import get_model_complexity_info

@torch.no_grad()
def compute_flops(model, img_size):
    """
    Computes FLOPs for a Lightning model's forward() using ptflops.
    Returns (flops, params).
    """
    # Use a dummy input with shape (1, 3, H, W)
    input_res = (3, img_size, img_size)

    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    flops = 2 * macs   # ptflops computes MACs → FLOPs = 2*MACs

    return flops, params