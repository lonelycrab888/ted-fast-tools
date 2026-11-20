"""Runtime combinations exercised with the Merizo CUDA fast paths."""
import torch


def supports_fast_path(device):
    if device.type != "cuda":
        return False
    if torch.__version__.split('+')[0] == '2.0.1':
        return True
    return torch.__version__ == '2.7.1+cu128' and torch.cuda.get_device_capability(device) == (12, 0)
