import torch


def GET_DEVICE(DEVICE=0):
    # Selecting training device, -1 for cpu
    if DEVICE == -1:
        return "cpu"
    elif torch.cuda.is_available():
        device_name = f"cuda:{DEVICE}"
        if torch.cuda.device_count() > DEVICE:
            return device_name
        print(f"No such cuda device: {DEVICE}")
        return "cpu"
