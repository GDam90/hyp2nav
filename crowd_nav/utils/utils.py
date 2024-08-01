import os
def select_last_checkpoint(model_dir):
    base_name = 'rl_model_'
    ext = '.pth'
    candidates = [f for f in os.listdir(model_dir) if f.startswith(base_name) and f.endswith(ext)]
    numbers = [int(f.split(base_name)[1].split(ext)[0]) for f in candidates]
    best_number = max(numbers)
    return str(best_number)