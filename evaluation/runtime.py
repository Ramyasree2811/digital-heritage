import time
import torch

def measure_inference_speed(model, input_shape=(1, 3, 256, 256), device="cpu", num_runs=20, warmup_runs=5):
    """
    Measures the inference speed (latency) of a PyTorch model.
    Returns: average runtime in milliseconds, std of runtime in milliseconds
    """
    model.eval()
    model.to(device)
    
    x = torch.randn(input_shape).to(device)
    
    # Warmup runs (to compile/load CUDA/CPU caches)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            
    # Measure runs
    runtimes = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(x)
            end_time = time.perf_counter()
            runtimes.append((end_time - start_time) * 1000.0)  # Convert to ms
            
    avg_time = float(time.mean(runtimes) if hasattr(time, 'mean') else np.mean(runtimes) if 'np' in globals() else sum(runtimes)/len(runtimes))
    # Let's import numpy inside or calculate mean manually
    import numpy as np
    avg_time = float(np.mean(runtimes))
    std_time = float(np.std(runtimes))
    
    return avg_time, std_time

if __name__ == "__main__":
    import torch.nn as nn
    dummy_model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 3, padding=1))
    avg, std = measure_inference_speed(dummy_model)
    print(f"Inference Latency: {avg:.2f} ms +/- {std:.2f} ms")
