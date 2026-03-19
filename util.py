import torch

def r2_score_func(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

class MetricTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.metrics = {}
    def update(self, name, val, n=1):
        if name not in self.metrics:
            self.metrics[name] = {"sum": 0.0, "count": 0}
        self.metrics[name]["sum"] += val * n
        self.metrics[name]["count"] += n
    def result(self):
        return {k: v["sum"] / v["count"] for k, v in self.metrics.items()}
