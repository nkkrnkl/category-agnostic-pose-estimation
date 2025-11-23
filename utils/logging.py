"""
Logging utilities for training and evaluation.
"""
import os
import json
import time
from collections import defaultdict
from datetime import datetime


class MetricLogger:
    """Logger for tracking training and evaluation metrics."""

    def __init__(self, log_dir='logs', experiment_name=None):
        """
        Initialize metric logger.

        Args:
            log_dir: directory to save logs
            experiment_name: name of the experiment (default: timestamp)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")

        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(float)
        self.epoch_counts = defaultdict(int)

        self.start_time = time.time()

    def log(self, metrics_dict, step=None, prefix=''):
        """
        Log metrics.

        Args:
            metrics_dict: dict of metric_name -> value
            step: optional step number (epoch, iteration, etc.)
            prefix: optional prefix for metric names (e.g., 'train/', 'val/')
        """
        for key, value in metrics_dict.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, (int, float)):
                self.metrics[full_key].append({
                    'step': step,
                    'value': float(value),
                    'time': time.time() - self.start_time
                })

    def update_epoch(self, metrics_dict, prefix=''):
        """
        Accumulate metrics for averaging at the end of an epoch.

        Args:
            metrics_dict: dict of metric_name -> value
            prefix: optional prefix for metric names
        """
        for key, value in metrics_dict.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, (int, float)):
                self.epoch_metrics[full_key] += float(value)
                self.epoch_counts[full_key] += 1

    def reset_epoch(self):
        """Reset epoch metrics."""
        self.epoch_metrics.clear()
        self.epoch_counts.clear()

    def get_epoch_metrics(self):
        """
        Get averaged epoch metrics.

        Returns:
            dict of metric_name -> averaged value
        """
        averaged = {}
        for key in self.epoch_metrics:
            if self.epoch_counts[key] > 0:
                averaged[key] = self.epoch_metrics[key] / self.epoch_counts[key]
        return averaged

    def print_epoch(self, epoch, metrics_dict=None):
        """
        Print epoch summary.

        Args:
            epoch: epoch number
            metrics_dict: optional dict of metrics to print (if None, use epoch metrics)
        """
        if metrics_dict is None:
            metrics_dict = self.get_epoch_metrics()

        elapsed = time.time() - self.start_time
        print(f"\nEpoch {epoch} | Time: {elapsed:.1f}s")

        for key, value in sorted(metrics_dict.items()):
            print(f"  {key}: {value:.6f}")

    def save(self):
        """Save all metrics to JSON file."""
        output = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'metrics': dict(self.metrics)
        }

        with open(self.log_file, 'w') as f:
            json.dump(output, f, indent=2)

    def load(self, log_file):
        """
        Load metrics from JSON file.

        Args:
            log_file: path to log file
        """
        with open(log_file, 'r') as f:
            data = json.load(f)

        self.experiment_name = data['experiment_name']
        self.metrics = defaultdict(list, data['metrics'])


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class ProgressTracker:
    """Track progress during training."""

    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()

    def update(self, epoch=None, step=None):
        if epoch is not None:
            self.current_epoch = epoch
        if step is not None:
            self.current_step = step

    def get_progress(self):
        """Get current progress as percentage."""
        total_steps = self.total_epochs * self.steps_per_epoch
        current_steps = self.current_epoch * self.steps_per_epoch + self.current_step
        return 100.0 * current_steps / total_steps if total_steps > 0 else 0

    def get_eta(self):
        """Get estimated time to completion."""
        elapsed = time.time() - self.start_time
        progress = self.get_progress() / 100.0

        if progress > 0:
            total_time = elapsed / progress
            eta = total_time - elapsed
            return eta
        else:
            return 0

    def format_time(self, seconds):
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def print_status(self):
        """Print current training status."""
        progress = self.get_progress()
        eta = self.get_eta()
        elapsed = time.time() - self.start_time

        print(f"Epoch {self.current_epoch}/{self.total_epochs} | "
              f"Step {self.current_step}/{self.steps_per_epoch} | "
              f"Progress: {progress:.1f}% | "
              f"Elapsed: {self.format_time(elapsed)} | "
              f"ETA: {self.format_time(eta)}")


def print_model_summary(model, input_shapes=None):
    """
    Print model summary.

    Args:
        model: PyTorch model
        input_shapes: optional dict of input_name -> shape
    """
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    if input_shapes:
        print("\nExpected input shapes:")
        for name, shape in input_shapes.items():
            print(f"  {name}: {shape}")

    print("=" * 80 + "\n")
