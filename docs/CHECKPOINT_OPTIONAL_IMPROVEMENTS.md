# Checkpoint System - Optional Improvements

This document lists optional enhancements to the checkpointing system that were **NOT implemented** but could improve robustness, efficiency, or usability in the future.

---

## 1. Atomic Checkpoint Writing

**Status**: ❌ Not Implemented

**Description**: Currently, checkpoints are saved directly to their final filename using `torch.save()`. If the process crashes or is killed during the save operation, the checkpoint file will be corrupted (partially written).

**Proposed Implementation**:
```python
import tempfile
import shutil

def save_checkpoint_atomic(checkpoint_dict, checkpoint_path):
    """Save checkpoint atomically using temp file + rename."""
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(
        delete=False, 
        dir=os.path.dirname(checkpoint_path),
        suffix='.tmp'
    ) as tmp:
        torch.save(checkpoint_dict, tmp.name)
        tmp_path = tmp.name
    
    # Atomic rename (replaces existing file if present)
    shutil.move(tmp_path, checkpoint_path)
```

**Benefits**:
- Prevents corrupted checkpoints from crashes during save
- Ensures checkpoint is either fully written or not present
- No partial/broken checkpoint files

**Tradeoffs**:
- Requires 2× disk space during save (temp + final)
- Slightly more complex code

**Priority**: Low (checkpoint corruption is rare with modern filesystems)

---

## 2. Automatic Checkpoint Cleanup (Keep Last N)

**Status**: ❌ Not Implemented

**Description**: Training for 300 epochs creates 300+ checkpoint files (~150GB disk usage). Old regular checkpoints (not best-*) are rarely needed after training completes.

**Proposed Implementation**:
```python
def cleanup_old_checkpoints(output_dir, keep_last=5):
    """
    Keep only the last N regular checkpoints, delete older ones.
    Never deletes best-loss or best-PCK checkpoints.
    """
    # Find all regular epoch checkpoints (not best-*)
    checkpoints = sorted(
        Path(output_dir).glob('checkpoint_e*.pth'),
        key=lambda x: x.stat().st_mtime  # Sort by modification time
    )
    
    # Exclude best checkpoints
    checkpoints = [
        ckpt for ckpt in checkpoints 
        if 'best_loss' not in ckpt.name and 'best_pck' not in ckpt.name
    ]
    
    # Delete all except last N
    for old_ckpt in checkpoints[:-keep_last]:
        old_ckpt.unlink()
        print(f"Deleted old checkpoint: {old_ckpt.name}")
```

**Usage**:
```python
# At end of each epoch, after saving checkpoint
if epoch > args.keep_last_n_checkpoints:
    cleanup_old_checkpoints(args.output_dir, keep_last=args.keep_last_n_checkpoints)
```

**Configuration**:
```bash
--keep_last_n_checkpoints 5  # Keep only last 5 regular checkpoints
```

**Benefits**:
- Reduces disk usage during long training runs
- Automatic (no manual cleanup needed)
- Still keeps all best-* checkpoints for analysis

**Tradeoffs**:
- Cannot resume from arbitrary old epoch (only last N)
- If cleanup runs before a crash, you lose older resume points

**Priority**: Medium (useful for long training, but manual cleanup works fine)

---

## 3. Disk Space Monitoring & Warnings

**Status**: ❌ Not Implemented

**Description**: Long training runs can fill disk unexpectedly. Adding disk space monitoring can prevent crashes from "disk full" errors.

**Proposed Implementation**:
```python
import shutil

def check_disk_space(output_dir, min_free_gb=10):
    """
    Check available disk space and warn/abort if too low.
    """
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    
    if free_gb < min_free_gb:
        print(f"⚠️  WARNING: Low disk space! Only {free_gb:.1f} GB remaining.")
        print(f"   Checkpoints require ~0.5 GB each.")
        
        if free_gb < 1.0:
            raise RuntimeError(f"ABORT: Disk space critically low ({free_gb:.2f} GB). Cannot save checkpoint safely.")
    
    return free_gb
```

**Usage**:
```python
# Before saving checkpoint
free_space = check_disk_space(args.output_dir, min_free_gb=5)
print(f"Disk space available: {free_space:.1f} GB")
```

**Benefits**:
- Prevents silent checkpoint save failures
- Gives user warning to free disk space
- Can abort gracefully before corrupting data

**Tradeoffs**:
- Adds overhead (disk stat check every epoch)
- Platform-dependent behavior

**Priority**: Low-Medium (useful, but most users monitor disk manually)

---

## 4. Robust Crash Handling & Recovery

**Status**: ❌ Not Implemented

**Description**: If training crashes unexpectedly (OOM, power loss, etc.), the current checkpoint might be incomplete. Adding a "last known good" checkpoint helps recovery.

**Proposed Implementation**:
```python
def save_checkpoint_with_backup(checkpoint_dict, checkpoint_path):
    """
    Save checkpoint with backup strategy:
    1. If checkpoint exists, rename to .backup
    2. Save new checkpoint
    3. If save succeeds, delete .backup
    4. If save fails, restore .backup
    """
    backup_path = checkpoint_path.with_suffix('.pth.backup')
    
    # Backup existing checkpoint
    if checkpoint_path.exists():
        shutil.copy(checkpoint_path, backup_path)
    
    try:
        # Save new checkpoint
        torch.save(checkpoint_dict, checkpoint_path)
        
        # Success - delete backup
        if backup_path.exists():
            backup_path.unlink()
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        
        # Restore backup if save failed
        if backup_path.exists():
            shutil.copy(backup_path, checkpoint_path)
            print(f"Restored backup checkpoint: {checkpoint_path}")
        
        raise
```

**Benefits**:
- Always have a valid checkpoint even if save fails
- Can recover from crashes during save
- More robust to filesystem issues

**Tradeoffs**:
- Requires 2× disk space (checkpoint + backup)
- More complex save logic
- Slower saves (extra copy operation)

**Priority**: Low (modern filesystems are robust, crashes during save are rare)

---

## 5. Checkpoint Integrity Validation

**Status**: ❌ Not Implemented

**Description**: Add checksums or validation to detect corrupted checkpoints before attempting to load them.

**Proposed Implementation**:
```python
import hashlib

def compute_checkpoint_hash(checkpoint_path):
    """Compute SHA256 hash of checkpoint file."""
    sha256 = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_checkpoint_with_hash(checkpoint_dict, checkpoint_path):
    """Save checkpoint and store its hash."""
    # Save checkpoint
    torch.save(checkpoint_dict, checkpoint_path)
    
    # Compute and save hash
    checkpoint_hash = compute_checkpoint_hash(checkpoint_path)
    hash_path = checkpoint_path.with_suffix('.pth.sha256')
    with open(hash_path, 'w') as f:
        f.write(checkpoint_hash)

def load_checkpoint_with_validation(checkpoint_path):
    """Load checkpoint with integrity check."""
    hash_path = checkpoint_path.with_suffix('.pth.sha256')
    
    if hash_path.exists():
        # Verify hash
        with open(hash_path, 'r') as f:
            expected_hash = f.read().strip()
        
        actual_hash = compute_checkpoint_hash(checkpoint_path)
        
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint corrupted! Hash mismatch:\n"
                             f"  Expected: {expected_hash}\n"
                             f"  Actual:   {actual_hash}")
    
    # Load checkpoint
    return torch.load(checkpoint_path)
```

**Benefits**:
- Detect corrupted checkpoints before loading
- Useful for distributed storage (network drives, cloud)
- Helps debug mysterious loading errors

**Tradeoffs**:
- Adds overhead (hash computation)
- Extra file per checkpoint
- Hash doesn't protect against in-memory corruption

**Priority**: Low (useful for production, overkill for research)

---

## 6. Save Full DataLoader Iterator State

**Status**: ❌ Not Implemented

**Description**: Currently, resuming training uses the same data ordering (due to RNG restoration) but doesn't save the exact position in the data loader. For true mid-epoch resume, we'd need to save which samples were already seen.

**Proposed Implementation**:
```python
# In training loop
for epoch in range(args.start_epoch, args.epochs):
    for batch_idx, batch in enumerate(data_loader):
        # Training step
        ...
        
        # Save checkpoint with batch position
        if batch_idx % 100 == 0:  # Every 100 batches
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,  # Position in epoch
                'dataloader_state': data_loader.state_dict(),  # If supported
                ...
            }
```

**Benefits**:
- Can resume mid-epoch (useful for very long epochs)
- No wasted computation on re-processing samples

**Tradeoffs**:
- DataLoader state saving not well-supported in PyTorch
- Requires custom iterator tracking
- Complex implementation
- Rarely needed (epoch checkpoints usually sufficient)

**Priority**: Very Low (rarely needed, complex to implement correctly)

---

## 7. PCK Metric Smoothing Before Best-Model Selection

**Status**: ❌ Not Implemented

**Description**: PCK can be noisy, especially early in training or with small validation sets. Using a smoothed or averaged PCK over the last N epochs might give more stable "best model" selection.

**Proposed Implementation**:
```python
from collections import deque

class SmoothedMetricTracker:
    def __init__(self, window=3):
        self.window = window
        self.history = deque(maxlen=window)
        self.best_smoothed = 0.0
    
    def update(self, value):
        self.history.append(value)
        smoothed = sum(self.history) / len(self.history)
        return smoothed
    
    def is_best(self, current_smoothed):
        if current_smoothed > self.best_smoothed:
            self.best_smoothed = current_smoothed
            return True
        return False

# In training loop
pck_tracker = SmoothedMetricTracker(window=3)

for epoch in range(args.epochs):
    val_stats = evaluate(...)
    val_pck = val_stats['pck']
    
    smoothed_pck = pck_tracker.update(val_pck)
    
    if pck_tracker.is_best(smoothed_pck):
        # Save best smoothed PCK model
        save_checkpoint(...)
```

**Benefits**:
- More stable best-model selection
- Less sensitive to noisy validation runs
- Better for small validation sets

**Tradeoffs**:
- Delays best-model detection by `window` epochs
- Adds complexity
- May miss true best if it's an outlier

**Priority**: Low (current raw PCK tracking works well, smoothing is optional)

---

## 8. Multi-Metric Best Model (Combined Loss + PCK)

**Status**: ❌ Not Implemented

**Description**: Instead of tracking best-loss and best-PCK separately, track a combined metric (e.g., weighted sum) for a single "best overall" model.

**Proposed Implementation**:
```python
def compute_combined_metric(val_loss, val_pck, loss_weight=0.5, pck_weight=0.5):
    """
    Compute combined metric (higher is better).
    val_loss: Lower is better → negate it
    val_pck: Higher is better → use as is
    """
    # Normalize to [0, 1] range (approximate)
    normalized_loss = max(0, 1 - val_loss)  # Assume loss in [0, 1]
    normalized_pck = val_pck  # Already in [0, 1]
    
    combined = loss_weight * normalized_loss + pck_weight * normalized_pck
    return combined

# In training loop
best_combined = 0.0

for epoch in range(args.epochs):
    val_stats = evaluate(...)
    val_loss = val_stats['loss']
    val_pck = val_stats['pck']
    
    combined = compute_combined_metric(val_loss, val_pck)
    
    if combined > best_combined:
        best_combined = combined
        save_checkpoint(..., name='checkpoint_best_combined.pth')
```

**Benefits**:
- Single "best" model instead of two
- Balances multiple objectives
- Simpler for users (one checkpoint to use)

**Tradeoffs**:
- Requires tuning weights (how much to weight loss vs PCK?)
- Less interpretable than separate best-loss and best-PCK
- May not be optimal for either metric

**Priority**: Low (having separate best-loss and best-PCK is more flexible)

---

## Summary of Optional Improvements

| Improvement | Priority | Complexity | Benefit | Recommended? |
|-------------|----------|------------|---------|--------------|
| Atomic checkpoint writing | Low | Low | Prevents corruption | Optional |
| Automatic checkpoint cleanup | Medium | Low | Saves disk space | Recommended for long runs |
| Disk space monitoring | Low-Medium | Low | Prevents crashes | Optional |
| Robust crash handling | Low | Medium | Better recovery | Not needed |
| Checkpoint integrity validation | Low | Medium | Detect corruption | Optional for production |
| DataLoader iterator state | Very Low | High | Mid-epoch resume | Not needed |
| PCK metric smoothing | Low | Low | Stable best-model | Optional |
| Multi-metric best model | Low | Low-Medium | Single best checkpoint | Not recommended |

**Recommendation**: The current checkpointing system is **production-ready** as-is. The only optional improvement worth considering for long training runs (300+ epochs) is **automatic checkpoint cleanup** to manage disk space.

All other improvements are "nice to have" but not necessary for safe, robust training.

