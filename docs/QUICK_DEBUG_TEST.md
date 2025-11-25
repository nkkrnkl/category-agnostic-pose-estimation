# Quick Debug Test - Verify Your Setup Works

## 🎯 Goal

Run a **5-minute sanity check** to verify your CAPE setup is working correctly.

---

## ⚡ The Fastest Way

```bash
# 1. Make scripts executable (one-time setup)
chmod +x run_overfit_test.sh

# 2. Run overfit test
./run_overfit_test.sh 40

# That's it! Should complete in ~5 minutes.
```

---

## 📊 What to Expect

### ✅ Success (Model is Working)

```
Epoch: [0]  [  0/10]  loss: 45.234  loss_ce: 8.123  loss_coords: 37.111
Epoch: [0]  [ 10/10]  loss: 42.156  loss_ce: 7.890  loss_coords: 34.266
Epoch: [1]  [  0/10]  loss: 35.678  loss_ce: 6.234  loss_coords: 29.444
Epoch: [2]  [  0/10]  loss: 28.123  loss_ce: 4.567  loss_coords: 23.556
...
Epoch: [10] [  0/10]  loss: 8.456   loss_ce: 1.234  loss_coords: 7.222
Epoch: [20] [  0/10]  loss: 0.678   loss_ce: 0.089  loss_coords: 0.589  ← Near zero!
```

**Key indicators:**
- ✅ Loss decreases steadily
- ✅ By epoch 10: Loss < 10
- ✅ By epoch 20: Loss < 1
- ✅ By epoch 50: Loss < 0.1

**This proves:**
- Model architecture is correct
- Data pipeline is working
- Optimizer is updating weights
- Loss computation is correct

### ❌ Failure (Something is Wrong)

```
Epoch: [0]  [  0/10]  loss: 45.234  loss_ce: 8.123  loss_coords: 37.111
Epoch: [1]  [  0/10]  loss: 44.987  loss_ce: 8.098  loss_coords: 36.889
Epoch: [2]  [  0/10]  loss: 44.756  loss_ce: 8.067  loss_coords: 36.689
...
Epoch: [10] [  0/10]  loss: 43.123  loss_ce: 7.890  loss_coords: 35.233  ← Still high!
```

**Key indicators:**
- ❌ Loss barely decreases
- ❌ After 10 epochs: Loss > 30
- ❌ After 20 epochs: Loss > 20

**This indicates a bug in:**
- Model architecture (layers not connected)
- Data pipeline (wrong targets)
- Loss masking (too aggressive)
- Optimizer (learning rate too low)

**Debugging steps:** See `docs/DEBUG_OVERFIT_MODE.md` section "Troubleshooting"

---

## 🔍 Understanding the Test

### What `--debug_overfit_category` Does

**Normal Training:**
- Uses all 69 categories
- 1000 episodes per epoch
- Each episode: random category + random images
- Diverse but slower to verify

**Overfit Mode:**
- Uses **only 1 category** (e.g., category 40 = zebra)
- 10 episodes per epoch (small, repeatable)
- Same category every time
- **Should easily overfit** (memorize these 10 episodes)

### Why This is Useful

**Analogy:** Testing a calculator by computing 2+2 before trying complex equations.

If your model **can't** overfit on 10 repeated episodes from 1 category, something is fundamentally broken:
- Data not reaching model
- Gradients not flowing
- Loss not decreasing
- Optimizer not updating

If your model **can** overfit (loss → 0), then the core machinery works!

---

## 📝 Alternative: Manual Test

If you prefer to run commands manually:

```bash
# Activate environment
source activate_venv.sh

# Run overfit test with explicit parameters
python train_cape_episodic.py \
  --dataset_root . \
  --debug_overfit_category 40 \
  --debug_overfit_episodes 10 \
  --epochs 50 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --lr 5e-4 \
  --early_stopping_patience 0 \
  --output_dir outputs/debug_overfit_cat40 \
  --print_freq 5
```

---

## 🚀 After Successful Overfit Test

Once you see loss → 0:

✅ **Your setup is working!**

**Next steps:**

1. **Quick test on full data (5 epochs):**
   ```bash
   python train_cape_episodic.py --dataset_root . --epochs 5 --output_dir outputs/test_run
   ```

2. **Full 300-epoch training:**
   ```bash
   ./START_CAPE_TRAINING.sh
   ```

3. **Monitor training:**
   - Watch `outputs/cape_run/log.txt`
   - Check validation PCK
   - Wait for early stopping or 300 epochs

---

## ⏱️ Time Estimates

| Task | Time (CPU) | Time (GPU) |
|------|------------|------------|
| Overfit test (50 epochs, 1 cat) | ~5 min | ~2 min |
| Quick test (5 epochs, all cats) | ~30 min | ~10 min |
| Full training (300 epochs) | ~72 hours | ~12 hours |

---

## 💡 Pro Tips

1. **Always start with overfit test** on new setups
2. **Use different categories** to verify it's not category-specific:
   ```bash
   ./run_overfit_test.sh 1   # person
   ./run_overfit_test.sh 17  # cat
   ./run_overfit_test.sh 40  # zebra
   ```

3. **Compare logs** between successful and failed overfit tests to identify issues

4. **Enable debug mode** for detailed tensor shapes:
   ```bash
   export DEBUG_CAPE=1
   ./run_overfit_test.sh 40
   ```

---

**Ready?** Run `./run_overfit_test.sh 40` and watch the magic happen! ✨

If loss decreases → You're good to go! 🚀

If loss stays high → Check `docs/DEBUG_OVERFIT_MODE.md` for troubleshooting. 🔧

