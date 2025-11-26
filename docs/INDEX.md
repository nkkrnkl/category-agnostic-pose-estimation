# Documentation Index

This folder contains all documentation for the Category-Agnostic Pose Estimation (CAPE) project on MP-100.

## 📚 Quick Navigation

### 🚀 Getting Started
- **[START_HERE.md](START_HERE.md)** - Begin here if you're new to the project
- **[QUICKSTART_CAPE.md](QUICKSTART_CAPE.md)** - Quick start guide for CAPE
- **[QUICK_START.md](QUICK_START.md)** - Alternative quick start
- **[README_MP100_CAPE.md](README_MP100_CAPE.md)** - MP-100 dataset overview
- **[CAPE_IMPLEMENTATION_GUIDE.md](CAPE_IMPLEMENTATION_GUIDE.md)** - Comprehensive implementation guide

### 🔧 Setup & Installation
- **[VENV_SETUP.md](../VENV_SETUP.md)** - Virtual environment setup (kept in root)
- **[MPS_FIX.md](MPS_FIX.md)** - Apple Silicon (MPS) compatibility fix

### 📖 Core Concepts

#### ⭐ Training & Inference Pipeline (CRITICAL - READ FIRST!)
- **[TRAINING_INFERENCE_PIPELINE.md](TRAINING_INFERENCE_PIPELINE.md)** - 🔥 **MUST READ**: What inputs are used in training vs. inference
- **[TRAINING_INFERENCE_IO.md](TRAINING_INFERENCE_IO.md)** - Complete technical specification with code references
- **[AUDIT_COMPLETE_REPORT.md](AUDIT_COMPLETE_REPORT.md)** - Full audit report (20 files, 6000+ lines verified)
- **[AUDIT_SUMMARY_Nov25_2025.md](AUDIT_SUMMARY_Nov25_2025.md)** - Executive audit summary
- **[DEBUG_AND_TESTING_GUIDE.md](DEBUG_AND_TESTING_GUIDE.md)** - How to use debug mode and run tests
- **[DEBUG_OVERFIT_MODE.md](DEBUG_OVERFIT_MODE.md)** - 🎯 **NEW**: Single-category overfitting for debugging

#### Model Architecture
- **[CAPE_POSITIONAL_ENCODING_DESIGN.md](CAPE_POSITIONAL_Encoding_Design.md)** - ⭐ Dual positional encoding (spatial + sequence)
- **[REFACTORING_STAGE_1_2_3_COMPLETE.md](REFACTORING_STAGE_1_2_3_COMPLETE.md)** - Geometry-only encoder refactoring (Stages 1-3.5)
- **[CAPE_LOSSES_REFACTORING.md](CAPE_LOSSES_REFACTORING.md)** - How CAPE extends Raster2Seq loss functions
- **[PCK_EVALUATION_IMPLEMENTATION.md](PCK_EVALUATION_IMPLEMENTATION.md)** - PCK metric implementation details

#### Data Pipeline
- **[NORMALIZATION_PIPELINE_EXPLAINED.md](NORMALIZATION_PIPELINE_EXPLAINED.md)** - 3-step coordinate normalization
- **[KEYPOINT_NORMALIZATION_PIPELINE.md](KEYPOINT_NORMALIZATION_PIPELINE.md)** - Detailed keypoint normalization
- **[BBOX_CROPPING_IMPLEMENTATION.md](BBOX_CROPPING_IMPLEMENTATION.md)** - Bounding box cropping logic
- **[MP100_CATEGORY_ANALYSIS.md](MP100_CATEGORY_ANALYSIS.md)** - MP-100 dataset category analysis

#### Meta-Learning Setup
- **[CORRECT_META_LEARNING_SETUP.md](CORRECT_META_LEARNING_SETUP.md)** - Episodic meta-learning explained
- **[MULTI_INSTANCE_LIMITATION.md](MULTI_INSTANCE_LIMITATION.md)** - Handling multiple instances per image

### 🐛 Critical Fixes & Solutions

#### 🚨 Critical Bugs (MUST READ!)
- **[CRITICAL_SINGLE_KEYPOINT_BUG.md](CRITICAL_SINGLE_KEYPOINT_BUG.md)** - 🔥 **CRITICAL**: Single-keypoint output bug (Nov 25, 2025)
  - **Symptom:** Model predicted only 1 keypoint instead of full sequence during inference
  - **Root Cause:** `forward_inference` not accumulating outputs across autoregressive loop
  - **Impact:** Blocked all evaluation, caused `PCK: N/A (TypeError)`
  - **Status:** ✅ FIXED with regression tests and debug instrumentation

#### Variable-Length Keypoints (THE BIG ONE!)
- **[TRIMMING_LOGIC_EXPLAINED.md](TRIMMING_LOGIC_EXPLAINED.md)** - ⭐ **DEEP DIVE**: Why and how we trim predictions
- **[VARIABLE_KEYPOINTS_FIX.md](VARIABLE_KEYPOINTS_FIX.md)** - Overview of the variable keypoint fix
- **[FINAL_FIX_SUMMARY.md](FINAL_FIX_SUMMARY.md)** - User-friendly summary of the final fix

#### Visibility & Masking
- **[VISIBILITY_BUG_FIX.md](VISIBILITY_BUG_FIX.md)** - Visibility bug fixes
- **[PCK_VISIBILITY_FIX.md](PCK_VISIBILITY_FIX.md)** - PCK evaluation visibility fix
- **[LOSS_MASKING_VERIFICATION.md](LOSS_MASKING_VERIFICATION.md)** - How visibility masks are used in loss

#### Checkpointing & Training
- **[CHECKPOINT_FIXES_SUMMARY.md](CHECKPOINT_FIXES_SUMMARY.md)** - Checkpointing system fixes
- **[PCK_BASED_EARLY_STOPPING.md](PCK_BASED_EARLY_STOPPING.md)** - PCK-based early stopping
- **[CHECKPOINT_OPTIONAL_IMPROVEMENTS.md](CHECKPOINT_OPTIONAL_IMPROVEMENTS.md)** - Future checkpoint enhancements

#### Data Quality
- **[ANNOTATION_CLEANUP_SUMMARY.md](ANNOTATION_CLEANUP_SUMMARY.md)** - Annotation file cleanup
- **[IMAGE_VALIDATION_FIX.md](IMAGE_VALIDATION_FIX.md)** - Image validation improvements

#### UI/UX Improvements
- **[TQDM_PROGRESS_BARS.md](TQDM_PROGRESS_BARS.md)** - Training progress visualization
- **[TRAINING_METRICS.md](TRAINING_METRICS.md)** - Metrics and logging
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - ⭐ How to visualize predictions and analyze results

### 📝 Issue-Specific Fixes

These documents track fixes for specific issues identified during audits:

- **[ISSUES_8_AND_9_FIXES.md](ISSUES_8_AND_9_FIXES.md)** - PCK logging & skeleton edges
- **[ISSUES_10_AND_11_FIXES.md](ISSUES_10_AND_11_FIXES.md)** - Support mask & category ID
- **[ISSUE_14_FIX.md](ISSUE_14_FIX.md)** - Validation dataset fix
- **[ISSUES_15_AND_17_FIXES.md](ISSUES_15_AND_17_FIXES.md)** - Gradient accumulation & empty annotations
- **[ISSUES_18_19_22_23_FIXES.md](ISSUES_18_19_22_23_FIXES.md)** - Tokenizer, augmentation, checkpointing, early stopping
- **[QUERY_METADATA_FIX.md](QUERY_METADATA_FIX.md)** - Query metadata propagation

### 🎯 Summary Documents

- **[CRITICAL_FIXES_SUMMARY.md](CRITICAL_FIXES_SUMMARY.md)** - All critical fixes overview
- **[REQUIRED_MODIFICATIONS_COMPLETE.md](REQUIRED_MODIFICATIONS_COMPLETE.md)** - Completed modifications
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Project completion summary
- **[FINAL_CLEANUP_REPORT.md](FINAL_CLEANUP_REPORT.md)** - Final cleanup report

### 🔄 Development History

- **[RESTORED_CHANGES_SUMMARY.md](RESTORED_CHANGES_SUMMARY.md)** - Restored changes after accidental revert
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Code cleanup summary
- **[FILE_INVENTORY.md](FILE_INVENTORY.md)** - Project file inventory

### ✅ Quality Assurance

- **[VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)** - Pre-training verification checklist
- **[OPTIONAL_IMPROVEMENTS.md](OPTIONAL_IMPROVEMENTS.md)** - Future enhancement ideas

---

## 📂 Documentation Organization

### By Topic

#### **Data Processing**
1. NORMALIZATION_PIPELINE_EXPLAINED.md
2. KEYPOINT_NORMALIZATION_PIPELINE.md
3. BBOX_CROPPING_IMPLEMENTATION.md
4. ANNOTATION_CLEANUP_SUMMARY.md
5. IMAGE_VALIDATION_FIX.md

#### **Model Architecture**
1. CAPE_POSITIONAL_ENCODING_DESIGN.md ⭐ **NEW** - Dual positional encoding
2. REFACTORING_STAGE_1_2_3_COMPLETE.md - Geometry-only encoder
3. CAPE_IMPLEMENTATION_GUIDE.md
4. CAPE_LOSSES_REFACTORING.md
5. CORRECT_META_LEARNING_SETUP.md
6. MULTI_INSTANCE_LIMITATION.md

#### **Training Pipeline**
1. DEBUG_OVERFIT_MODE.md 🎯 **NEW** - Single-category debugging
2. CHECKPOINT_FIXES_SUMMARY.md
3. PCK_BASED_EARLY_STOPPING.md
4. TQDM_PROGRESS_BARS.md
5. TRAINING_METRICS.md

#### **Evaluation**
1. PCK_EVALUATION_IMPLEMENTATION.md
2. PCK_VISIBILITY_FIX.md
3. TRIMMING_LOGIC_EXPLAINED.md ⭐

#### **Troubleshooting**
1. VARIABLE_KEYPOINTS_FIX.md
2. FINAL_FIX_SUMMARY.md
3. VISIBILITY_BUG_FIX.md
4. MPS_FIX.md

### By Importance

#### **⭐ Must Read**
1. **TRAINING_INFERENCE_PIPELINE.md** - 🔥 **CRITICAL**: Training vs. inference inputs (VERIFIED CORRECT)
2. **START_HERE.md** - Project overview
3. **TRIMMING_LOGIC_EXPLAINED.md** - Critical concept for variable-length sequences
4. **FINAL_FIX_SUMMARY.md** - Latest bug fixes and current status
5. **CAPE_IMPLEMENTATION_GUIDE.md** - Complete implementation guide

#### **📖 Important**
1. NORMALIZATION_PIPELINE_EXPLAINED.md - Understand coordinate transformations
2. LOSS_MASKING_VERIFICATION.md - How visibility affects training
3. CORRECT_META_LEARNING_SETUP.md - Episodic training explained
4. CHECKPOINT_FIXES_SUMMARY.md - Safe long-training setup

#### **📋 Reference**
- All ISSUES_*_FIXES.md files - Specific bug fixes
- All *_IMPLEMENTATION.md files - Implementation details
- OPTIONAL_IMPROVEMENTS.md - Future work

---

## 🔍 Finding What You Need

### I want to...

**...understand training vs. inference inputs (CRITICAL!)**
→ Read TRAINING_INFERENCE_PIPELINE.md first! 🔥

**...verify my model can learn (debug mode)**
→ Use DEBUG_OVERFIT_MODE.md to run single-category test! 🎯

**...understand how the model works**
→ Start with CAPE_IMPLEMENTATION_GUIDE.md

**...fix a training error**
→ Check FINAL_FIX_SUMMARY.md and TROUBLESHOOTING section

**...understand the single-keypoint bug fix**
→ Read CRITICAL_SINGLE_KEYPOINT_BUG.md 🚨

**...understand coordinate transformations**
→ Read NORMALIZATION_PIPELINE_EXPLAINED.md

**...set up the environment**
→ Follow QUICKSTART_CAPE.md

**...understand variable-length keypoints**
→ Read TRIMMING_LOGIC_EXPLAINED.md (detailed) or VARIABLE_KEYPOINTS_FIX.md (overview)

**...understand checkpointing**
→ Check CHECKPOINT_FIXES_SUMMARY.md

**...understand evaluation metrics**
→ Read PCK_EVALUATION_IMPLEMENTATION.md

**...know what was recently fixed**
→ Check FINAL_FIX_SUMMARY.md

**...debug why my model won't train**
→ Use DEBUG_OVERFIT_MODE.md for quick diagnostics! 🎯

---

## 📅 Chronological Order (Latest First)

1. **CAPE_POSITIONAL_ENCODING_DESIGN.md** - Sequence positional encoding integration (Nov 26, 2025) ⭐
2. **REFACTORING_STAGE_1_2_3_COMPLETE.md** - Stage 3.5 completion (Nov 26, 2025)
3. **CRITICAL_SINGLE_KEYPOINT_BUG.md** - Critical autoregressive bug fix (Nov 25, 2025) 🚨
4. **DEBUG_OVERFIT_MODE.md** - Single-category debugging mode (Nov 25, 2025) 🎯
5. **TRAINING_INFERENCE_PIPELINE.md** - Training/inference audit (Nov 25, 2025) 🔥
4. **AUDIT_COMPLETE_REPORT.md** - Full audit verification report
3. **TRAINING_INFERENCE_IO.md** - Complete I/O specification
4. **DEBUG_AND_TESTING_GUIDE.md** - Debug mode and test suite
5. **TRIMMING_LOGIC_EXPLAINED.md** - Deep dive into trimming logic
6. **FINAL_FIX_SUMMARY.md** - Variable keypoint fix summary
7. **VARIABLE_KEYPOINTS_FIX.md** - Variable keypoint issue resolution
4. **TQDM_PROGRESS_BARS.md** - Progress bar integration
5. **RESTORED_CHANGES_SUMMARY.md** - Restored appearance augmentation
6. **MPS_FIX.md** - Apple Silicon compatibility
7. **ANNOTATION_CLEANUP_SUMMARY.md** - Annotation cleanup
8. **MP100_CATEGORY_ANALYSIS.md** - Category split derivation
9. **CORRECT_META_LEARNING_SETUP.md** - Meta-learning validation
10. **PCK_BASED_EARLY_STOPPING.md** - Early stopping update
11. **CHECKPOINT_FIXES_SUMMARY.md** - Checkpointing overhaul
... (earlier documents)

---

## 🏷️ Tags

Documents are tagged by category:

- `#setup` - Environment and installation
- `#architecture` - Model architecture
- `#data` - Data processing and loading
- `#training` - Training pipeline
- `#evaluation` - Metrics and evaluation
- `#bugfix` - Bug fixes
- `#optimization` - Performance improvements
- `#documentation` - Documentation updates

---

## 🤝 Contributing

When adding new documentation:
1. Use descriptive, all-caps filenames (e.g., `NEW_FEATURE_EXPLAINED.md`)
2. Add an entry to this INDEX.md under the appropriate section
3. Include tags at the top of the document
4. Cross-reference related documents

---

## 📧 Questions?

If you can't find what you're looking for:
1. Check **START_HERE.md** for an overview
2. Use your editor's search to grep across all .md files
3. Check **FINAL_FIX_SUMMARY.md** for recent changes

Last updated: November 26, 2025 (Sequence Positional Encoding Integrated ⭐)

