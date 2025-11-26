# Setting Up Google Cloud VM with CUDA GPUs for CAPE Training

This guide walks you through creating a GPU-enabled VM on Google Cloud Platform and setting it up to run CAPE training.

## 🚀 Quick Start

### Option 1: Using gcloud CLI (Recommended)

```bash
# Set your project
export PROJECT_ID="dl-category-agnostic-pose-est"
gcloud config set project $PROJECT_ID

# Create VM with GPU (using us-central1-a to match GCS bucket location)
gcloud compute instances create cape-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True"
```

### Option 2: Using Google Cloud Console

1. Go to [Compute Engine > VM instances](https://console.cloud.google.com/compute/instances)
2. Click "Create Instance"
3. Configure:
   - **Name**: `cape-training-vm`
   - **Region/Zone**: `us-central1-a` (or your preferred zone with GPU availability)
   - **Machine type**: `n1-standard-8` (8 vCPUs, 30GB RAM)
   - **GPU**: Add `NVIDIA T4` (1 GPU)
   - **Boot disk**: Ubuntu 22.04 LTS, 200GB SSD
   - **Firewall**: Allow HTTP/HTTPS traffic (optional)
4. Click "Create"

## 📋 Prerequisites

1. **Google Cloud Project** with billing enabled
2. **GPU Quota**: Request GPU quota if needed:
   ```bash
   gcloud compute project-info describe --project=$PROJECT_ID
   # If quota is 0, request increase at:
   # https://console.cloud.google.com/iam-admin/quotas
   ```

3. **gcloud CLI** installed (optional, for CLI method):
   ```bash
   # Install: https://cloud.google.com/sdk/docs/install
   gcloud auth login
   gcloud config set project $PROJECT_ID
   ```

## 🔧 Step-by-Step Setup

### 1. Create the VM

```bash
# Set variables
PROJECT_ID="dl-category-agnostic-pose-est"
ZONE="us-central1-a"  # Matches GCS bucket location (US-CENTRAL1) - reduces data transfer costs
VM_NAME="cape-training-vm"

# Create VM with GPU
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Note**: GPU availability varies by zone. Check availability:
```bash
gcloud compute accelerator-types list --filter="zone:us-central1-a"
```

### 2. SSH into the VM

```bash
# SSH using gcloud
gcloud compute ssh $VM_NAME --zone=$ZONE

# Or use the web console:
# https://console.cloud.google.com/compute/instances
# Click "SSH" button next to your VM
```

### 3. Install CUDA Drivers and Toolkit

Once SSH'd into the VM:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install CUDA drivers (automatically installed if metadata was set)
# Verify installation:
nvidia-smi

# If nvidia-smi doesn't work, install drivers manually:
sudo apt-get install -y nvidia-driver-535
sudo reboot
# After reboot, verify:
nvidia-smi
```

### 4. Install Python and PyTorch

```bash
# Install Python 3.10+ and pip
sudo apt-get install -y python3.10 python3-pip python3-venv git

# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install PyTorch with CUDA support
# Check CUDA version: nvidia-smi shows driver version
# For CUDA 11.8 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 5. Clone Repository

```bash
# Clone your repository
cd ~
git clone https://github.com/nkkrnkl/category-agnostic-pose-estimation.git
cd category-agnostic-pose-estimation

# Switch to your branch
git checkout pavlos-topic-copy
```

### 6. Install Dependencies

```bash
# Install requirements
pip install -r requirements_cape.txt

# Install detectron2 (for CUDA 11.8)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install additional dependencies
pip install descartes shapely>=1.8.0
```

### 7. Set Up GCS Bucket Access

```bash
# Authenticate with GCP (if not already done)
gcloud auth application-default login

# Mount GCS bucket using gcsfuse
sudo apt-get install -y gcsfuse

# Create mount point
mkdir -p ~/data
gcsfuse --implicit-dirs dl-category-agnostic-pose-mp100-data ~/data

# Create symlink (if needed)
cd ~/category-agnostic-pose-estimation
ln -s ~/data data
```

**Alternative**: Use `gsutil` to copy data (one-time):
```bash
# Copy data from bucket to VM (one-time, takes time)
gsutil -m cp -r gs://dl-category-agnostic-pose-mp100-data/* ~/data/
```

### 8. Run Training

```bash
# Activate virtual environment
source ~/venv/bin/activate

# Navigate to project
cd ~/category-agnostic-pose-estimation

# Run single image training
python -m models.train_cape_episodic \
    --debug_single_image_path "bison_body/000000001120.jpg" \
    --dataset_root . \
    --category_split_file category_splits.json \
    --output_dir output/single_image_gcp \
    --device cuda:0 \
    --epochs 20 \
    --batch_size 1 \
    --num_queries_per_episode 1 \
    --episodes_per_epoch 20 \
    --use_amp \
    --cudnn_benchmark \
    --num_workers 4
```

## 💰 Cost Optimization

### Stop VM When Not in Use

```bash
# Stop VM (saves compute costs, keeps disk)
gcloud compute instances stop $VM_NAME --zone=$ZONE

# Start VM when needed
gcloud compute instances start $VM_NAME --zone=$ZONE
```

### Use Preemptible Instances (Cheaper, but can be terminated)

```bash
# Add --preemptible flag when creating VM
gcloud compute instances create $VM_NAME \
    ... \
    --preemptible
```

**Cost**: Preemptible VMs are ~80% cheaper but can be terminated with 30s notice.

### Use Spot VMs (Even Cheaper)

```bash
# Use --provisioning-model=SPOT
gcloud compute instances create $VM_NAME \
    ... \
    --provisioning-model=SPOT
```

## 📊 GPU Options and Pricing

| GPU Type | vRAM | Approx. Cost/Hour | Best For |
|----------|------|------------------|----------|
| **T4** | 16GB | ~$0.35 | Good balance, recommended |
| **V100** | 16GB | ~$2.48 | Faster training |
| **A100** | 40GB | ~$3.67 | Large models, fastest |
| **L4** | 24GB | ~$0.50 | Newer, efficient |

**Recommendation**: Start with **T4** for cost-effectiveness.

## 🔐 Security Best Practices

### 1. Use Service Account (Recommended)

```bash
# Create service account
gcloud iam service-accounts create cape-training-sa \
    --display-name="CAPE Training Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cape-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"  # For GCS bucket access

# Create VM with service account
gcloud compute instances create $VM_NAME \
    ... \
    --service-account=cape-training-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

### 2. Use SSH Keys Instead of Browser SSH

```bash
# Generate SSH key pair
ssh-keygen -t rsa -f ~/.ssh/gcp_cape -C "cape-training"

# Add to GCP
gcloud compute project-info add-metadata \
    --metadata-from-file ssh-keys=~/.ssh/gcp_cape.pub
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check if GPU is visible
lspci | grep -i nvidia

# Check driver installation
nvidia-smi

# Reinstall drivers if needed
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Out of Memory

```bash
# Reduce batch size in training command
--batch_size 1
--num_workers 2

# Or use smaller GPU (T4 instead of V100)
```

### GCS Bucket Access Denied

```bash
# Verify service account has permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:cape-training-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Grant storage access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cape-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

### Training Script Not Found

```bash
# Verify you're in the right directory
cd ~/category-agnostic-pose-estimation

# Check Python path
which python3
python3 -m models.train_cape_episodic --help
```

## 📝 Complete Setup Script

Save this as `setup_gcp_vm.sh` and run on the VM:

```bash
#!/bin/bash
set -e

echo "Setting up CAPE training environment on GCP VM..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3.10 python3-pip python3-venv git build-essential

# Install CUDA drivers (if not already installed)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    echo "⚠️  Please reboot and run this script again"
    exit 0
fi

# Verify GPU
echo "Verifying GPU..."
nvidia-smi

# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')"

# Clone repository
cd ~
if [ ! -d "category-agnostic-pose-estimation" ]; then
    git clone https://github.com/nkkrnkl/category-agnostic-pose-estimation.git
fi
cd category-agnostic-pose-estimation
git checkout pavlos-topic-copy

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements_cape.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install descartes shapely>=1.8.0

# Set up GCS access
echo "Setting up GCS access..."
sudo apt-get install -y gcsfuse
mkdir -p ~/data
gcsfuse --implicit-dirs dl-category-agnostic-pose-mp100-data ~/data
cd ~/category-agnostic-pose-estimation
ln -sf ~/data data

echo "✅ Setup complete!"
echo ""
echo "To activate environment:"
echo "  source ~/venv/bin/activate"
echo "  cd ~/category-agnostic-pose-estimation"
echo ""
echo "To run training:"
echo "  python -m models.train_cape_episodic --debug_single_image_path 'bison_body/000000001120.jpg' ..."
```

## 🚀 Quick Commands Reference

```bash
# Start VM
gcloud compute instances start cape-training-vm --zone=us-central1-a

# Stop VM
gcloud compute instances stop cape-training-vm --zone=us-central1-a

# SSH into VM
gcloud compute ssh cape-training-vm --zone=us-central1-a

# Check VM status
gcloud compute instances describe cape-training-vm --zone=us-central1-a

# Delete VM (when done)
gcloud compute instances delete cape-training-vm --zone=us-central1-a
```

## 💡 Tips

1. **Use tmux/screen** for long-running training:
   ```bash
   sudo apt-get install -y tmux
   tmux new -s training
   # Run training command
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Transfer files** to/from VM:
   ```bash
   # From local to VM
   gcloud compute scp local_file.txt cape-training-vm:~/ --zone=us-central1-a
   
   # From VM to local
   gcloud compute scp cape-training-vm:~/output/checkpoint.pth . --zone=us-central1-a
   ```

4. **Check costs**:
   ```bash
   # View current month costs
   gcloud billing accounts list
   ```

## 📚 Additional Resources

- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [GCS FUSE Documentation](https://cloud.google.com/storage/docs/gcs-fuse)

---

**Last Updated**: 2025-01-XX

