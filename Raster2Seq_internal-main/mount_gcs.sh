#!/bin/bash

# Script to mount GCS bucket to Raster2Seq_internal-main/data
# Make sure macfuse is installed first: brew install --cask macfuse

BUCKET_NAME="dl-category-agnostic-pose-mp100-data"
MOUNT_POINT="Raster2Seq_internal-main/data"
REMOTE_NAME="gcs_storage"
RCLONE_PATH="/Users/nikikaranikola/bin/rclone"

# Check if rclone exists
if [ ! -f "$RCLONE_PATH" ]; then
    echo "Error: rclone not found at $RCLONE_PATH"
    echo "Please install rclone or update RCLONE_PATH in this script"
    exit 1
fi

# Check if mount point exists
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point: $MOUNT_POINT"
    mkdir -p "$MOUNT_POINT"
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null || [ -n "$(mount | grep "$MOUNT_POINT")" ]; then
    echo "Already mounted at $MOUNT_POINT"
    exit 0
fi

echo "Mounting gs://$BUCKET_NAME to $MOUNT_POINT"
echo "This will run in the background. Check rclone_mount.log for status."

# Mount the bucket with macOS-friendly options
# Note: --allow-other removed as it requires special macOS configuration
"$RCLONE_PATH" mount "$REMOTE_NAME:$BUCKET_NAME" "$MOUNT_POINT" \
    --vfs-cache-mode writes \
    --vfs-cache-max-size 10G \
    --vfs-read-ahead 128M \
    --daemon \
    --log-file=rclone_mount.log \
    --log-level INFO \
    --umask 000

# Wait a moment for mount to initialize
sleep 2

# Check if mount was successful
if mountpoint -q "$MOUNT_POINT" 2>/dev/null || [ -n "$(mount | grep "$MOUNT_POINT")" ]; then
    echo "Successfully mounted! Check rclone_mount.log for details."
    echo "To unmount later, run: ./unmount_gcs.sh"
    echo "Or manually: fusermount -u $MOUNT_POINT"
    echo ""
    echo "Testing mount by listing contents..."
    ls -la "$MOUNT_POINT" | head -10
else
    echo "Mount may have failed. Check rclone_mount.log for details."
    echo "Common issues:"
    echo "  1. macfuse not properly installed - run: brew install --cask macfuse"
    echo "  2. macfuse kernel extension not enabled - check System Settings → Privacy & Security"
    echo "  3. Try running without --allow-other flag if you get permission errors"
    exit 1
fi

