#!/bin/bash

# Script to unmount GCS bucket from Raster2Seq_internal-main/data

MOUNT_POINT="Raster2Seq_internal-main/data"

if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "Unmounting $MOUNT_POINT..."
    fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT"
    if [ $? -eq 0 ]; then
        echo "Successfully unmounted!"
    else
        echo "Unmount failed. You may need to use: sudo umount $MOUNT_POINT"
    fi
else
    echo "Not mounted at $MOUNT_POINT"
fi

