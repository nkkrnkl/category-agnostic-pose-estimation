# Mounting GCS Bucket Instructions

This guide explains how to mount your GCS bucket `dl-category-agnostic-pose-mp100-data` to `Raster2Seq_internal-main/data`.

## Prerequisites

1. **Install macfuse** (required for mounting on macOS):
   ```bash
   brew install --cask macfuse
   ```
   Note: This will require your password and may need you to enable the kernel extension in System Settings → Privacy & Security.

2. **rclone is already installed and configured** ✅
   - Remote name: `gcs_storage`
   - Project: `dl-category-agnostic-pose-est`
   - Bucket: `dl-category-agnostic-pose-mp100-data`

## Mounting the Bucket

Run the mount script:
```bash
cd Raster2Seq_internal-main
./mount_gcs.sh
```

This will mount the bucket to `Raster2Seq_internal-main/data`. The mount runs in the background (daemon mode).

## Unmounting

When you're done, unmount with:
```bash
cd Raster2Seq_internal-main
./unmount_gcs.sh
```

Or manually:
```bash
fusermount -u Raster2Seq_internal-main/data
# or
umount Raster2Seq_internal-main/data
```

## Verification

After mounting, you should be able to see your bucket contents:
```bash
ls Raster2Seq_internal-main/data
```

You should see folders like: `alpaca_face`, `annotations`, `bed`, `car`, etc.

## Troubleshooting

- **"macfuse not found"**: Install macfuse first: `brew install --cask macfuse`
- **Permission errors**: Make sure you've authorized rclone (already done ✅)
- **Mount fails**: Check `rclone_mount.log` for details
- **Can't unmount**: Try `sudo umount Raster2Seq_internal-main/data`

## Using the Mounted Data

Once mounted, you can use the data in `Raster2Seq_internal-main/data` just like a local directory. All preprocessing scripts can point to this path.

Example:
```bash
# In your preprocessing scripts, use:
--data_root=Raster2Seq_internal-main/data/cubicasa5k/
# or whatever subdirectory structure exists in your bucket
```

