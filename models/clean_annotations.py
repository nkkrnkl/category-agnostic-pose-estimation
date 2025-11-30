#!/usr/bin/env python3
"""
Script to clean annotation files by removing entries for non-existent images.
For each JSON file in the `annotations` folder:
  - Remove `images` whose files don't exist under the data folder.
  - Remove `annotations` whose `image_id` no longer exists.
  - Keep all `categories`, but report which ones end up with 0 images
    (i.e., we did not find a single existing image for that category).
It also creates:
  - A backup of every original JSON next to it as `*.json.backup`.
  - A text report `annotation_cleanup_report.txt` in PROJECT_ROOT summarizing:
        * images/annotations before vs after
        * categories that have zero images after cleanup
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = Path(
    "/Users/pavlosrousoglou/Desktop/Cornell/Deep Learning/category-agnostic-pose-estimation"
)
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
IMAGES_DIR = PROJECT_ROOT / "data"
DATA_PREFIX = "data/"
def get_existing_images():
    """Collect all existing image paths under IMAGES_DIR."""
    existing_images = set()
    if IMAGES_DIR.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_file in IMAGES_DIR.rglob(pattern):
                rel_path = img_file.relative_to(IMAGES_DIR)
                existing_images.add(str(rel_path))
    print(f"Found {len(existing_images)} existing image files")
    return existing_images
def clean_annotation_file(json_path, existing_images):
    """Clean a single annotation JSON file and return stats."""
    print(f"\nProcessing: {json_path.name}")
    with open(json_path, "r") as f:
        original_text = f.read()
    data = json.loads(original_text)
    backup_path = json_path.with_suffix(json_path.suffix + ".backup")
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)
        print(f"  Backup created: {backup_path.name}")
    else:
        print(f"  Backup already exists, not overwriting: {backup_path.name}")
    original_images = len(data.get("images", []))
    original_annotations = len(data.get("annotations", []))
    original_categories = len(data.get("categories", []))
    kept_images = []
    kept_image_ids = set()
    removed_images = []
    for img in data.get("images", []):
        img_filename = img.get("file_name", "")
        clean_path = img_filename
        if clean_path.startswith(DATA_PREFIX):
            clean_path = clean_path[len(DATA_PREFIX) :]
        file_exists = (
            clean_path in existing_images or (IMAGES_DIR / clean_path).exists()
        )
        if file_exists:
            kept_images.append(img)
            kept_image_ids.add(img["id"])
        else:
            removed_images.append(img_filename)
    kept_annotations = [
        ann
        for ann in data.get("annotations", [])
        if ann.get("image_id") in kept_image_ids
    ]
    used_category_ids = {ann.get("category_id") for ann in kept_annotations}
    kept_categories = data.get("categories", [])
    active_categories = [
        cat for cat in kept_categories if cat["id"] in used_category_ids
    ]
    data["images"] = kept_images
    data["annotations"] = kept_annotations
    data["categories"] = kept_categories
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"  Original: {original_images} images, {original_annotations} annotations, "
        f"{original_categories} categories"
    )
    print(
        f"  Cleaned:  {len(kept_images)} images, {len(kept_annotations)} annotations, "
        f"{len(kept_categories)} categories"
    )
    print(
        f"  Removed:  {original_images - len(kept_images)} images, "
        f"{original_annotations - len(kept_annotations)} annotations"
    )
    print(f"  Active categories: {len(active_categories)}/{len(kept_categories)}")
    if 0 < len(removed_images) <= 10:
        print(f"  Removed images (file_name): {removed_images}")
    elif len(removed_images) > 10:
        print(
            f"  Sample removed images: {removed_images[:5]} "
            f"... (and {len(removed_images) - 5} more)"
        )
    return {
        "file": json_path.name,
        "original_images": original_images,
        "kept_images": len(kept_images),
        "removed_images": original_images - len(kept_images),
        "original_annotations": original_annotations,
        "kept_annotations": len(kept_annotations),
        "removed_annotations": original_annotations - len(kept_annotations),
        "total_categories": len(kept_categories),
        "active_categories": len(active_categories),
    }
def analyze_missing_categories(annotation_files):
    """
    After cleaning, analyze categories across all JSONs and count
    how many images each category is associated with.
    Category IDs come from the 'categories' list in the JSON.
    We count images per category using the surviving annotations.
    """
    category_image_count = defaultdict(lambda: {"total": 0, "splits": defaultdict(int)})
    all_categories = {}
    for json_path in annotation_files:
        split_name = json_path.stem
        with open(json_path, "r") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            if cat["id"] not in all_categories:
                all_categories[cat["id"]] = cat.get("name", f"id_{cat['id']}")
        image_categories = defaultdict(set)
        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            img_id = ann.get("image_id")
            if cat_id is not None and img_id is not None:
                image_categories[cat_id].add(img_id)
        for cat_id, img_ids in image_categories.items():
            count = len(img_ids)
            category_image_count[cat_id]["total"] += count
            category_image_count[cat_id]["splits"][split_name] = count
    return all_categories, category_image_count
def write_cleanup_report(results, all_categories, category_image_count, output_path):
    """Write final text report with annotation stats and missing categories."""
    total_original_images = sum(r["original_images"] for r in results)
    total_kept_images = sum(r["kept_images"] for r in results)
    total_original_annotations = sum(r["original_annotations"] for r in results)
    total_kept_annotations = sum(r["kept_annotations"] for r in results)
    missing_categories = []
    for cat_id, cat_name in all_categories.items():
        total = category_image_count[cat_id]["total"]
        if total == 0:
            missing_categories.append((cat_id, cat_name))
    missing_categories.sort(key=lambda x: x[0])
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("ANNOTATION CLEANUP REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write("Per-file summary (images & annotations before/after):\n")
        f.write("-" * 100 + "\n")
        header = (
            f"{'File':40s} "
            f"{'Img(before)':>12s} {'Img(after)':>11s} "
            f"{'Ann(before)':>12s} {'Ann(after)':>11s}\n"
        )
        f.write(header)
        f.write("-" * 100 + "\n")
        for r in results:
            line = (
                f"{r['file'][:40]:40s} "
                f"{r['original_images']:12d} {r['kept_images']:11d} "
                f"{r['original_annotations']:12d} {r['kept_annotations']:11d}\n"
            )
            f.write(line)
        f.write("\nTotals across all files:\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"  Images:      {total_kept_images} kept / {total_original_images} before "
            f"({total_original_images - total_kept_images} removed)\n"
        )
        f.write(
            f"  Annotations: {total_kept_annotations} kept / {total_original_annotations} before "
            f"({total_original_annotations - total_kept_annotations} removed)\n"
        )
        f.write("\n\nCategories with NO images after cleanup:\n")
        f.write("-" * 100 + "\n")
        if missing_categories:
            for cat_id, cat_name in missing_categories:
                f.write(f"  ID {str(cat_id):>3}: {cat_name}\n")
        else:
            f.write("  None – every category has at least one image.\n")
    print(f"\nDetailed cleanup report written to: {output_path}")
def main():
    print("=" * 80)
    print("Cleaning annotation files")
    print("=" * 80)
    existing_images = get_existing_images()
    if not existing_images:
        print("\nWARNING: No images found! Please check IMAGES_DIR.")
        return
    annotation_files = sorted([
        f for f in ANNOTATIONS_DIR.glob("*.json") 
        if not f.name.startswith("._")
    ])
    if not annotation_files:
        print(f"\nNo annotation files found in {ANNOTATIONS_DIR}")
        return
    print(f"\nFound {len(annotation_files)} annotation file(s) to process")
    results = []
    for json_path in annotation_files:
        result = clean_annotation_file(json_path, existing_images)
        results.append(result)
    all_categories, category_image_count = analyze_missing_categories(annotation_files)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_removed_images = sum(r["removed_images"] for r in results)
    total_removed_annotations = sum(r["removed_annotations"] for r in results)
    total_kept_images = sum(r["kept_images"] for r in results)
    total_kept_annotations = sum(r["kept_annotations"] for r in results)
    print(f"\nTotal across all files:")
    print(f"  Images:      {total_kept_images} kept, {total_removed_images} removed")
    print(
        f"  Annotations: {total_kept_annotations} kept, "
        f"{total_removed_annotations} removed"
    )
    missing_categories = []
    for cat_id, cat_name in all_categories.items():
        total = category_image_count[cat_id]["total"]
        if total == 0:
            missing_categories.append((cat_id, cat_name))
    missing_categories.sort(key=lambda x: x[0])
    if missing_categories:
        print("\nCategories with NO images after cleanup:")
        for cat_id, cat_name in missing_categories[:50]:
            print(f"  ID {str(cat_id):>3}: {cat_name}")
        if len(missing_categories) > 50:
            print(f"  ... and {len(missing_categories) - 50} more")
    else:
        print("\nEvery category has at least one image after cleanup.")
    report_path = PROJECT_ROOT / "annotation_cleanup_report.txt"
    write_cleanup_report(results, all_categories, category_image_count, report_path)
    print(
        "\nBackup files were created next to each JSON with the '.json.backup' suffix."
    )
    print("Done!")
if __name__ == "__main__":
    main()