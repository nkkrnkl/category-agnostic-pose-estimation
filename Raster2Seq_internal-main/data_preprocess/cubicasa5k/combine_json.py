import json
import os
import glob
import shutil
from pathlib import Path
from copy import copy

def combine_json_files(input_pattern, data_path, split_type, output_file, start_image_id=0):
    """
    Combines multiple COCO-style JSON annotation files into a single file.
    
    Args:
        input_pattern: Glob pattern to match the input JSON files (e.g., "annotations/*.json")
        output_file: Path to the output combined JSON file
    """
    # Initialize combined data structure
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track image and annotation IDs to avoid duplicates
    image_ids_seen = set()
    annotation_ids_seen = set()

    next_image_id = start_image_id
    next_annotation_id = 0
    skip_file_list = []
    image_id_mapping = {}
    
    # Find all matching JSON files
    json_files = sorted(glob.glob(input_pattern))
    print(f"Found {len(json_files)} JSON files to combine")

    
    # Process each file
    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Store categories from the first file
        if i == 0 and data.get("categories"):
            # save_list = []
            # for key, value in CC5K_CLASS_MAPPING.items():
            #     type_dict = {"supercategory": "room", "id": value, "name": key}
            #     save_list.append(type_dict)
            # combined_data["categories"] = save_list
            combined_data["categories"] = data["categories"]
        
        # empty annos
        if len(data['annotations']) == 0:
            skip_file_list.append(data['images'][0]['id'])
            continue
        
        # Process images
        for image in data.get("images", []):
            if image['id'] not in image_id_mapping:
                image_id_mapping[image['id']] = next_image_id
            else:
                skip_file_list.append(image['id'])
                continue
            image['id'] = next_image_id
            next_image_id += 1
            # org_file_name = copy(image['file_name'])
            image['file_name'] = str(image['id']).zfill(5) + ".png"
            org_file_name = os.path.basename(json_file).replace(".json", ".png")
            if image['file_name'] != org_file_name and os.path.exists(f"{data_path}/{split_type}/{org_file_name}"):
                # shutil.copy(f"{data_path}/{split_type}/{org_file_name}", f"{data_path}/{split_type}_aux/{org_file_name}")
                shutil.move(f"{data_path}/{split_type}/{org_file_name}", f"{data_path}/{split_type}/{image['file_name']}")
            combined_data["images"].append(image)
        
        # Process annotations
        for annotation in data.get("annotations", []):
            annotation["id"] = next_annotation_id
            next_annotation_id += 1
            annotation["image_id"] = image_id_mapping[annotation["image_id"]]

            annotation_ids_seen.add(annotation["id"])
            combined_data["annotations"].append(annotation)
    
    # Write combined data to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    with open(output_path.parent / f"{output_path.name.split('.')[0]}_image_id_mapping.json", 'w') as f:
        json.dump(image_id_mapping, f, indent=2)

    if len(skip_file_list):
        with open(output_path.parent / f"{output_path.name.split('.')[0]}_skipped.txt", 'w') as f:
            f.write("\n".join([str(x) for x in skip_file_list]))
    
    print(f"Combined data written to {output_file}")
    print(f"Total images: {len(combined_data['images'])}")
    print(f"Total annotations: {len(combined_data['annotations'])}")
    print(f"Total categories: {len(combined_data['categories'])}")
    print(f"Skipped images: {len(skip_file_list)}")
    
    return combined_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple COCO-style JSON annotation files")
    parser.add_argument("--input", required=True, help="Glob pattern for input JSON files, e.g., 'annotations/*.json'")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for i, split in enumerate(splits):
        if split == 'train':
            start_image_id = 0
        else:
            start_image_id += len(list(Path(f"{args.input}/{splits[i-1]}").glob("*.png")))
            
        combine_json_files(f"{args.input}/annotations_json/{split}/*.json", 
                           args.input,
                           split, 
                           f"{args.output}/{split}.json",
                           start_image_id=start_image_id)