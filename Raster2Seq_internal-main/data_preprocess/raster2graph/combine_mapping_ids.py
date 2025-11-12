import json
import os

def generate_combined_mapping(file_mapping_path, image_id_mapping_path, output_path):
    """
    Generates a combined mapping file from an original filename mapping
    and an image ID mapping.

    Args:
        file_mapping_path (str): Path to the text file mapping original filenames
                                 to intermediate 6-digit IDs.
        image_id_mapping_path (str): Path to the JSON file mapping intermediate
                                     IDs to destination IDs.
        output_path (str): Path where the new combined mapping file will be saved.
    """
    # 1. Read test_file_mapping.txt
    org_fn_to_intermediate_id = {}
    try:
        with open(file_mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    org_fn = parts[0]
                    # Convert the 6-digit string ID to an integer for lookup
                    intermediate_id_str = parts[1]
                    # Remove leading zeros and convert to int
                    intermediate_id = int(intermediate_id_str)
                    org_fn_to_intermediate_id[org_fn] = intermediate_id
    except FileNotFoundError:
        print(f"Error: The file '{file_mapping_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading '{file_mapping_path}': {e}")
        return

    # 2. Read test_image_id_mapping.json
    intermediate_id_to_dst_fn = {}
    try:
        with open(image_id_mapping_path, 'r') as f:
            image_id_data = json.load(f)
            for key, value in image_id_data.items():
                # Keys in JSON are strings, convert to int for consistency
                intermediate_id_to_dst_fn[int(key)] = value
    except FileNotFoundError:
        print(f"Error: The file '{image_id_mapping_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{image_id_mapping_path}'. Please ensure it's valid JSON.")
        return
    except Exception as e:
        print(f"Error reading '{image_id_mapping_path}': {e}")
        return

    # 3. Create the combined mapping and write to output file
    combined_mappings = []
    found_mappings_count = 0
    for org_fn, intermediate_id in org_fn_to_intermediate_id.items():
        if intermediate_id in intermediate_id_to_dst_fn:
            dst_fn = intermediate_id_to_dst_fn[intermediate_id]
            combined_mappings.append(f"{org_fn} {dst_fn}")
            found_mappings_count += 1
        else:
            # Optionally, you can print a warning for IDs not found
            print(f"Warning: Intermediate ID '{intermediate_id}' for '{org_fn}' not found in image ID mapping.")

    try:
        with open(output_path, 'w') as f:
            for mapping_line in combined_mappings:
                f.write(mapping_line + '\n')
        print(f"\nSuccessfully generated combined mapping to '{output_path}'.")
        print(f"Total original filenames processed: {len(org_fn_to_intermediate_id)}")
        print(f"Total combined mappings written: {found_mappings_count}")
    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")


# Define file paths
file_mapping_path = 'data/R2G_hr_dataset_processed/test_file_mapping.txt'
image_id_mapping_path = 'data/R2G_hr_dataset_processed_v1/annotations/test_image_id_mapping.json'
output_mapping_path = 'data/R2G_hr_dataset_processed_v1/annotations/test_combined_mapping.txt'

# Run the mapping function
generate_combined_mapping(file_mapping_path, image_id_mapping_path, output_mapping_path)

# You can optionally print the content of the generated file to verify
print("\n--- Content of combined_mapping.txt ---")
try:
    with open(output_mapping_path, 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("Output file was not created.")

# Clean up dummy files (optional)
# os.remove(file_mapping_path)
# os.remove(image_id_mapping_path)
