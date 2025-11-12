# preprocess raw Raster2Graph high-resolution dataset
python dataset_preprocess/image_process.py --data_root=data/R2G_hr_dataset/

# convert to COCO-style dataset
python dataset_prep/convert_to_coco.py --dataset_path data/R2G_hr_dataset/ --output_dir data/R2G_hr_dataset_processed/

# combine JSON files into single JSON file per split
python dataset_prep/combine_json.py \
    --input data/R2G_hr_dataset_processed/ \
    --output data/R2G_hr_dataset_processed_v1/ \