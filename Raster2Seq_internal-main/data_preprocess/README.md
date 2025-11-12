## Data preprocessing

### Structured3D

Simply download preprocessed data by RoomFormer at [here](https://polybox.ethz.ch/index.php/s/wKYWFsQOXHnkwcG). For more details, please refer to [RoomFormer's instructions](https://github.com/ywyue/RoomFormer/tree/main/data_preprocess).

### CubiCasa5K
Step 1: Download and extract [CubiCasa5K](https://zenodo.org/record/2613548) dataset.

Step 2: Run `bash data_preprocess/cubicasa5k/run.sh`.

### Raster2Graph
The instruction mainly follows Raster2Graph's instruction.

Step 1: Due to dataset proprietary restrictions, please apply for access to LIFULL HOME'S Data [here](https://www.nii.ac.jp/dsc/idr/en/lifull/).

Step 2: After obtaining access, download only the "photo-rent-madori-full-00" folder, which contains approximately 300,000 images. 

Step 3: Apply for access to the annotation [here](https://docs.google.com/forms/d/e/1FAIpQLSexqNMjyvPMtPMPN7bSh_1u4Q27LZAT-S9lR_gpipNIMKV5lw/viewform). 

The package has 3 folders:
- annot_npy, annot_json: the annotations saved in npy and json, respectively.
- original_vector_boundary: boundary boxes of "LIFULL HOME'S Data" which is used to create centered 512x512 images.

These folders should be saved in the same directory as `photo-rent-madori-full-00`. For example: `data/R2G_hr_dataset/`.

Step 4: Run `bash data_preprocess/raster2graph/run.sh`.