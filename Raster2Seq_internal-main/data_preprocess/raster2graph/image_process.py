import os, cv2, numpy as np, json
from tqdm import *
from PIL import Image
import argparse


parser = argparse.ArgumentParser("Preprocess LIFULL HOMES DATA (HIGH RESOLUTION) Dataset")
parser.add_argument('--data_root', type=str, default=r'R2G_hr_dataset/', 
                    help='path to the root folder of the dataset')
args = parser.parse_args()

SIZE = 512
MARGIN = 64
np.set_printoptions(threshold=np.inf, linewidth=999999)

# original_images_path = r'E:/LIFULL HOMES DATA (HIGH RESOLUTION)/photo-rent-madori-full-00'
original_images_path = args.data_root

with open(f'{args.data_root}/annot_json/instances_train.json', mode='r') as f_train:
    train_jpgs = [_['file_name'] for _ in json.load(f_train)['images']]
with open(f'{args.data_root}/annot_json/instances_val.json', mode='r') as f_val:
    val_jpgs = [_['file_name'] for _ in json.load(f_val)['images']]
with open(f'{args.data_root}/instances_test.json', mode='r') as f_test:
    test_jpgs = [_['file_name'] for _ in json.load(f_test)['images']]
jpgs = {'train': train_jpgs, 'val': val_jpgs, 'test': test_jpgs}

start_idx = 0
for mode in ['train', 'val', 'test']:
    output_dir = './' + mode
    os.makedirs(output_dir, exist_ok=True)
    for fnames in [jpgs[mode]]:
        for i in tqdm(range(len(fnames))):
            fn = fnames[i].replace('.jpg', '')
            if os.path.exists(os.path.join(f'{args.data_root}/annot_npy', fn + '.npy')) and \
                os.path.exists(os.path.join(f'{args.data_root}/original_vector_boundary', fn + '.npy')):

                img_original = Image.open(os.path.join(original_images_path, fn.replace('-', '/') + '.jpg'))
                boundary_path = os.path.join(f'{args.data_root}/original_vector_boundary', fn + '.npy')
                boundary = np.load(boundary_path, allow_pickle=True).item()
                x_min = boundary['x_min']
                x_max = boundary['x_max']
                y_min = boundary['y_min']
                y_max = boundary['y_max']
                width = x_max - x_min
                mid_width = (x_max + x_min) / 2
                height = y_max - y_min
                mid_height = (y_max + y_min) / 2
                if width > height:
                    scale = (SIZE - 2 * MARGIN) / width
                else:
                    scale = (SIZE - 2 * MARGIN) / height
                # print(x_min, y_min, x_max, y_max, width, height, scale)

                original_width, original_height = img_original.size
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_image = img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (512, 512), (255, 255, 255))
                # print(new_width, new_height)
                x_topleft_offset = int(512/2 - mid_width * scale)
                y_topleft_offset = int(512/2 - mid_height * scale)
                canvas.paste(scaled_image, (x_topleft_offset, y_topleft_offset))

                canvas.save(os.path.join(output_dir, fn + '.png'))
            
            start_idx += 1