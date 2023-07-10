import numpy as np
import os
# Load the PNG mask image
import SimpleITK as sitk
from PIL import Image

data_dir = '/fileser51/zhengxb/RJJSZ_0617_2/test_image_files'
for item in os.listdir(data_dir):
    pngdir_path = os.path.join(data_dir, item, 'res_mask')
    save_dir = os.path.join(data_dir, item, 'res_mha')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for png_file in os.listdir(pngdir_path):
        png_image = Image.open(os.path.join(pngdir_path, png_file))
        png_image = png_image.convert("L")  # Convert to grayscale if necessary
        # Convert the PIL image to SimpleITK image
        mask_image = sitk.GetImageFromArray(np.array(png_image))
        # Write the mask image to MHA file
        sitk.WriteImage(mask_image, os.path.join(save_dir, png_file.replace('.png', '.mha')))
