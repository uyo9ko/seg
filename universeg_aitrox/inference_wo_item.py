import os
import pathlib
import itertools
import cv2
import numpy as np
from PIL import Image
import torch
from dataclasses import dataclass
import SimpleITK as sitk
from typing import Tuple
from universeg import universeg
import argparse
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TestData:
    data: list
    names: list

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = Image.open(path)
    img = img.resize(size, resample=Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img

def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = Image.open(path)
    seg = seg.convert("L")
    seg = seg.resize(size, resample=Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 255])
    seg = seg.astype(np.float32)
    return seg

def check_strings(str1, str2):
    split_str1 = str1[:-4].split('_')
    split_str2 = str2[:-4].split('_')
    if split_str1[0] == split_str2[0] and int(split_str1[1]) == int(split_str2[1]):
        return True
    else:
        return False

def load_folder(path, size: Tuple[int, int] = (256, 256)):
    data = []
    for file in os.listdir(os.path.join(path, 'image')):
        img = process_img(os.path.join(path, 'image', file), size=size)
        seg_file = [seg_filename for seg_filename in os.listdir(os.path.join(path, 'mask'))]# if check_strings(seg_filename, file)]
        seg = process_seg(os.path.join(path, 'mask', seg_file[0]), size=size)
        data.append((img / 255.0, seg))
    return data

def load_test_folder(path, size: Tuple[int, int] = (256, 256)):
    data = []
    file_names = []
    for file in os.listdir(os.path.join(path)):
        img = process_img(os.path.join(path, file), size=size)
        data.append(img / 255.0)
        file_names.append(file)
    return TestData(data, file_names)

@torch.no_grad()
def inference(model, image, support_images, support_labels):
    image = image.to(device)
    logits = model(image[None], support_images[None], support_labels[None])[0]
    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)
    return {'Image': image, 'Soft Prediction': soft_pred, 'Prediction': hard_pred}

def overlay_mask_on_image(image, mask, alpha=0.5):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = np.squeeze(mask)
    overlay = cv2.addWeighted(image, 1, red_mask, alpha, 0)
    return overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform semantic segmentation using UniverseG model.')
    parser.add_argument('-sd','--supportdir', type=str, required=True, help='Directory path of the support images.')
    parser.add_argument('-t','--testsetdir', type=str, required=True, help='Directory path of the test images.')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512], help='Image size for resizing. Default is 512x512.')
    parser.add_argument('--support_size', type=int, default=None, help='Number of support images to use. Default is all.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use. Default is "cpu".')
    args = parser.parse_args()

    support_dir = args.supportdir
    test_set_dir = args.testsetdir
    size = tuple(args.size)
    device = args.device

    items = os.listdir(support_dir)
    model = universeg(pretrained=True)
    model.eval()
    model = model.to(device)


    save_test_mask_path = os.path.join(test_set_dir , 'res_mask')
    save_test_vis_path = os.path.join(test_set_dir , 'res_vis')
    os.makedirs(save_test_mask_path, exist_ok=True)
    os.makedirs(save_test_vis_path, exist_ok=True)
    s_data = load_folder(support_dir, size)
    if args.support_size:
        support_size = args.support_size
    else:
        support_size = len(s_data)
    support_images, support_labels = zip(*itertools.islice(s_data, support_size))
    support_images = torch.tensor(support_images).unsqueeze(1).to(device)
    support_labels = torch.tensor(support_labels).to(device)

    t_data = load_test_folder(test_set_dir+'/image', size)
    item_images = {}

    for i, img in enumerate(t_data.data):
        print(f"Processing image {i+1}/{len(t_data.data)}")
        image = torch.tensor(img).unsqueeze(0).to(device)
        vals = inference(model, image, support_images, support_labels)

        image_np = vals['Image'].cpu().numpy().transpose(1, 2, 0)
        mask_np = vals['Prediction'].cpu().numpy().transpose(1, 2, 0)

        mask_np = np.uint8(mask_np * 255)
        image_np = np.uint8(image_np * 255)

        overlay = overlay_mask_on_image(image_np, mask_np,0.2)
        cv2.imwrite(os.path.join(save_test_vis_path, t_data.names[i]), overlay)
        cv2.imwrite(os.path.join(save_test_mask_path, t_data.names[i]), mask_np)

