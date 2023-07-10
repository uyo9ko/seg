import os
import pandas as pd
import requests
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import pydicom
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download and process medical image files.')
    parser.add_argument('--mha_df_path', type=str, required=True, help='Path of the mha data CSV file.')
    parser.add_argument('--img_df_path', type=str, required=True, help='Path of the image data CSV file.')
    parser.add_argument('--data_save_dir', type=str, required=True, help='Directory to save downloaded and processed files.')
    parser.add_argument('--mha_region_col_name', type=str, default='备注',required=True, help='Column name in mha_df for MHA region types.')
    parser.add_argument('--ww', type=int, default=400, help='Window width for DICOM image processing.')
    parser.add_argument('--wc', type=int, default=50, help='Window center for DICOM image processing.')
    parser.add_argument('--dcm_url_col_name', type=str, default='文件内网地址', help='Column name in img_df for DCM file URLs.')
    parser.add_argument('--dcm_seq_number_col_name', type=str, default='序列号', help='Column name in img_df for DCM sequence numbers.')
    parser.add_argument('--dcm_seq_index_col_name', type=str, default='文件编号', help='Column name in img_df for DCM sequence index.')
    parser.add_argument('--mha_url_col_name', type=str, default='影像结果', help='Column name in mha_df for MHA file URLs.')
    parser.add_argument('--mha_seq_number_col_name', type=str, default='序列编号', help='Column name in mha_df for MHA sequence numbers.')
    args = parser.parse_args()
    return args

# Function to download file
def download_file(url, file_path):
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as out_file:
        out_file.write(response.content)

def get_dcm_file(dcm_path, new_window_width=350,new_window_center = 45):
    dcm = pydicom.dcmread(dcm_path)
    pixel_array = dcm.pixel_array
    pixel_array = pixel_array.astype(np.float32)
    if hasattr(dcm,'RescaleIntercept') and hasattr(dcm,'RescaleSlope'):
        ri = dcm.RescaleIntercept
        rs = dcm.RescaleSlope
        pixel_array = pixel_array*np.float32(rs)+np.float32(ri)
    res = np.clip(pixel_array,new_window_center-new_window_width//2,new_window_center+new_window_width//2)
    return res

if __name__ == "__main__":
    args = parse_args()
    data_save_dir = args.data_save_dir 
    ww = args.ww
    wc = args.wc
    mha_region_col_name = args.mha_region_col_name
    dcm_url_col_name = args.dcm_url_col_name
    dcm_seq_number_col_name = args.dcm_seq_number_col_name
    dcm_seq_index_col_name = args.dcm_seq_index_col_name
    mha_url_col_name = args.mha_url_col_name
    mha_seq_number_col_name = args.mha_seq_number_col_name


    dcm_save_dir = data_save_dir + '/raw_data/dcm_files'
    mha_save_dir = data_save_dir + '/raw_data/mha_files'

    # Load CSV files
    mha_df = pd.read_csv(args.mha_df_path)
    img_df = pd.read_csv(args.img_df_path)

    # Ensure download directories exist
    os.makedirs(dcm_save_dir, exist_ok=True)
    os.makedirs(mha_save_dir, exist_ok=True)

    # Download DCM files
    total_dcm = len(img_df)
    for index, row in img_df.iterrows():
        url = row[dcm_url_col_name]
        filename = f"{row[dcm_seq_number_col_name]}_{row[dcm_seq_index_col_name]}.dcm"
        file_path = os.path.join(dcm_save_dir, filename)
        if os.path.exists(file_path):
            continue
        try:
            download_file(url, file_path)
            print(f"Downloading DCM: {index + 1}/{total_dcm}", file_path)
        except Exception as e:
            print(f"Error downloading DCM file {file_path}: {e}")

    # Download MHA files
    total_mha = len(mha_df)
    for index, row in mha_df.iterrows():
        url = row[mha_url_col_name]
        filename = f"{row[mha_seq_number_col_name]}_{row[mha_region_col_name]}_{index}.mha"
        file_path = os.path.join(mha_save_dir, filename)
        if os.path.exists(file_path):
            continue
        try:
            download_file(url, file_path)
            print(f"Downloading MHA: {index + 1}/{total_mha}",file_path)
        except Exception as e:
            print(f"Error downloading MHA file {file_path}: {e}")

    total_mha = len(mha_df)
    bad_list = []

    for index, row in mha_df.iterrows():
        print(f"Processing MHA: {index + 1}/{total_mha}")
        if not isinstance(row[mha_region_col_name], str):
            row[mha_region_col_name] = str(row[mha_region_col_name])
        folder = os.path.join(data_save_dir, 'image_files', row[mha_region_col_name])
        os.makedirs(os.path.join(folder, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'image'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'org_vis'), exist_ok=True)

        mha_filepath = os.path.join(mha_save_dir, f"{row[mha_seq_number_col_name]}_{row[mha_region_col_name]}_{index}.mha")
        if not os.path.exists(mha_filepath):
            print(f"MHA file does not exist: {mha_filepath}")
            bad_list.append(mha_filepath)
            continue

        try:
            mha_file = sitk.ReadImage(mha_filepath)
            mha_arr = sitk.GetArrayFromImage(mha_file)
        except Exception as e:
            print(f"Error reading MHA file {mha_filepath}: {e}")
            bad_list.append(mha_filepath)
            continue

        mhalen = mha_arr.shape[0]

        for layer_index, layer in enumerate(mha_arr):
            print(f"Processing MHA {index + 1}/{total_mha}, layer {layer_index + 1}/{mhalen}")
            if np.any(layer > 0):
                save_name = f'{row[mha_seq_number_col_name]}_{row[mha_region_col_name]}_{layer_index}_{mhalen}.png'
                plt.imsave(os.path.join(folder, 'mask', save_name), layer, cmap='gray')
                dcm_filepath = os.path.join(dcm_save_dir, f"{row[mha_seq_number_col_name]}_{mhalen+1-layer_index}.dcm")
                if not os.path.exists(dcm_filepath):
                    print(f"DCM file does not exist: {dcm_filepath}")
                    bad_list.append(dcm_filepath)
                    continue

                try:
                    dcm_arr = get_dcm_file(dcm_filepath, new_window_width=ww, new_window_center=wc)
                except Exception as e:
                    print(f"Error reading DCM file {dcm_filepath}: {e}")
                    bad_list.append(dcm_filepath)
                    continue

                plt.imsave(os.path.join(folder, 'image', save_name), dcm_arr, cmap='gray')

                # Create overlay image
                layer_rgb = cv2.cvtColor(layer.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                dcm_arr = cv2.normalize(dcm_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                dcm_arr_rgb = cv2.cvtColor(dcm_arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)

                layer_rgb[:, :, 2] = 0
                layer_rgb[:, :, 1] = 0  # only red channel
                overlay_img = cv2.addWeighted(dcm_arr_rgb, 1, layer_rgb * 255, 0.2, 0)

                plt.imsave(os.path.join(folder, 'org_vis',save_name), overlay_img)

    print('These files had issues:')
    print(bad_list)
