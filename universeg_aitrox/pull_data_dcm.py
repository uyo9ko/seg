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
    parser.add_argument('--img_df_path', type=str, required=True, help='Path of the image data CSV file.')
    parser.add_argument('--data_save_dir', type=str, required=True, help='Directory to save downloaded and processed files.')
    parser.add_argument('--dcm_url_col_name', type=str, default='文件内网地址', help='Column name in img_df for DCM file URLs.')
    parser.add_argument('--dcm_seq_number_col_name', type=str, default='序列号', help='Column name in img_df for DCM sequence numbers.')
    parser.add_argument('--dcm_seq_index_col_name', type=str, default='文件编号', help='Column name in img_df for DCM sequence index.')
    args = parser.parse_args()
    return args

# Function to download file
def download_file(url, file_path):
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as out_file:
        out_file.write(response.content)


if __name__ == "__main__":
    args = parse_args()
    data_save_dir = args.data_save_dir 
    dcm_url_col_name = args.dcm_url_col_name
    dcm_seq_number_col_name = args.dcm_seq_number_col_name
    dcm_seq_index_col_name = args.dcm_seq_index_col_name

    dcm_save_dir = data_save_dir + '/dcm_files'

    # Load CSV files
    img_df = pd.read_csv(args.img_df_path)

    # Ensure download directories exist
    os.makedirs(dcm_save_dir, exist_ok=True)

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

