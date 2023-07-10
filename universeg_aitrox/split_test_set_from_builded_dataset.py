import os
import shutil
import random

num_test_files = 10
image_data_dir = '/fileser51/zhengxb/RJJSZ_0617_3/image_files'
test_image_data_dir = '/fileser51/zhengxb/RJJSZ_0617_3/test_image_files'
for item in os.listdir(image_data_dir):
    os.makedirs(os.path.join(test_image_data_dir, item+'_test/image'),exist_ok=True)
    os.makedirs(os.path.join(test_image_data_dir, item+'_test/mask'),exist_ok=True)
    os.makedirs(os.path.join(test_image_data_dir, item+'_test/org_vis'),exist_ok=True)
    test_filenames = random.sample(os.listdir(os.path.join(image_data_dir, item, 'image')), num_test_files)
    for test_filename in test_filenames:
        shutil.move(os.path.join(image_data_dir, item, 'image', test_filename), os.path.join(test_image_data_dir, item+'_test/image', test_filename))
        shutil.move(os.path.join(image_data_dir, item, 'mask', test_filename), os.path.join(test_image_data_dir, item+'_test/mask', test_filename))
        shutil.move(os.path.join(image_data_dir, item, 'org_vis', test_filename), os.path.join(test_image_data_dir, item+'_test/org_vis', test_filename))
    