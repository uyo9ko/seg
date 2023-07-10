# import requests
# def download_file(url, file_path):
#     response = requests.get(url, stream=True)
#     with open(file_path, 'wb') as out_file:
#         out_file.write(response.content)
# download_file('http://proxima.cn-sh2.ufileos.com/tmp/202306261048/mask/1.2.840.113704.7.32.0.416.44735569778971425311136757179911150814_2.mha?UCloudPublicKey=TOKEN_1e516ca2-3866-4401-ac0a-6d47852df61c&Expires=1690167522&Signature=a/22NRRuPgSgJJrZrNabUqpnesM=','/fileser51/zhengxb/RJJSZ_0617_3/platform_test/mha_files/1.2.840.113704.7.32.0.416.44735569778971425311136757179911150814_2.mha')
import SimpleITK as sitk
import numpy as np
# mha_file = sitk.ReadImage('/fileser51/zhengxb/RJJSZ_0617_3/test_image_files/res_mha/1.2.840.113704.7.32.0.416.44735569778971425311136757179911150814/1.2.840.113704.7.32.0.416.44735569778971425311136757179911150814_1.mha')
mha_file= sitk.ReadImage("/fileser51/zhengxb/RJJSZ_0617_3/test_image_files_1/res_mha/1.2.840.113704.7.32.0.416.10808053043591086622359476660957622655/1.2.840.113704.7.32.0.416.10808053043591086622359476660957622655_0.mha")
mha_arr = sitk.GetArrayFromImage(mha_file)
count =0
for i in range(mha_arr.shape[0]):
    if np.any(mha_arr[i]):
        count+=1
print(mha_arr.dtype)
print(mha_arr.shape)
print(mha_arr.max())
print(np.unique(mha_arr))
print(count)