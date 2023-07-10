# universeg_aitrox

## 1. prepare dataset

run prepare_data.py

need to handle these two path:

'--img_df_path', the img df path is like the "文件内网地址信息-导出结果 (5)"

'--mha_df_path',the mha df path is like 'image_anno_TASK_6787'

## 2. split support set and testset (optional)

run split_test_set_from_builded_dataset.py

## 3. inference the universeg model

run inference.py

-s mean the support set path

-t mean the test set path
