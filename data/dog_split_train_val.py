import global_configs
import os
import scipy.io as sio
import shutil

full_path = global_configs.DOG_FULL_DATA_PATH
folder_dir = os.path.dirname(os.path.dirname(full_path))

train_mat = sio.loadmat(os.path.join(folder_dir, "train_list.mat"))
test_mat = sio.loadmat(os.path.join(folder_dir, "test_list.mat"))

train_list = []
test_list = []
for file in train_mat['file_list']:
    train_list.append(file[0][0])
for file in test_mat['file_list']:
    test_list.append(file[0][0])

for folder_dir in [global_configs.DOG_TRAIN_DATA_PATH, global_configs.DOG_TEST_DATA_PATH]:
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)

    shutil.copytree(full_path, folder_dir, ignore=shutil.ignore_patterns('*.jpg'))

for type_list in [train_list, test_list]:
    if type_list == train_list:
        folder_dir = global_configs.DOG_TRAIN_DATA_PATH
    else:
        folder_dir = global_configs.DOG_TEST_DATA_PATH

    for file in type_list:
        shutil.copy(os.path.join(full_path, file), os.path.join(folder_dir, file))

file_cnt = 0
for root, dirs, files in os.walk(global_configs.DOG_TRAIN_DATA_PATH):
    file_cnt += len(files)

print(f"Total files in train folder: {file_cnt}")

file_cnt = 0
for root, dirs, files in os.walk(global_configs.DOG_TEST_DATA_PATH):
    file_cnt += len(files)

print(f"Total files in test folder: {file_cnt}")
