from scipy.io import loadmat
import os


labels_mat = loadmat('imagelabels.mat')
mat_data = loadmat('setid.mat')
labels = labels_mat['labels'][0]

if not os.path.exists('train'):
    os.mkdir('train')

if not os.path.exists('dev'):
    os.mkdir('dev')

if not os.path.exists('test'):
    os.mkdir('test')

files = os.listdir('jpg')
files.sort()

# Move the files in separate directories, and rename the files to contain the label
# The train dataset has 1000 samples and the test has 6000, so I switched them to have
# more training data.
for file_id in mat_data['trnid'][0]:
    crt_label = labels[file_id - 1]
    original_name = files[file_id - 1]
    str_id = original_name.split('.')[0].split("_")[1]
    new_file_name = 'image_' + str(crt_label) + "_" + str_id + '.jpg'
    os.rename(os.path.join('jpg', original_name), os.path.join('test', new_file_name))

for file_id in mat_data['valid'][0]:
    crt_label = labels[file_id - 1]
    original_name = files[file_id - 1]
    str_id = original_name.split('.')[0].split("_")[1]
    new_file_name = 'image_' + str(crt_label) + "_" + str_id + '.jpg'
    os.rename(os.path.join('jpg', original_name), os.path.join('dev', new_file_name))

for file_id in mat_data['tstid'][0]:
    crt_label = labels[file_id - 1]
    original_name = files[file_id - 1]
    str_id = original_name.split('.')[0].split("_")[1]
    new_file_name = 'image_' + str(crt_label) + "_" + str_id + '.jpg'
    os.rename(os.path.join('jpg', original_name), os.path.join('train', new_file_name))