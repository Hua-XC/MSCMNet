import numpy as np
from PIL import Image
import pdb
import os


data_path = './dataset/SYSU-MM01/'
rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
#'/root/HXC/reid/dataset/SYSU_MM01/exp/train_id.txt'
file_path_train = os.path.join(data_path,'exp/train_id.txt')
#'/root/HXC/reid/dataset/SYSU_MM01/exp/val_id.txt'
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    #[1,2,3,4,5,...]
    ids = [int(y) for y in ids[0].split(',')]
    #['0001','0002',...]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    #[334,335,336,337,...]
    ids = [int(y) for y in ids[0].split(',')]
     #['0334','0335',...]
    id_val = ["%04d" % x for x in ids]
    
# combine train and val split   
id_train.extend(id_val) #id_val
#print(id_train.extend(id_val) )
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        #'/root/HXC/reid/dataset/SYSU_MM01/cam6/0533'
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
            
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
#
#files_rgb files_ir
# relabel
pid_container = set()
#img_path：'/root/HXC/reid/dataset/SYSU_MM01/cam3/0533/0020.jpg'
for img_path in files_ir:
    #533
    pid = int(img_path[-13:-9])
    #{1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, ...,533}
    pid_container.add(pid)
#{1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 18: 12, 19: 13, ...}
pid2label = {pid:label for label, pid in enumerate(pid_container)}
fix_image_width = 192
fix_image_height = 384

#train_image
def read_imgs(train_image):
    train_img = []
    train_label = []
    #img_path：'/root/HXC/reid/dataset/SYSU_MM01/cam3/0533/0020.jpg'
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        #pix_array.shape
        #(288, 144, 3)
        pix_array = np.array(img)

        train_img.append(pix_array) 
        
        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)
       
# rgb imges
#['/root/HXC/reid/dataset/SYSU_MM01/cam1/0001/0001.jpg', '/root/HXC/reid/datas...1/0002.jpg', '/root/HXC/reid/datas...1/0003.jpg',
# '/root/HXC/reid/datas...1/0004.jpg', '/root/HXC/reid/datas...1/0005.jpg', '/root/HXC/reid/datas...1/0006.jpg', 
# '/root/HXC/reid/datas...1/0007.jpg', '/root/HXC/reid/datas...1/0008.jpg', '/root/HXC/reid/datas...1/0009.jpg', 
# '/root/HXC/reid/datas...1/0010.jpg', '/root/HXC/reid/datas...1/0011.jpg', '/root/HXC/reid/datas...1/0012.jpg', 
# '/root/HXC/reid/datas...1/0013.jpg', '/root/HXC/reid/datas...1/0014.jpg', ...]
train_img, train_label = read_imgs(files_rgb)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
#files_ir:
#['/root/HXC/reid/dataset/SYSU_MM01/cam3/0001/0001.jpg', '/root/HXC/reid/datas...1/0002.jpg', '/root/HXC/reid/datas...1/0003.jpg', 
# '/root/HXC/reid/datas...1/0004.jpg', '/root/HXC/reid/datas...1/0005.jpg', '/root/HXC/reid/datas...1/0006.jpg', 
# '/root/HXC/reid/datas...1/0007.jpg', '/root/HXC/reid/datas...1/0008.jpg', '/root/HXC/reid/datas...1/0009.jpg', 
# '/root/HXC/reid/datas...1/0010.jpg', '/root/HXC/reid/datas...1/0011.jpg', '/root/HXC/reid/datas...1/0012.jpg', 
# '/root/HXC/reid/datas...1/0013.jpg', '/root/HXC/reid/datas...1/0014.jpg', ...]
train_img, train_label = read_imgs(files_ir)
#train_img:
#train_label:array([  0,   0,   0, ..., 394, 394, 394])
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)
print(data_path + 'train_ir_resized_label.npy')


#train_img:
# #array([[[[167, 167, 167],
#          [168, 168, 168],
#          [168, 168, 168],
#          ...,
#          [156, 156, 156],
#          [155, 155, 155],
#          [155, 155, 155]],

#         [[167, 167, 167],
#          [168, 168, 168],
#          [168, 168, 168],
#          ...,
#          [154, 154, 154],
#          [154, 154, 154],
#          [154, 154, 154]],

#         [[167, 167, 167],
#          [168, 168, 168],
#          [168, 168, 168],
#          ...,
#          [152, 152, 152],
#          [152, 152, 152],
#          [153, 153, 153]],

#         ...,

#         [[103, 103, 103],
#          [103, 103, 103],
#          [103, 103, 103],
#          ...,
#          [101, 101, 101],
#          [101, 101, 101],
#          [101, 101, 101]],

#         [[104, 104, 104],
#          [104, 104, 104],
#          [104, 104, 104],
#          ...,
#          [102, 102, 102],
#          [102, 102, 102],
#          [102, 102, 102]],

#         [[105, 105, 105],
#          [105, 105, 105],
#          [105, 105, 105],
#          ...,
#          [103, 103, 103],
#          [103, 103, 103],
#          [103, 103, 103]]],


#        [[[143, 143, 143],
#          [143, 143, 143],
#          [143, 143, 143],
#          ...,
#          [124, 124, 124],
#          [124, 124, 124],
#          [124, 124, 124]],

#         [[144, 144, 144],
#          [144, 144, 144],
#          [144, 144, 144],
#          ...,
#          [124, 124, 124],
#          [124, 124, 124],
#          [124, 124, 124]],

#         [[145, 145, 145],
#          [144, 144, 144],
#          [144, 144, 144],
#          ...,
#          [125, 125, 125],
#          [125, 125, 125],
#          [125, 125, 125]],

#         ...,

#         [[ 90,  90,  90],
#          [ 90,  90,  90],
#          [ 90,  90,  90],
#          ...,
#          [ 81,  81,  81],
#          [ 83,  83,  83],
#          [ 84,  84,  84]],

#         [[ 90,  90,  90],
#          [ 90,  90,  90],
#          [ 90,  90,  90],
#          ...,
#          [ 85,  85,  85],
#          [ 87,  87,  87],
#          [ 87,  87,  87]],

#         [[ 90,  90,  90],
#          [ 90,  90,  90],
#          [ 90,  90,  90],
#          ...,
#          [ 90,  90,  90],
#          [ 90,  90,  90],
#          [ 90,  90,  90]]],


#        [[[133, 133, 135],
#          [127, 127, 129],
#          [118, 118, 120],
#          ...,
#          [146, 146, 146],
#          [145, 145, 145],
#          [142, 142, 142]],

#         [[138, 138, 140],
#          [131, 131, 133],