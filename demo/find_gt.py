import numpy as np
import os
from tqdm import tqdm
import shutil
import cv2



if __name__ == '__main__':
    
    ## bdd
    origin_dir = 'E:/DGtask/datasets/bdd10k/bdd100k_sem_seg_labels_trainval/labels/sem_seg/colormaps/val'
    img_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/img/cs2bdd'
    target_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/gt'
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    target_dir = os.path.join(target_dir, 'cs2bdd')
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    img_names = []
    t_img_names = []
    for img_name in os.listdir(img_dir):
        img_names.append(img_name)
        t_img_names.append(img_name.split('_')[1]+'-'+img_name.split('_')[2])
    for i, img_name in tqdm(enumerate(os.listdir(origin_dir))):
        if img_name.strip('.png') in t_img_names:
            gt_pth = os.path.join(origin_dir, img_name)
            new_gt_pth = os.path.join(target_dir, img_name)
            img = cv2.imread(gt_pth)
            img = cv2.resize(img, (1920, 1080))
            cv2.imwrite(new_gt_pth, img)
    
    
    ## gta
    origin_dir = 'E:/DGtask/datasets/GTAV/labels/val'
    img_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/img/cs2gta'
    target_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/gt'
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    target_dir = os.path.join(target_dir, 'cs2gta')
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    img_names = []
    t_img_names = []
    for img_name in os.listdir(img_dir):
        img_names.append(img_name)
        t_img_names.append(img_name.split('_')[1])
    for i, img_name in tqdm(enumerate(os.listdir(origin_dir))):
        if img_name.strip('.png') in t_img_names:
            gt_pth = os.path.join(origin_dir, img_name)
            new_gt_pth = os.path.join(target_dir, img_name)
            img = cv2.imread(gt_pth)
            img = cv2.resize(img, (1920, 1080))
            cv2.imwrite(new_gt_pth, img)
    
    
    ## map
    origin_dir = 'E:/DGtask/datasets/mapillary/validation/labels'
    img_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/img/cs2map'
    target_dir = 'E:/DGtask/DGViT/Mask2Former-main/demo/all_test/gt'
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    target_dir = os.path.join(target_dir, 'cs2map')
    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)
    img_names = []
    t_img_names = []
    for img_name in os.listdir(img_dir):
        img_names.append(img_name)
        t_img_names.append(img_name.split('_')[1])
    for i, img_name in tqdm(enumerate(os.listdir(origin_dir))):
        if str(i) in t_img_names:
            gt_pth = os.path.join(origin_dir, img_name)
            new_gt_pth = os.path.join(target_dir, img_name)
            img = cv2.imread(gt_pth)
            img = cv2.resize(img, (1920, 1080))
            cv2.imwrite(new_gt_pth, img)