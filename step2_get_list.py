# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:25:39 2018

@author: chlian
"""
import numpy as np
import os

if __name__ == '__main__':

    data_path = './augmentation_vtk_data/'
    output_path = './'
    num_samples = 30 # need to define
    num_augmentations = 20 # need to define
    train_size = 0.8 # need to define

    #training and validation sets
    sample_list = list(range(1, num_samples+1))

    idx = int(np.round(train_size*len(sample_list)))
    train_list, val_list = np.split(sample_list, [idx])
    #test list: 31--36

    #training
    train_name_list = []
    for i_sample in train_list:
        for i_aug in range(num_augmentations):
            print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
            subject_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
            subject2_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample+1000)
            train_name_list.append(os.path.join(data_path, subject_name))
            train_name_list.append(os.path.join(data_path, subject2_name))

    with open(os.path.join(output_path, 'train_list.csv'), 'w') as file:
        for f in train_name_list:
            file.write(f+'\n')

    #validation
    val_name_list = []
    for i_sample in val_list:
        for i_aug in range(num_augmentations):
            print('Computing Sample: {0}; Aug: {1}...'.format(i_sample, i_aug))
            subject_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
            subject2_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample+1000)
            val_name_list.append(os.path.join(data_path, subject_name))
            val_name_list.append(os.path.join(data_path, subject2_name))

    with open(os.path.join(output_path, 'val_list.csv'), 'w') as file:
        for f in val_name_list:
            file.write(f+'\n')

    print('# of train:', len(train_name_list))
    print('# of validation:', len(val_name_list))
