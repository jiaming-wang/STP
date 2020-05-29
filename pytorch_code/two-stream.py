#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-05-18 15:27:45
@LastEditTime: 2020-05-30 02:15:24
@Description: file content
'''
import numpy as np
import os
import scipy.io as scio
import csv
import pandas as pd 

def main():
    path = './txt'
    path1 = './txt-rgb'
    all_num = 0
    true_num = 0
    for root, dirs, files in os.walk(path1):
        for filespath in files:
            txt_name = filespath
            rgb_txt = path1 + '/' + txt_name
            flow_txt = path + '/' + txt_name
            rgb_list = []
            flow_list = []
            all_list = []
            csv_rgb = pd.read_csv(rgb_txt)
            csv_flow = pd.read_csv(flow_txt)
 
            result = np.array(csv_rgb)[1:,:] + np.array(csv_flow)

            probs = np.argmax(result/2, axis=1)
            label = txt_name.split('_')[1]
            if label == 'Inshore':
                num_label = 0
            elif label == 'Offshore':
                num_label = 1
            elif label == 'Neg':
                num_label = 2
            elif label == 'Traffic':
                num_label = 3
            probs = probs - num_label
            all_num = all_num + len(probs)
            true_num = true_num + np.sum(probs == 0)

    print(true_num/all_num)

def load_mat():
    path = './mat'
    for root, dirs, files in os.walk(path):
        for filespath in files:

            all_list = scio.loadmat(path + '/' + filespath)['data']

            rgb_list1 = all_list[0][0]
            flow_list = all_list[0][1]

            rgb_list = rgb_list1[1:]

            for i in range(0, len(rgb_list)):

                result = np.add(np.array(rgb_list[i]) + np.array(flow_list[i]))


if __name__ == '__main__':
    main()
    # load_mat()