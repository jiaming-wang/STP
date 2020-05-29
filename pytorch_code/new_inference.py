#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-11-05 08:42:43
@LastEditTime: 2020-05-30 02:16:47
@Description: file content
'''

import torch
import numpy as np
from network import C3D_model, R2Plus1D_model,R3D_model, p3d_model, I3D_model
import cv2
import os
torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame, crop_size):

    frame = frame[8:int(8 + crop_size), 30: int(30+crop_size), :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ferryboat_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    num_classes = 4
    modelName = 'I3D'
    if modelName == 'I3D':
      model = I3D_model.InceptionI3d(num_classes=num_classes, in_channels=3)
      size = (240, 284)
      crop_size = 224
    elif modelName == 'R2Plus1D':
      model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
      size = (171, 128)
      crop_size = 112
    elif modelName == 'C3D':
      model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
      size = (171, 128)
      crop_size = 112
    elif modelName == 'P3D':
      model = p3d_model.P3D63(num_classes=num_classes)
      size = (176, 210)
      crop_size = 160
    elif modelName == 'R3D':
      model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
      size = (171, 128)
      crop_size = 112

    checkpoint = torch.load('./models/I3D-ferryboat4_epoch-199.pth.tar', map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
    model_dict.update(checkpoint_load)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    
    for root, dirs ,files in os.walk('./VAR/ferryboat/test/'):

        l_names = locals()

        l_names['Inshore'] = 0
        l_names['Neg'] = 0
        l_names['Offshore'] = 0
        l_names['Traffic'] = 0
        l_names['Inshore1'] = 0
        l_names['Neg1'] = 0
        l_names['Offshore1'] = 0
        l_names['Traffic1'] = 0

        if len(dirs) > 4:
            video_name = dirs
            for name in video_name:
                class_name = name.split('_')[1]
                video = './ferryboat/' + class_name + "/" + name + '.avi'
                clip = []
                print(video)
                cap = cv2.VideoCapture(video)
                retaining = True
                while retaining:
                    retaining, frame = cap.read()
                    if not retaining and frame is None:
                        continue
    
                    tmp_ = center_crop(cv2.resize(frame, size), crop_size)
                    tmp = tmp_
   
                    clip.append(tmp)
                    if len(clip) == 16:
                        inputs = np.array(clip).astype(np.float32)
                        inputs = np.expand_dims(inputs, axis=0)
                        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                        inputs = torch.from_numpy(inputs)
                        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                        with torch.no_grad():
                            outputs, index = model.forward(inputs)
                        iii = index.cpu().data
                        print(iii)
                        probs = torch.nn.Softmax(dim=1)(outputs)
                        label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                        if modelName == 'I3D':
                            label = int(label[0])
                        pre = class_names[label].split(' ')[1][:-1]
                        l_names[str(class_name)] = l_names[str(class_name)] + 1

                        if str(pre) == str(class_name):
                            l_names[str(class_name) + '1'] = l_names[str(class_name)+ '1'] + 1
                        elif str(pre) == 'Ne' and str(class_name) == 'Neg':
                            l_names[str(class_name) + '1'] = l_names[str(class_name)+ '1'] + 1
                        elif str(pre) == 'Traffi' and str(class_name) == 'Traffic':
                            l_names[str(class_name) + '1'] = l_names[str(class_name)+ '1'] + 1

                        clip.pop(0)

                    cv2.waitKey(30)

                cap.release()
                cv2.destroyAllWindows()
            print(str(class_name)+ '_acc:' + str(int(l_names[str(class_name)+ '1']) / int(l_names[str(class_name)])))



if __name__ == '__main__':
    main()









