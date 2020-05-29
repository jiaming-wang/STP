#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-11-05 08:42:43
@LastEditTime : 2020-01-08 22:51:59
@Description: file content
'''
import torch
import numpy as np
from network import C3D_model, R2Plus1D_model,R3D_model, p3d_model, I3D_model
import cv2
torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    # frame = frame[8:120, 30:142, :]
    # frame = frame[8:168, 30:190, :]
    frame = frame[8:232, 30:254, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    # model = R2Plus1D_model.R2Plus1DClassifier(num_classes=3, layer_sizes=(3, 4, 6, 3))
    # model = C3D_model.C3D(num_classes=3, pretrained=False)
    # model = p3d_model.P3D63(num_classes=3)
    model = I3D_model.InceptionI3d(num_classes=3, in_channels=3)
    # model = R3D_model.R3DClassifier(num_classes=3, layer_sizes=(3, 4, 6, 3))
    checkpoint = torch.load('./models/I3D-ucf50_epoch-199.pth.tar', map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
    model_dict.update(checkpoint_load)
    model.load_state_dict(model_dict)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    # video = './ucf50/Inshore/v_Inshore_g01_c01_5.avi'
    # video = 'video1.avi'
    video = './ucf50/Offshore/v_Offshore_g03_c01.avi'
    #video = './ucf50/Neg/v_Neg_g04_c011.avi'
    cap = cv2.VideoCapture(video)
    retaining = True
    modelName = 'I3D'
    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        # tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        # tmp_ = center_crop(cv2.resize(frame, (176, 210)))
        tmp_ = center_crop(cv2.resize(frame, (240, 284)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            # print(class_names[label].split(' ')[-1].strip())
            if modelName == 'I3D':
                 label = int(label[0])
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









