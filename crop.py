import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import argparse
from folder import ImageFolder, collate_fn

parser = argparse.ArgumentParser("crop")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save', type=str, required=True)
config, _ = parser.parse_known_args()

import cv2
from matlab_cp2tform import get_similarity_transform_for_cv2

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (128, 128)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    tfm += np.array([[0, 0, 16], [0, 0, 8]])
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

import face_alignment
import numpy as np
import scipy.misc
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

def image_alignment(image):
    eye_left = (pred[36] + pred[39]) // 2
    eye_right = (pred[42] + pred[45]) // 2
    nose = pred[30] // 1
    mouth_left = pred[48]
    mouth_right = pred[54]
    img = alignment(image, [eye_left, eye_right, nose, mouth_left, mouth_right])

def make_dir(dir):
    try:
        os.mkdir(dir)
    except Exception as e:
        pass

if __name__ == '__main__':
    data_root = config.dataset
    save_root = config.save
    save_dir = save_root + '/{}'
    save_path = save_dir + '/{}'
    dataset = ImageFolder(root=data_root)
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    print('data loaded')
    print(len(dataset))

    make_dir(save_root)
    maked_label = {}
    for i, it in enumerate(dataloader):
        for image, label, fname in it:
            if label not in maked_label:
                maked_label[label] = 0
                make_dir(save_dir.format(label))

            try:
                preds = fa.get_landmarks(image)
            except Exception as e:
                print(label, fname, 'error')
                continue
                
            if not preds or len(preds) != 1:
                continue
                
            pred = preds[0]
            if all(np.max(pred, axis=0) - np.min(pred, axis=0) < 128):
                continue
                
            eye_left = (pred[36] + pred[39]) // 2
            eye_right = (pred[42] + pred[45]) // 2
            nose = pred[30]
            mouth_left = pred[48]
            mouth_right = pred[54]
            
            img = alignment(image, [eye_left, eye_right, nose, mouth_left, mouth_right])

            if np.sum(np.max(img, axis=2) == 0) >= 1638:
                print(label, fname, 'cut')
                continue

            print(save_path.format(label, fname))
            scipy.misc.imsave(save_path.format(label, fname), img)
            maked_label[label] += 1

