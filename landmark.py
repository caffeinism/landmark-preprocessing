import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import argparse
from folder import ImageFolder, collate_fn

parser = argparse.ArgumentParser("landmark")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save', type=str, required=True)
config, _ = parser.parse_known_args()

import face_alignment
import numpy as np
import scipy.misc
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

landmark_lines = [  ([36, 41], (255, 0, 0), True),
                    ([42, 47], (255, 0, 0), True),
                    ([27, 30], (0, 255, 0), False),
                    ([30, 35], (0, 255, 0), True),
                    ([48, 59], (0, 0, 255), True), #outer mouth
                    ([60, 67], (0, 0, 255), True),] #inner mouth

import cv2
def draw(image, pred):
    for order, color, is_cycle in landmark_lines:
        for i, j in zip(range(order[0], order[1]), range(order[0]+1, order[1]+1)):
            cv2.line(image, tuple(pred[i]), tuple(pred[j]), color, 1)
        if is_cycle:
            cv2.line(image, tuple(pred[order[0]]), tuple(pred[order[-1]]), color, 1)

def make_dir(dir):
    try:
        os.mkdir(dir)
    except Exception as e:
        pass

import numpy as np

if __name__ == '__main__':
    data_root = config.dataset
    landmark_save_root = config.save
    landmark_save_dir = landmark_save_root + '/{}'
    landmark_save_path = landmark_save_dir + '/{}'

    dataset = ImageFolder(root=data_root)
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    print('data loaded')
    print(len(dataset))
    
    make_dir(landmark_save_root)
    maked_label = {}
    for i, it in enumerate(dataloader):
        for image, label, fname in it:
            fname = fname.split('.')[0] + '.png'

            if label not in maked_label:
                maked_label[label] = 0
                make_dir(landmark_save_dir.format(label))

            try:
                preds = fa.get_landmarks(image)
            except Exception as e:
                print(label, fname, 'error')
                continue

            if not preds or len(preds) != 1:
                continue
            
            pred = preds[-1]

            result = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
            draw(result, pred)

            print(landmark_save_path.format(label, fname))
            cv2.imwrite(landmark_save_path.format(label, fname), result)