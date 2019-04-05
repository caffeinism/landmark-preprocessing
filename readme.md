# face alignment and landmark preprocessing

The script is very dirty because it is a script I made for temporary use. But it does work.

Image preprocessing for [FaceGAN](https://github.com/MakeDirtyCode/FaceGAN)

## Requirements

- Python3
- Pytorch
- opencv-python
- numpy

## Usage

crop
```
python crop.py --dataset=dataset_dir --save=save_dir
```

landmark extract
```
python landmark.py --dataset=dataset_dir --save=save_dir
```

## Result

crop

![1](result/testset-crop/testset/1.jpg)
![2](result/testset-crop/testset/2.jpg)
![3](result/testset-crop/testset/3.jpg)
![4](result/testset-crop/testset/4.jpg)

landmark

![1](result/testset-landmark/testset/1.png)
![2](result/testset-landmark/testset/2.png)
![3](result/testset-landmark/testset/3.png)
![4](result/testset-landmark/testset/4.png)


## Reference

[face_alignment](https://github.com/1adrianb/face-alignment)

[matlab_cp2tform](https://github.com/clcarwin/sphereface_pytorch)