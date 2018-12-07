# GANimation-tf
A tensorflow implementation of GANimation
paper: https://arxiv.org/abs/1807.09251
Author's implementation: https://github.com/albertpumarola/GANimation

## Requirements
- ubuntu 14.04
- python 3.6
- opencv 3.4.3
- tensorflow-gpu 1.12
- face-recognition 1.2.3

## Data
1. Apply for [Emotionet dataset](http://cbcsl.ece.ohio-state.edu/emotionet.html) and select 20k images.
2. Extract AU feature by [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units).
```
FeatureExtraction -fdir src_image_dir -aus
```
3. Detect and crop face by [face-recognition](https://pypi.org/project/face_recognition/) library and resize to 128x128x3.
4. Generate 'aus.pkl', (please refer to paper author's github) and move it to 'data/'.
5. move the face images to folder 'data/imgs'.

## Training
```
python train.py
```

## Testing
```
python test.py
```
