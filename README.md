# Reference-Guided Contrastive Clustering (RGCC)

This repository contains the implementation of **Reference-Guided Contrastive
Learning and Clustering Constraint (RGCC)** for binary classification of
glioblastoma (GBM) and solitary brain metastasis (SBM) from MRI.


## Environment

Use a Python environment with PyTorch and the following packages installed:

torch == 2.9.0
torchvision == 0.24.0
numpy == 2.2.6
tensorboard == 2.20.0
scikit-learn == 1.7.2
monai == 1.5.1
nibabel == 5.3.2
SimpleITK == 2.5.2
skimage == 0.25.2

The code is configured for CUDA training. Select the GPU through `--gpu_ids`.

## Dataset-list files

Each list is a plain-text file. Each non-empty line contains an image path and
its integer class label, separated by whitespace:

Class labels are:

```text
0  GBM
1  SBM
```

Required list files:

| File argument | Purpose |
| --- | --- |
| `--txt_file_train` | All training samples (GBM and SBM). |
| `--txt_file_test` | All test/validation samples (GBM and SBM). |
| `--txt_file_gbm_in_train` | GBM-only training samples used to construct/update the GBM reference bank. |
| `--txt_file_mate_in_train` | SBM-only training samples used to construct/update the SBM reference bank.

## Training

Run the following command from the repository root. 
python train.py 
  --data_path "D:\\dataset" 
  --txt_file_train "D:\\data_set\\train.txt" 
  --txt_file_test "D:\\data_set\\test.txt" 
  --txt_file_gbm_in_train "D:\\data_set\\train_0.txt" 
  --txt_file_mate_in_train "D:\\data_set\\train_1.txt" 
  --save_folder "D:\\results" 
  --sub_save_folder "rgcc_fold1" 
  --gpu_ids 0 
  --data_shape  
  --crop_scale
  --pre_train "D:\\checkpoints\\pretrained_model.pth"

--data_shape and --crop_scale each take four integers in the order
C D H W (channels, depth, height, width).

If no pretrained checkpoint is used, supply an empty string for --pre_train


## Model Weights

The trained model weights are available in
https://drive.google.com/drive/folders/1nq321XjIdh-4C-F8JL1Pni86nYDFXTM5?usp=drive_link

