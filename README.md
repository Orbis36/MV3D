* This repo is for implementing MV3D from this paper: https://arxiv.org/abs/1611.07759 * 

* This implementation are completed based on the open resource by [bostondidi](https://github.com/bostondiditeam/MV3D) 

# Modification
- The data loader are changed and make it can be tested on KITTI-3D detection dataset.
- Front stream of MV3D are implemented in this repository.
- New data augmentor are implemented from scratch here, include Scale, Translate and Flip.
- Deep fusion method which mentioned in their paper are implemented here.
- New evaluation related codes are added here.
- The ROI-Pooling and NMS related compile configuration files are changed here inorder to adapt 30-series and 10-series graphics cards.

# Final outcome

Since here just half size of BEV(a lower resolution) are used for better efficiency and the original half-channel 
Imagenet pretrain weight cannot be obtained here, we still have some gap compared with the AP in the paper, but this is the
nearest AP which are open sourced so far. Considering it's a paper published 6 years ago, all AP are calculated under 0.5 IOU

The training here is performed on a single RTX3090. Here we use all augmentation method mentioned above with Adam, lr=0.001 for
100k iteration and 1/10 lr for 20k iteration finetune.

|                 | training time |   CarR40@Easy    | CarR40@Moderate  | CarR40@Hard  | 
|-----------------|--------------:|:----------------:|:----------------:|:------------:|
| Paper           | not mentioned |      96.02       |      89.05       |    88.38     | 
| This repository |     ~12 hours |      90.51       |      88.48       |    79.87     |


# Key Dependency
- 10/30 series card are better as they tested for compile the CUDA codes.
- TensorFlow 1.15.4 by Nvidia are needed.
- GCC 7.5.0 and G++ 8.4.0.
- Please check the env.yaml for detailed information.

# File Structure
Please refer to the Readme.md of [bostondidi](https://github.com/bostondiditeam/MV3D) for detailed way to organize the files.  
After that, please change the following line to your own dictionary to the first layer folder of dataset.
```angular2html
__C.RAW_DATA_SETS_DIR = '/media/tianranliu/HC310-4T/KITTI/3D-Object' # src/config.py
```

# Training steps
If you are using 3090, no change need to run the codes. However, for other types of cards, please refer the NVIDIA Graphic Card 
CUDA capabilities and change the figure after 'SM' in the 'src/net/lib/' folder.

```
# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   6.1		           Titan X so CUDA_MODEL = 61
#   8.6                    RTX 3090
#   Other Nvidia shader models should work, but they will require extra startup
#   time as the code is pre-optimized for them.
CUDA_MODELS=30 35 37 52 60 61
```
For training, you need to provide iteration and tags as following, basically we need all parts of network, so in -t, all
parts of subnetwork, you can also modifiy the name of it in line 28 of mv3d_net.py.
```angular2html
-n <name of tags> -i <number of iteration> -t top_view_rpn,image_feature,fusion,front_feature_net # training from scarch
```
To continue training, use -c and -w like:
```angular2html
-n <name of tags> -i <number of iteration> -t top_view_rpn,image_feature,fusion,front_feature_net 
    -c Ture -w top_view_rpn,image_feature,fusion,front_feature_net
```
To get inference output, using following command:
```angular2html
-n <name of test tag> -w <name of the tag of the weight you want use>
```
To get final AP:
```angular2html
cd ./eval
./evaluate_object_3d_offline <abs address to folder label2> ./prediction
```