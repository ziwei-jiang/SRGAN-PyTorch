SRGAN-PyTorch
============================
A Pytorch implementation of SRGAN based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

Requirement
----------------------------
* Argparse
* Numpy
* Pillow
* Python 3.7
* PyTorch
* TorchVision
* tqdm


Usage
----------------------------

### Training

Download the data to the ./data/ folder. The [pretrained weight of SRResNet](https://drive.google.com/file/d/126GzYaRBprQYju1g0WGVF_5UvbMIYkh3/view?usp=sharing) is optional
Run the script train.py
```
$ python train.py --trainset_dir $TRAINDIR --validset_dir $VALIDDIR --upscale_factor 4 --pretrain SRResNet_weight.pth --cuda

usage: train.py [-h] [--trainset_dir TRAINSET_DIR]
                [--validset_dir VALIDSET_DIR] [--upscale_factor {2,4,8}]
                [--epochs EPOCHS] [--resume RESUME]
                [--mode {adversarial,generator}] [--pretrain PRETRAIN]
                [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --trainset_dir TRAINSET_DIR
                        training dataset path
  --validset_dir VALIDSET_DIR
                        validation dataset path
  --upscale_factor {2,4,8}
                        super resolution upscale factor
  --epochs EPOCHS       training epoch number
  --resume RESUME       continues from epoch number
  --mode {adversarial,generator}
                        apply adversarial training
  --pretrain PRETRAIN   load pretrained generator model
  --cuda                Using GPU to train
```

### Testing

Download the [SRGAN weight](https://drive.google.com/file/d/1dsa67sCyM29_Tor124KjP2vVYxO8sXD3/view?usp=sharing) that was trained with DIV2K dataset.

Run the script test_image.py

```
$ python test_image.py --image $IMG --weight $WEIGHT --cuda

usage: test_image.py [-h] [--image IMAGE] [--upscale_factor {2,4,8}]
                     [--weight WEIGHT] [--downsample {None,bicubic}] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         input image
  --upscale_factor {2,4,8}
                        super resolution upscale factor
  --weight WEIGHT       generator weight file
  --downsample {None,bicubic}
                        Downsample the input image before applying SR
  --cuda                Using GPU to run
```

### Crop image

To visualize and compare the detail in the image, this script to save multiple patches from input image with colored bounding box. The cropped images will be saved in the same directory as input image. When the saved coordinates is not specified, the program will prompt image for used to select bounding box from image. Then the coordinates will be saved to crop other images.


```
$ python get_img_crop.py --image $IMG

usage: get_img_crop.py [-h] [--image IMAGE] [--coords COORDS]

optional arguments:
  -h, --help       show this help message and exit
  --image IMAGE    input image to be croped
  --coords COORDS  Loading the bounding box coordinates from saved file.
                   Manual selecting boxes when no saved coordinates is used.
```

Sample Results
----------------------------
### Sample from DIV2K validation set  

#### Bicubic
![sample1_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample1_lr.png "Bicubic")   

#### SRGAN
![sample1_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample1_sr.png "SRGAN")   


### Sample from the xView dataset   

#### Bicubic
![sample2_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample2_lr.png "Bicubic")   

#### SRGAN
![sample2_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample2_sr.png "SRGAN")    

#### Bicubic
![sample3_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample3_lr.png "Bicubic")   

#### SRGAN
![sample3_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample3_sr.png "SRGAN")   


### Test image   

#### Low Res
![nya_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/nya_lr.png "Low Resolution")   

#### SRGAN
![nya_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/nya_sr.png "SRGAN")   






