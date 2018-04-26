# tf_fcn

Implented FCN8,FCN16,FCN32,Using CRF layer
[CVPR 2015] Long et al. [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
infering [tf_fcn](https://github.com/Yuliang-Zou/tf_fcn)


## Requirements
* tensorflow 1.2
* python3.5

## Prepare dataset

In this implementation, we use the [VOC2012 dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).Download dataset and put in `./dataset/` folder


## Pre-trained model

```bash
mkdir models
```
We use a ImageNet pre-trained model to initialize the network, please download the npy file [here](https://drive.google.com/file/d/0B2SnTpv8L4iLRTFZb0FWenRJTlU/view?usp=sharing) and put it under the `./models` folder.


## How to train

Since input images have different sizes, in order to make them as minibatch input, we used two different strategies: 1) padding to a large size; or 2) resize to a small size (256, 256)

```bash
cd src
python train.py          # padding
python train_small.py    # resize
```


## Demo

```bash
python demo.py
python demo_crf.py #add crf layer
```

You can change the `config` dictionary to use custom settings.


## Models

Padding to (640, 640):

- FCN32_adam_20000: [ckpt](https://drive.google.com/file/d/0B3vJudZqxciYbTRuY21WZXREV0E/view?usp=sharing), [npy](https://drive.google.com/file/d/0B2SnTpv8L4iLNEVFd2RHcUZOX00/view?usp=sharing)

- FCN16_adam_5000:  [ckpt](https://drive.google.com/file/d/0B2SnTpv8L4iLT2VuREZwUHg4cjg/view?usp=sharing)

- FCN8_adam_10000:  [ckpt](https://drive.google.com/file/d/0B2SnTpv8L4iLRExqQTVONWxTX0U/view?usp=sharing)

Padding to (500, 500):

- FCN32_adam_35000: [ckpt](https://drive.google.com/file/d/0B3vJudZqxciYVWZfbXdybzFhWDA/view?usp=sharing) (You can extract npy with `extract` method defined in `Model.py`)

- FCN8_adam_30000: [ckpt](https://drive.google.com/file/d/0B3vJudZqxciYVWZfbXdybzFhWDA/view?usp=sharing)


**Note:** When you train the shortcut version model (FCN16 and FCN8), you will need FCN32 model npy file as initialization, instead of the ImageNet pre-trained model npy file.
