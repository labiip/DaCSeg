# DaCSeg

##  This project is referenced from the source code in the official torchvision module of pytorch (with slight differences in the use of pycocotools).
* https://github.com/pytorch/vision/tree/master/references/detection

## Environment Configuration:
* Python3.6/3.7/3.8
* Pytorch1.10
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`
* Ubuntu or Centos(Not recommended for Windows)
* Best trained using GPU
* `requirements.txt` for detailed environment configuration.


```

## Training methods
* Ensure that the dataset is prepared in advance
* Ensure that the corresponding pre-trained model weights are downloaded in advance
* Make sure to set up `--num-classes` and `--data-path`.
* To train with a single GPU use the train.py training script directly
* To train with multiple GPUs, use the `torchrun --nproc_per_node=8 train_multi_GPU.py` command, with the `nproc_per_node` parameter being the number of GPUs to use
* If you want to specify which GPU devices to use you can prefix the command with `CUDA_VISIBLE_DEVICES=0,3` (e.g. I only want to use GPU devices 1 and 4 of the devices).
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

