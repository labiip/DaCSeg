# DaCSeg

## This project is referenced from the source code in the official torchvision module of pytorch (with slight differences in the use of pycocotools).
* https://github.com/pytorch/vision/tree/master/references/detection

## Environment Configuration：
* Python3.6/3.7/3.8
* Pytorch1.10or more
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(No additional vs installation required))
* Ubuntu or Centos(Not recommended for Windows)
* Best trained using GPU
* For detailed environment configuration, see`requirements.txt`

## File structure：
```
  ├── backbone: feature extraction network
  ├── network_files: Mask R-CNN, DaCSeg
  ├── train_utils: Training validation related modules (including coco validation related)
  ├── my_dataset_chromosome.py: Define dataset for reading chromosome dataset
  ├── train.py: Single GPU/CPU Training Scripts
  ├── train_multi_GPU.py: For users with multiple GPUs
  ├── predict.py: Simple prediction script using trained weights
  ├── validation.py: Validate/test the COCO metrics of the data using the trained weights and generate the record_mAP.txt file
  └── transforms.py: Data preprocessing (random horizontal flip images as well as bboxes, conversion of PIL images to Tensor)
```

## Pre-training weights download address (download and place in current folder):
* Resnet50 pre-training weight https://download.pytorch.org/models/resnet50-0676ba61.pth (Note that you have to rename the pre-training weights after downloading them.
For example, the `resnet50.pth` file is read in train.py, not `resnet50-0676ba61.pth`)
* Mask R-CNN(Resnet50+FPN) pre-training weight https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth (Attention.
Rename the pre-training weights after loading them, e.g. in train.py the `maskrcnn_resnet50_fpn_coco.pth` file is read, not `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)
 
 

## Training methods
* Ensure that the dataset is prepared in advance
* Ensure that the corresponding pre-trained model weights are downloaded in advance
* Make sure to set up `--num-classes` and `--data-path`.
* To train with a single GPU use the train.py training script directly
* To train with multiple GPUs, use the `torchrun --nproc_per_node=8 train_multi_GPU.py` command, with the `nproc_per_node` parameter being the number of GPUs to use
* If you want to specify which GPU devices to use you can prefix the command with `CUDA_VISIBLE_DEVICES=0,3` (e.g. I only want to use GPU devices 1 and 4 of the devices).
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## caveat
1. When using the training script, be careful to set `--data-path` to the **root directory** where you store your own dataset:
2. If `batch_size` is multiplied, it is recommended that the learning rate be multiplied as well. Suppose `batch_size` is set from 4 to 8, then the learning rate `lr` is set from 0.004 to 0.008.
3. If using the Batch Normalization module, the `batch_size` must not be smaller than 4, otherwise the result will be worse. **If there is not enough video memory and batch_size must be less than 4**, it is recommended that when creating `resnet50_fpn_backbone`, the Set `norm_layers` to `FrozenBatchNorm2d` or set `trainable_layers` to 0 (i.e. freeze the whole `backbone`)
4. The `det_results.txt` (target detection task) and `seg_results.txt` (instance segmentation task) saved during training are the COCO metrics for each epoch on the validation set, with the first 12 values being the COCO metrics, and the last two values being the average training loss as well as the learning rate
5. When using the prediction script, set `weights_path` to your own generated weights path.
6. When using the validation file, take care to ensure that your validation or test set must contain targets for each class, and use it with modifications to `--num-classes`, `--data-path`, `--weights-path`, and
`--label-json-path` (this parameter is set based on the training dataset). Other code should be left as unchanged as possible

