import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

import transforms
from network_files import MaskRCNN
from network_files import OverlapMaskRCNN
from backbone import resnet50_fpn_backbone
from backbone import resnet101_fpn_backbone
# from my_dataset_coco import CocoDetection
# from my_dataset_voc import VOCInstances
from my_dataset_overlap_else import Chromdataset_else
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups


def create_model(num_classes, load_pretrain_weights=True):
    # If the GPU memory is very small, batch_size can't be set very large, it is recommended to set the norm_layer to FrozenBatchNorm2d (default is nn.BatchNorm2d)
    # FrozenBatchNorm2d is similar to BatchNorm2d, but the parameters cannot be updated.
    # trainable_layers include ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']，
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet101_fpn_backbone(pretrain_path="resnet101.pth", trainable_layers=3)
    # backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)

    model = MaskRCNN(backbone, num_classes=num_classes)
    # model = OverlapMaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # The file used to save coco_info
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        # "train": transforms.Compose([transforms.ToTensor(),
        #                              transforms.RandomHorizontalFlip(0.5)],
        #                              transforms.Resize((1333, 800), interpolation=3),
        #                              transforms.Pad(10),
        #                              transforms.RandomCrop((1333, 800)),
        #                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path

    # load train data set
    # data -> annotations -> instances_train2017.json
    train_dataset = Chromdataset_else(data_root, "train", data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # Whether to compose a batch by sampling images with similar aspect ratios.
    # This reduces the amount of GPU memory required for training, and is used by default.
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # Statistical index of the position of all image aspect ratios in the bins interval
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # Each batch image is taken from the same aspect ratio interval
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # Note that the collate_fn here is customized because the data read includes images and targets, which can't be synthesized directly using the default method of batch synthesis
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # If the image is sampled according to the image aspect ratio, the dataloader needs to use the batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    # for item in train_dataset:
    #     print(item)
    # load validation data set
    # data -> annotations -> instances_val2017.json
    val_dataset = Chromdataset_else(data_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    # model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=False)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    # If the resume parameter is passed in, i.e. the address of the weights from the last training, then the training continues with the last parameter
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # Read the previously saved weights file (including the optimizer as well as the learning rate strategy)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt
        with open(det_results_file, "a") as f:
            # The data written includes coco metrics as well as loss and learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # The data written includes coco metrics as well as loss and learning rate
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "/home/xinyu.fan/results/model_{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # Type of Training Equipment
    parser.add_argument('--device', default='cuda:0', help='device')
    # The root directory of the training dataset
    parser.add_argument('--data-path', default='./data/chrom', help='dataset')
    # Number of detection target categories (without background)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # File save address
    parser.add_argument('--output-dir', default="/home/xinyu.fan/", help='path where to save')
    # If you need to follow the last training, specify the address of the last training weights file.
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # Specify the number of epochs to start training from.
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # Total number of epochs trained
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    # learning rate
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # parser.add_argument('--lr', default=0.00001, type=float,
    #                     help='initial learning rate, 0.02 is the default value for training '
    #                          'on 8 gpus and 2 images_per_gpu')
    # momentum parameters of SGD
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # weight_decay parameters of SGD
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-steps', default=[27, 34], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # parser.add_argument('--lr-steps', default=[82, 94], nargs='+', type=int,
    #                     help='decrease lr every step-size epochs')
    # Parameters for torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    # Batch size for training (larger settings are recommended if memory/GPU graphics are plentiful)
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # Whether to train with mixed precision (requires GPU support for mixed precision)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # Check if the save weights folder exists, create it if it doesn't
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
