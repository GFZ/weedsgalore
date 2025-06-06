# SPDX-FileCopyrightText: 2025 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Ekin Celikkan <ekin.celikkan@gfz.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import deeplabv3plus_resnet50, deeplabv3plus_resnet50_do
from pathlib import Path
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_integer('dataset_size_train', 104, 'dataset size of train set')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')
flags.DEFINE_boolean('dlv3p_do', False, 'set True to use probabilistic variant of DLv3+ with dropout')
flags.DEFINE_boolean('pretrained_backbone', True, 'set True to use pretrained ResNet50 backbone')
flags.DEFINE_string('ckpt_resnet', 'ckpts/resnet50-19c8e357.pth', 'ckpt path for pretrained backbone')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('num_workers', 4, 'number of subprocesses')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_string('out_dir', 'out_dir', 'directory to save logs and ckpts')
flags.DEFINE_integer('log_interval', 25, 'number of iterations to log scalars')
flags.DEFINE_integer('ckpt_interval', 500, 'number of iterations to save ckpts')

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"Cuda current device: {torch.cuda.current_device()}")
        print(f"Cuda device name: {torch.cuda.get_device_name(0)}")


    # Dataset
    train_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=FLAGS.dataset_size_train, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=True, split='train', augmentation=True)
    val_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=False, split='val', augmentation=False)

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                  num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=FLAGS.batch_size, shuffle=False,
                                num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True)

    # Network
    if FLAGS.dlv3p_do:
        net = deeplabv3plus_resnet50_do(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)  # probabilistic DeepLabv3+
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)  # (determinsitic) DeepLabv3+

    # Modify first layer
    if FLAGS.in_channels == 5:
        net.backbone.conv1 = torch.nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device)

    # Model to device
    net.to(device=device)

    # Loss criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=FLAGS.ignore_index).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    # Metric
    evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)

    # Logging
    accum_loss, accum_iter, tot_iter = 0, 0, 0
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    writer = SummaryWriter(f'{FLAGS.out_dir}')
    print(f'Logging to: {FLAGS.out_dir}')
    Path(FLAGS.out_dir).mkdir(parents=True, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)

    # Train
    for epoch in range(FLAGS.epochs):
        net.train()
        train_iter = iter(train_dataloader)
        for i, data in enumerate(train_iter):
            features, unique_labels, binary_labels = data
            if FLAGS.num_classes == 3:
                labels = binary_labels
            else:
                labels = unique_labels
            features, labels = features.to(device), labels.to(device)  # NCHW

            optimizer.zero_grad()
            out = net(features)
            loss = criterion(out, labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss
            accum_iter += 1
            tot_iter += 1

            # compute miou
            _, pred = torch.max(out, 1)
            evaluator.update(pred, labels)

            # log scalars
            if tot_iter % FLAGS.log_interval == 0 or tot_iter == 1:
                metrics = evaluator.compute() * 100

                print(f'Epoch: {epoch} iter: {tot_iter}, Loss: {(accum_loss / accum_iter):.2f}')
                print(f'mIoU: {metrics.mean():.2f}%')

                writer.add_scalar('Training Loss', accum_loss / accum_iter, tot_iter)
                writer.add_scalar('miou (%)', metrics.mean(), tot_iter)
                writer.add_scalar('iou_crop (%)', metrics[1], tot_iter)
                for weed_idx, weed_iou in enumerate(metrics[2:], start=2):
                    writer.add_scalar(f'iou_weed_{weed_idx-1} (%)', weed_iou, tot_iter)

                evaluator.reset()
                accum_loss, accum_iter = 0, 0

            # save ckpt
            if tot_iter % FLAGS.ckpt_interval == 0 or tot_iter == 1:
                torch.save(net.state_dict(), f'{FLAGS.out_dir}/{str(epoch)}.pth')
                torch.save(optimizer.state_dict(), f'{FLAGS.out_dir}/optimizer.pth')

if __name__ == '__main__':
    app.run(main)
