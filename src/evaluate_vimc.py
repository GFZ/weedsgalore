from absl import app, flags
import torch
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Ekin Celikkan <ekin.celikkan@gfz-potsdam.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassCalibrationError
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from nets import deeplabv3plus_resnet50_do

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'weedsgalore', 'options: weedsgalore')
flags.DEFINE_string('dataset_path', '/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('split', 'test', 'Options: val, test')

flags.DEFINE_string('network', 'deeplabv3plus', 'options: deeplabv3plus')
flags.DEFINE_string('ckpt', '/dlv3p_do_rgb_3.pth', 'checkpoint directory')

flags.DEFINE_integer('in_channels', 3, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 3, 'options: 3 (uni-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')

flags.DEFINE_integer('mc_samples', 5, 'number of forward passes for probabilistic inference')


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"Cuda current device: {torch.cuda.current_device()}")
        print(f"Cuda device name: {torch.cuda.get_device_name(0)}")

    net = deeplabv3plus_resnet50_do(num_classes=FLAGS.num_classes)
    net = net.to(device)

    # first conv to fit input channels
    net.backbone.conv1 = nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2,
                                   padding=3, bias=False, device=device)

    # load checkpoint
    model_weights_dir = FLAGS.ckpt
    model_dict = torch.load(model_weights_dir, map_location=device)
    net.load_state_dict(model_dict)

    # Dataset and dataloader
    dataset_path = FLAGS.dataset_path
    dataset = WeedsGaloreDataset(dataset_path=dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                 num_classes=FLAGS.num_classes, is_training=False, split=FLAGS.split,
                                 augmentation=False)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=None,
                            drop_last=True)
    dataset_iter = iter(dataloader)

    # iou evaluator
    evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(
        device)

    # mcce evaluator
    mcce = MulticlassCalibrationError(num_classes=FLAGS.num_classes, n_bins=FLAGS.num_classes, norm='l1',
                                      ignore_index=FLAGS.ignore_index).to(device)

    net.eval()
    enable_dropout(net)

    # Evaluation on dataset
    for i, data in enumerate(dataset_iter):
        features, unique_labels, binary_labels = data
        features, unique_labels, binary_labels = features.to(device), unique_labels.to(device), binary_labels.to(device)

        # init tensor to collect all forward pass predictions
        predictions = torch.empty((0, FLAGS.num_classes, features.shape[2], features.shape[3]), device=device)

        if FLAGS.num_classes == 3:
            labels = binary_labels
        else:
            labels = unique_labels

        # Loop 2: same scene, multiple passes
        for j in range(FLAGS.mc_samples):
            print(f'Testing forward pass {j + 1}/{FLAGS.mc_samples}')

            with torch.no_grad():
                out = net(features)

            out = torch.nn.functional.softmax(out, dim=1)  # [1 x C x H x W]

            # stack predictions from all forward passes
            predictions = torch.vstack((predictions, out))  # [K x C x H x W]

        # mean prediction
        _, pred = torch.max(torch.mean(predictions, dim=0), dim=0, keepdim=True)


        # update metrics
        evaluator.update(pred, labels)
        scene_scores = evaluator(pred, labels)
        mcce.update(torch.mean(predictions, dim=0, keepdim=True), labels)

        # calculate entropy (for each scene)
        epsilon = +1e-8
        entropy = -torch.sum(torch.mean(predictions, dim=0) * torch.log(torch.mean(predictions, dim=0) + epsilon), axis=0)  # [H, W]


    # print scores over all dataset
    print(f"\n{'=' * 40}")
    print(f"{'Overall scores':^40}")
    print(f"{'=' * 40}")
    scores = evaluator.compute()
    print(f'Split: {FLAGS.split}')
    print(f'mIoU: {scores.mean() * 100:.2f}%')
    print(f'iou bg: {scores[0] * 100:.2f}%')
    print(f'iou crop: {scores[1] * 100:.2f}%')
    for weed_idx, weed_iou in enumerate(scores[2:], start=2):
        print(f'iou weed_{weed_idx-1}: {weed_iou * 100:.2f}%')
    print(f"ECE: {mcce.compute()}")


if __name__ == '__main__':
    app.run(main)
