import nn.classifier
import nn.unet as unet
import helpers

import argparse


import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.fetcher import DatasetFetcher
import nn.classifier
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback

import os
from multiprocessing import cpu_count

from data.dataset import TrainImageDataset, TestImageDataset
import img.transformer as transformer

# copy from example
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--mode', default="TRAIN", help='TRAIN or TEST')
parser.add_argument('--model', default="", help='Load model path')

opt = parser.parse_args()
print(opt)



def main():
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    img_resize = (1024, 1024)
    batch_size = 2   # 2
    epochs = 50
    
    if opt.mode == 'TEST':
        batch_size = 1
        epochs = 1

    threshold = 0.5
    validation_size = 0.2
    sample_size = None  # Put None to work on full dataset

    # Training on 4576 samples and validating on 512 samples
    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Download the datasets
    ds_fetcher = DatasetFetcher(opt)
    ds_fetcher.download_dataset(False)

    # Get the path to the files for the neural net
    # We don't want to split train/valid for KFold crossval
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size,
                                                                    validation_size=validation_size)
    full_x_test = ds_fetcher.get_test_files(sample_size)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)  # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output/submit.csv.gz'),
                                             origin_img_size, threshold)

    # Define our neural net architecture
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.CarvanaClassifier(net, epochs, opt)
    
    if opt.model != "":
        classifier.restore_model(opt.model)
        classifier.net.eval()

    train_ds = TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.augment_img)
    train_loader = DataLoader(train_ds, batch_size,
                            sampler=RandomSampler(train_ds),
                            num_workers=threads,
                            pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                            sampler=SequentialSampler(valid_ds),
                            num_workers=threads,
                            pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples "
        .format(len(train_loader.dataset), len(valid_loader.dataset)))

    if opt.mode == 'TRAIN':
        classifier.train(train_loader, valid_loader, epochs,
                        callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])
    else:
        classifier.train(train_loader, valid_loader, epochs,
                        callbacks=[])

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])
    pred_saver_cb.close_saver()


if __name__ == "__main__":
    main()
