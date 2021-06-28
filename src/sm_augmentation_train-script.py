import argparse
import json
import logging
import os
import sys
import copy
import time
import io
from PIL import Image

import boto3
import torch
import torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


Image.MAX_IMAGE_PIXELS = None


"""
Method to augment and load data on CPU with PyTorch Dataloader
"""

def augmentation_pytorch_dataloader(train_dir, batch_size, workers, is_distributed, use_cuda, aug_load):
    
    
    print ("Image augmentation using PyTorch Dataloaders on CPUs")
    
    #https://pytorch.org/vision/stable/transforms.html
    aug_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #TODO: Uncomment once DALI operation errors are fixed
            #transforms.RandomRotation(5),
            #transforms.Pad(4),
            #transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
            #transforms.RandomAutocontrast(),
            #transforms.functional.affine()
    ]
    
    crop_norm_ops = [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]) 
    ]
    
    train_aug_ops = []
    # Repeating Augmentation to influence bottleneck
    for iteration in range (aug_load):
        train_aug_ops = train_aug_ops + aug_ops

    data_transforms = {
        'train': transforms.Compose(train_aug_ops + crop_norm_ops),
        #'val': transforms.Compose(val_aug_ops),
        'val': transforms.Compose(crop_norm_ops),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(train_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    train_sampler = torch.utils.data.distributed.DistributedSampler(image_datasets) if is_distributed else None
    dataloaders = {x: torch.utils.data.DataLoader(dataset = image_datasets[x], 
                                                  batch_size = batch_size,
                                                  shuffle = train_sampler, 
                                                  num_workers = workers, 
                                                  pin_memory = True if use_cuda else False)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


"""
Method to augment and load data on CPU or GPU with NVIDIA DALI
"""

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu, is_training, aug_load):
    
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    
    images = fn.decoders.image(images,
                               device=decoder_device,
                               output_type=types.RGB,
                               device_memory_padding=device_memory_padding,
                               host_memory_padding=host_memory_padding)
    
    if is_training:
        
        # Repeating Augmentation to influence bottleneck
        for x in range (aug_load):
            
            
            #https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html
            images = fn.flip(images, device=dali_device, horizontal=1, vertical=1)
            
            #TODO: Fix operation errors
            #images = fn.rotate(images, angle=5, device=dali_device)
            #images = fn.pad(images, device=dali_device)
            #images = fn.gaussian_blur(images, device=dali_device)
            #images = fn.contrast(images, device=dali_device)
            #images = fn.warp_affine(images, device=dali_device)
        
    images = fn.random_resized_crop(images, 
                                    size = size, 
                                    device=dali_device,
                                    random_aspect_ratio=[0.8, 1.25],
                                    random_area=[0.1, 1.0],
                                    num_attempts=100,
                                    interp_type=types.INTERP_TRIANGULAR)

    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255])
    images = images.gpu()
    labels = labels.gpu()
    
    return images, labels

def augmentation_dali(train_dir, batch_size, workers, is_distributed, use_cuda, host_rank, world_size, aug_load, dali_cpu):
    
    if dali_cpu:
        print ("Image augmentation using DALI pipelines on CPUs")
    else:
        print ("Image augmentation using DALI pipelines on GPUs")
        
    dataloaders = {}
    dataset_sizes = {}
    
    train_path = train_dir+'/train/'
    dataset_sizes['train'] = sum([len(files) for r, d, files in os.walk(train_path)])
    train_pipe = create_dali_pipeline(batch_size=batch_size,
                                            num_threads=workers,
                                            device_id=host_rank,
                                            seed=12 + host_rank,
                                            data_dir=train_path,
                                            crop=224,
                                            size=256,
                                            dali_cpu=dali_cpu,
                                            shard_id=host_rank,
                                            num_shards=world_size,
                                            is_training=True,
                                            aug_load=aug_load)
    train_pipe.build()
    dataloaders['train'] = DALIClassificationIterator(train_pipe, 
                                                  reader_name="Reader", 
                                                  last_batch_policy=LastBatchPolicy.PARTIAL)

    # validation data
    val_path = train_dir+'/val/'
    dataset_sizes['val'] = sum([len(files) for r, d, files in os.walk(val_path)])
    val_pipe = create_dali_pipeline(batch_size=batch_size,
                                            num_threads=workers,
                                            device_id=host_rank,
                                            seed=12 + host_rank,
                                            data_dir=val_path,
                                            crop=224,
                                            size=256,
                                            dali_cpu=dali_cpu,
                                            shard_id=host_rank,
                                            num_shards=world_size,
                                            is_training=False,
                                            aug_load=aug_load)
    val_pipe.build()
    dataloaders['val'] = DALIClassificationIterator(val_pipe, 
                                                  reader_name="Reader", 
                                                  last_batch_policy=LastBatchPolicy.PARTIAL)
        
    return dataloaders, dataset_sizes
    

"""
Method to train models for the given number of epochs
"""

def run_training_epochs(model_ft, num_epochs, criterion, optimizer_ft,
                        dataloaders, dataset_sizes, device, USE_PYTORCH):
    
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Running Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        
        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()
            
            running_loss = 0.0
            running_corrects = 0
                    
            # Data iteration if using DALI Pipelines for loading the augmented data
            if not USE_PYTORCH:
    
                for i, data in enumerate(dataloaders[phase]):
                    inputs = data[0]["data"]
                    labels = data[0]["label"].squeeze(-1).long()
                    
                    #inputs = inputs.to(device) #todo commented earlier
                    #labels = labels.to(device) #todo commented earlier

                    optimizer_ft.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
            # Data iteration if using PyTorch Dataloader for loading the augmented data
            else:
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer_ft.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            # Calculate Train/Val Loss and Accuracy for every Epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{}-loss: {:.4f} {}-acc: {:.4f}'.format(
                phase, epoch_loss, phase, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())
         
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch completed in {:.0f}m {:.0f}s'.format(
            epoch_time_elapsed // 60, epoch_time_elapsed % 60))
        print()
        
        # uncomment this if using G4 instances with DALI-GPU
        
        #if not USE_PYTORCH:
        #    for phase in ['train', 'val']:
        #        dataloaders[phase].reset()
    
    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    return model_ft, best_acc


"""
Method for handling model training
"""
    
def training(args):
    
    num_gpus = args.num_gpus
    hosts = args.hosts
    current_host = args.current_host
    backend = args.backend
    seed = args.seed
    
    is_distributed = len(hosts) > 1 and backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = num_gpus > 0
    logger.debug("Number of gpus available - {}".format(num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")
    
    world_size = len(hosts)
    os.environ['WORLD_SIZE'] = str(world_size)
    host_rank = hosts.index(current_host)
        
    if is_distributed:
        # Initialize the distributed environment.
        dist.init_process_group(backend=backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), num_gpus))
    # set the seed for generating random numbers
    torch.manual_seed(seed)
    
    if use_cuda:
        torch.cuda.manual_seed(seed)
        
    # Loading training and validation data
    batch_size = args.batch_size
    train_dir = args.train_dir 
    
    workers = os.cpu_count() if use_cuda else 0
                               
    aug_load = args.aug_load
    
    # Image augmentation using DALI pipelines on GPUs
    USE_PYTORCH = False
    USE_DALI_CPU = False
    
    # Image augmentation using PyTorch Dataloaders on CPUs
    if args.aug == 'pytorch-cpu':
        USE_PYTORCH = True
        
    # Image augmentation using DALI pipelines on CPUs
    if args.aug == 'dali-cpu':
        USE_DALI_CPU = True
    
    
    if USE_PYTORCH == True:
        dataloaders, dataset_sizes = augmentation_pytorch_dataloader(train_dir, 
                                                                     batch_size, 
                                                                     workers, 
                                                                     is_distributed, 
                                                                     use_cuda,
                                                                     aug_load)
    else:
        dataloaders, dataset_sizes = augmentation_dali(train_dir, 
                                                   batch_size,
                                                   workers, 
                                                   is_distributed, 
                                                   use_cuda, 
                                                   host_rank, 
                                                   world_size,  
                                                   aug_load,
                                                   dali_cpu=USE_DALI_CPU)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.pretrained_model_type == 'RESENT18':
        model_ft = models.resnet18(pretrained=False)
    else:
        model_ft = models.resnet50(pretrained=False)
    model_ft = model_ft.to(device)
    if is_distributed and use_cuda:
        model_ft = torch.nn.parallel.DistributedDataParallel(model_ft)
    else:
        model_ft = torch.nn.DataParallel(model_ft)
        
    num_epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), args.lr, args.momentum)
    
    # Running Model Training   
    since = time.time()
    model_ft, best_acc = run_training_epochs(model_ft, 
                                             num_epochs, 
                                             criterion, 
                                             optimizer_ft,
                                             dataloaders, 
                                             dataset_sizes, 
                                             device,
                                             USE_PYTORCH)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Saving model
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model_ft.cpu().state_dict(), path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--pretrained-model-type', type=str, default=None,
                        help='Pre-trained model architecture to start training from')
    
    parser.add_argument('--aug', type=str, default=None, help='Hardcoded to expect either: pytorch-cpu, dali-cpu, or dali-gpu')
    parser.add_argument('--aug-load', type=int, default=1, help='Number of times Augmentation should be repeated to create bottleneck')
    
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val_dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    
    training(parser.parse_args())