# amazon-sagemaker-training-augmentations

This repository contains implementation of data pre-processing on both GPUs and CPUs for computer vision — allowing you to reduce Amazon SageMaker training time (thus improving performance-cost ratio) by addressing CPU bottlenecks caused by increasing data pre-processing load. This is achieved by moving JPEG image decoding and augmentation load to GPU by using [NVIDIA DALI](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). It also allows you to visualize system utilizations using Amazon Sagemaker Debugger for identifying bottlenecks.

## Module Description:

- `util_train.py`: Launch [Amazon Sagemaker PyTorch traininng](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html) jobs with your custom training script.
- `sm_augmentation_train-script.py`: Custom training script to train models of different complexities (`RESNET-18`, `RESNET-50`, `RESNET-152`) with data pre-processing implementation for: 
  - JPEG Decoding and Augmentation on CPUs using [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - JPEG Decoding and Augmentation on GPUs or CPUs using [NVIDIA DALI](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- `util_debugger.py`: Extract system utilization metrics with [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html).

## Results with default values:
- `sm_augmentation_train-script.py`: Notebook to compare seconds/epoch and system utilization for training jobs by toggling the following parameters:
  - `instance_type` (default: `ml.p3.2xlarge`)
  -  `batch_size` (default: `32`)
  -  `aug_load_factor` (default: `12`)
  -  `AUGMENTATION_APPROACHES` (default: `['pytorch-cpu', 'dali-gpu']`)
   
  
- Seconds/ Epoch improvement (72.59%) in Amazon SageMaker training job by offloading JPEG decoding and heavy augmentation to GPU — addressing data pre-processing bottleneck.
- Using the above startegy, training time improvement gets higher for lighter models like `RESNET-18` (which causes more CPU bottlenecks) over heavier model such as `RESNET-152` as the `aug_load_factor` is increased from `1` to `12`, while keeping lower batch size of `32`.
- System utilization metrics and visuals of CPU bottleneck improvement with SageMaker Debugger.


## License
This library is licensed under the MIT-0 License. See the LICENSE file.