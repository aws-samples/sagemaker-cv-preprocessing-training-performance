# sagemaker-cv-preprocessing-training-performance

This repository contains [Amazon SageMaker](https://aws.amazon.com/sagemaker/) training implementation with data pre-processing (decoding + augmentations) on both GPUs and CPUs for computer vision — allowing you to compare and reduce training time by addressing CPU bottlenecks caused by increasing data pre-processing load. This is achieved by GPU-accelerated JPEG image decoding and offloading of augmentation to GPUs using [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/). Performance bottlenecks and ystem utilizations metrics are compared using [Amazon Sagemaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html).

## Module Description:

- `util_train.py`: Launch [Amazon Sagemaker PyTorch traininng](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html) jobs with your custom training script.
- `src/sm_augmentation_train-script.py`: Custom training script to train models of different complexities (`RESNET-18`, `RESNET-50`, `RESNET-152`) with data pre-processing implementation for: 
  - JPEG decoding and augmentation on CPUs using PyTorch Dataloader
  - JPEG decoding and augmentation on CPUs & GPUs using NVIDIA DALI 
- `util_debugger.py`: Extract system utilization metrics with [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html).

## Run SageMaker training job with decoding and augmentation on GPU:
- Parameters such as training data path, S3 bucket, epochs and other training hyperparameters can be adapted at `util_train.py`. 
- The custom custom training script used is  `src/sm_augmentation_train-script.py`.
```
from util_debugger import get_sys_metric
from util_train import aug_exp_train
aug_exp_train(model_arch = 'RESNET50', 
              batch_size = '32', 
              aug_operator = 'dali-gpu', 
              instance_type='ml.p3.2xlarge',  
              curr_sm_role = 'to-be-added')
```

## Experiment to compare bottlenecks:

- Create an Amazon S3 bucket called `sm-aug-test` and upload the [Imagenette dataset](https://github.com/fastai/imagenette) ([download link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)).
- Update your SageMaker execution role in the notebook `sm_augmentation_train-script.py` and run the notebook to compare seconds/epoch and system utilization for training jobs by toggling the following parameters:
  - `instance_type` (default: `ml.p3.2xlarge`)
  - `model_arch` (default: `RESNET18`)
  -  `batch_size` (default: `32`)
  -  `aug_load_factor` (default: `12`)
  -  `AUGMENTATION_APPROACHES` (default: `['pytorch-cpu', 'dali-gpu']`)
- Comparison results using the above default parameter setup:
  - Seconds/ Epoch improvement of `72.59%` in Amazon SageMaker training job by offloading JPEG decoding and heavy augmentation to GPU — addressing data pre-processing bottleneck to improve performance-cost ratio.
  - Using the above strategy, training time improvement is higher for lighter models like `RESNET-18` (which causes more CPU bottlenecks) over heavier model such as `RESNET-152` as the `aug_load_factor` is increased while keeping lower batch size of `32`.
  - System utilization Histograms and CPU bottleneck Heatmaps are generated with SageMaker Debugger in the notebook. Profiler Report and other interactive visuals available on SageMaker Studio.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
