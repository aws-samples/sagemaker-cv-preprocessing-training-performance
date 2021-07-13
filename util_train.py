import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def aug_exp_train(model_arch, batch_size, aug_operator, aug_load_factor, instance_type, sm_role):

    # Amazon S3 bucket and prefix for fetching and storing data:
    BUCKET = 'sm-aug-test'

    # Full size download of https://github.com/fastai/imagenette
    # 1.3GB — 13,395 (9469 train, 3925 val images) from 10 classes
    train_data_s3 = 's3://{}/{}'.format(BUCKET, 'imagenette2')
    src_code_s3 = 's3://{}/{}'.format(BUCKET, 'training_jobs')
    training_job_output_s3 = 's3://{}/{}'.format(BUCKET, 'training_jobs_output')

    # Encapsulate training on SageMaker with PyTorch:
    train_estimator = PyTorch(entry_point='sm_augmentation_train-script.py',
                              source_dir='./src',
                              role=sm_role,
                              framework_version='1.8.1',
                              py_version='py3',
                              debugger_hook_config=False,

                              instance_count=1,
                              instance_type=instance_type,

                              output_path=training_job_output_s3,
                              code_location=src_code_s3,

                              hyperparameters={'epochs': 2,
                                               'backend': 'nccl',
                                               'model-type': model_arch,
                                               'lr': 0.001,
                                               'batch-size': batch_size,
                                               'aug': aug_operator,
                                               'aug-load-factor': aug_load_factor
                                               }
                              )

    # Setting up File-system to import data from S3
    data_channels = {'train': sagemaker.inputs.TrainingInput(
        s3_data_type='S3Prefix',
        s3_data=train_data_s3,
        content_type='image/jpeg',
        input_mode='File'),
        'val': sagemaker.inputs.TrainingInput(
            s3_data_type='S3Prefix',
            s3_data=train_data_s3,
            content_type='image/jpeg',
            input_mode='File')
    }

    # Launching SageMaker training job
    train_job_id = 'sm-aug-' + str(datetime.now().strftime("%H-%M-%S"))
    train_estimator.fit(inputs=data_channels, job_name=train_job_id)

    return train_job_id, train_estimator
