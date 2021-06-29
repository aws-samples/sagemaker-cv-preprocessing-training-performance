import time
import boto3
from datetime import datetime
from smdebug.profiler.system_metrics_reader import S3SystemMetricsReader
from smdebug.profiler.algorithm_metrics_reader import S3AlgorithmMetricsReader
from smdebug.profiler.analysis.notebook_utils.metrics_histogram import MetricsHistogram
from smdebug.profiler.analysis.notebook_utils.heatmap import Heatmap

def get_sys_metric(train_estimator):
    
    path = train_estimator.latest_job_profiler_artifacts_path()
    system_metrics_reader = S3SystemMetricsReader(path)

    sagemaker_client = boto3.client("sagemaker")
    training_job_name = train_estimator.latest_training_job.name

    training_job_status = ""
    training_job_secondary_status = ""
    while system_metrics_reader.get_timestamp_of_latest_available_file() == 0:
        system_metrics_reader.refresh_event_file_list()
        client = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        if "TrainingJobStatus" in client:
            training_job_status = f"TrainingJobStatus: {client['TrainingJobStatus']}"
        if "SecondaryStatus" in client:
            training_job_secondary_status = f"TrainingJobSecondaryStatus: {client['SecondaryStatus']}"

        print(
            f"Profiler data from system not available yet. {training_job_status}. {training_job_secondary_status}."
        )
        time.sleep(20)

    print("\n\nProfiler data from system is available")
    
    framework_metrics_reader = S3AlgorithmMetricsReader(path)
    system_metrics_reader.refresh_event_file_list()
    
    NUM_CPU = 8
    NUM_GPU = 1

    dim_to_plot = ["CPU", "GPU"]
    events_to_plot = []
    for x in range(NUM_CPU):
        events_to_plot.append("cpu"+str(x))
    for x in range(NUM_GPU):
        events_to_plot.append("gpu"+str(x))
        
    resultant_heatmap = Heatmap(
        system_metrics_reader,
        framework_metrics_reader,
        select_dimensions=dim_to_plot,
        select_events=events_to_plot,
        plot_height=400
    )
    
    system_metrics_reader.refresh_event_file_list()
    resultant_metric_hist = MetricsHistogram(system_metrics_reader).plot( 
                        select_dimensions=dim_to_plot,
                        select_events=events_to_plot
    )
    
    return resultant_heatmap, resultant_metric_hist