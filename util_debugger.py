import time
import boto3
from datetime import datetime
import json 
import pandas as pd
from IPython.display import FileLink
from smdebug.profiler.system_metrics_reader import S3SystemMetricsReader
from smdebug.profiler.algorithm_metrics_reader import S3AlgorithmMetricsReader
from smdebug.profiler.analysis.notebook_utils.metrics_histogram import MetricsHistogram
from smdebug.profiler.analysis.notebook_utils.heatmap import Heatmap

def get_sys_metric(train_estimator, instance_type):
    
    NUM_CPU = 8
    NUM_GPU = 1
    
    if instance_type == 'ml.p3.8xlarge':
        NUM_CPU = 32
        NUM_GPU = 4
    
    if instance_type == 'ml.g4.12xlarge':
        NUM_CPU = 48
        NUM_GPU = 4

    path = train_estimator.latest_job_profiler_artifacts_path()
    
    system_metrics_reader = S3SystemMetricsReader(path)
    
    '''
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
    '''
    
    framework_metrics_reader = S3AlgorithmMetricsReader(path)
    
    
    dim_to_plot = ["CPU", "GPU"]
    events_to_plot = []
    for x in range(NUM_CPU):
        events_to_plot.append("cpu"+str(x))
    for x in range(NUM_GPU):
        events_to_plot.append("gpu"+str(x))
        
    system_metrics_reader.refresh_event_file_list()
    framework_metrics_reader.refresh_event_file_list()
    
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
    
    
    # Getting System Statistics from Profiler
    
    profiler_report_name = [
    rule["RuleConfigurationName"]
    for rule in train_estimator.latest_training_job.rule_job_summary()
    if "Profiler" in rule["RuleConfigurationName"]][0]
    profiler_report_name
    
    rule_output_path = train_estimator.output_path + '/' + train_estimator.latest_training_job.job_name + "/rule-output"
    s3_sys_usage_json_path = rule_output_path + "/" + profiler_report_name + "/profiler-output/profiler-reports/OverallSystemUsage.json"
    print ("Fetching data from: ", s3_sys_usage_json_path)
    
    path_without_s3_pre = s3_sys_usage_json_path.split("//",1)[1] 
    s3_bucket = path_without_s3_pre.split("/",1)[0]
    s3_prefix = path_without_s3_pre.split("/",1)[1]

    s3 = boto3.resource('s3')
    content_object = s3.Object(s3_bucket, s3_prefix)
    json_content = content_object.get()['Body'].read().decode('utf-8')
    sys_usage = json.loads(json_content)

    cpu_usage = sys_usage["Details"]["CPU"]["algo-1"]
    gpu_usage = sys_usage["Details"]["GPU"]["algo-1"]
    sys_util_data = [['CPU', NUM_CPU, cpu_usage["p50"], cpu_usage["p95"], cpu_usage["p99"]], 
                        ['GPU', NUM_GPU, gpu_usage["p50"], gpu_usage["p95"], gpu_usage["p99"]],
    ]
    sys_util_df = pd.DataFrame(sys_util_data, columns = ['Metric', '#', 'p50', 'p95', 'p99'])
    
    return resultant_heatmap, resultant_metric_hist, sys_util_df

