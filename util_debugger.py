import boto3
import json 
import pandas as pd
from smdebug.profiler.system_metrics_reader import S3SystemMetricsReader
from smdebug.profiler.algorithm_metrics_reader import S3AlgorithmMetricsReader
from smdebug.profiler.analysis.notebook_utils.metrics_histogram import MetricsHistogram
from smdebug.profiler.analysis.notebook_utils.heatmap import Heatmap

"""
Method to get system metrics for the training job
"""


def get_sys_metric(train_estimator, num_cpu, num_gpu):
    path = train_estimator.latest_job_profiler_artifacts_path()
    system_metrics_reader = S3SystemMetricsReader(path)    
    framework_metrics_reader = S3AlgorithmMetricsReader(path)

    """
    Metric histograms and Heatmaps of system usage
    """
    
    dim_to_plot = ["CPU", "GPU"]
    events_to_plot = []

    for x in range(num_cpu):
        events_to_plot.append("cpu"+str(x))
    for x in range(num_gpu):
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

    """
    Fetching system statistics from profiler report
    """

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
    sys_util_data = [['CPU', num_cpu, cpu_usage["p50"], cpu_usage["p95"], cpu_usage["p99"]], 
                        ['GPU', num_gpu, gpu_usage["p50"], gpu_usage["p95"], gpu_usage["p99"]],
    ]
    sys_util_df = pd.DataFrame(sys_util_data, columns = ['Metric', '#', 'p50', 'p95', 'p99'])
    
    return resultant_heatmap, resultant_metric_hist, sys_util_df

