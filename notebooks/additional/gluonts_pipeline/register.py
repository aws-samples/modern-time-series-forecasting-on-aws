from typing import Dict
from sagemaker.estimator import Estimator

def register(
    training_job_name,
    model_package_group_name,
    model_approval_status='PendingManualApproval',
    pipeline_run_id=None,
)-> Dict[str,str]:
    """
    Register model trained by the pipeline trained job in SageMaker model registry.
    """

    print(f'Attaching estimator to the job {training_job_name}')
    estimator = Estimator.attach(training_job_name)

    print(f'Registering the model in {model_package_group_name}')
    supported_instances = ['ml.m5.xlarge', 'ml.m5.2xlarge', "ml.g5.xlarge", 'ml.g5.2xlarge']
    
    model_package = estimator.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=supported_instances,
        transform_instances=supported_instances,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_name="gluonts-tft-model",
        domain="MACHINE_LEARNING",
        task="OTHER", 
        framework='PYTORCH',
        framework_version='2.3',
        description='GluonTS TFT model group',
        customer_metadata_properties={
            'pipeline_run_id':pipeline_run_id,
        }
    )

    print('### Model registration completed. Exiting.')
    
    return {
        "model_package_arn":model_package.model_package_arn,
        "model_package_group_name":model_package_group_name,
    }