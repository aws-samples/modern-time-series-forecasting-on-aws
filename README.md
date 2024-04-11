# Time Series forecasting with AWS services

## Overview
This project demonstrates how to use AWS services to implement time-series forecasting. It contains examples for the following services and approaches:
1. Amazon SageMaker Canvas
2. Amazon SageMaker Autopilot
3. Amazon SageMaker DeepAR
4. AutoGluon
5. Chronos
6. Amazon SageMaker JumpStart
7. Amazon QuickSight forecast


## Getting started
To run the notebooks in this project you must use [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) which requires a [SageMaker domain](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-entity-status.html).

If you'd lke to create a new domain, you can follow the onboarding [instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) in the Developer Guide or use the provided AWS CloudFormation [template]() that creates a SageMaker domain, a user profile, and adds the IAM roles required for executing the provided notebooks.

## Datasets

1. For Canvas: `consumer_electronics`
2. For Autopilot
3. For SageMaker DeepAR
4. For Chronos


## Example 1: Amazon SageMaker Canvas
[Amazon SageMaker Canvas](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas.html)

Time-series forecast model
`consumer_electronics.csv` dataset

Canvas time-series model training uses the [Sagemaker Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html), which enables the use of various Autopilot’s public APIs. These include operations like `CreateAutoMLJobV2`, `ListCandidatesForAutoMLJob`, and `DescribeAutoMLJobV2` among others. This integration facilitates a streamlined process for training machine learning models directly within the Canvas environment.

[Time-series forecasting algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/timeseries-forecasting-algorithms.html)

### Experiment 1
**Quick build mode**

![](img/canvas-model-overview.png)

In quick build mode Canvas trains one model. The model and metrics are captured in SageMaker Experiments.

See Experiments in Studio Classic:
![](img/experiments-quick-builld.png)

Predict:

![](img/canvas-predictions.png)

To generate predictions after you build a model in Canvas, Canvas automatically deploys an asynchronous SageMaker endpoint into your AWS account. The endpoint is temporary and Canvas uses it to generate single prediction. For batch predictions, Canvas starts a SageMaker batch transform job. The endpoint deployed by Canvas can be used only for in-app predictions and cannot be used outside Canvas.

### Experiment 2
**Standard build mode**

Customizing model build:

![](img/canvas-configure-model-config.png)
![](img/canvas-configure-model-metric.png)
![](img/canvas-configure-model-quantiles.png)

In standard build mode Canvas trains the [six built-in algorithms]((https://docs.aws.amazon.com/sagemaker/latest/dg/timeseries-forecasting-algorithms.html)) with your target time-series. Then, using a stacking ensemble method, it combines these model candidates to create an optimal forecasting model for a given objective metric.

All models and metrics are captured in SageMaker experiments.

See Experiments in Studio Classic:

![](img/experiments-standard-builld.png)

Predict:

![](img/canvas-predictions-standard-build.png)

### Model deployment
Currently Canvas doesn't support model deployment to SageMaker from the UX. To deploy the trained model as a SageMaker real-time endpoint or use it in batch transform outside of Canvas UX, you can use any of two options:

- [Autopilot model deployment and forecasts](https://docs.aws.amazon.com/sagemaker/latest/dg/timeseries-forecasting-deploy-models.html)
- boto3 Python SDK as shown in Section 4 in [Time-Series Forecasting with Amazon SageMaker Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/autopilot_time_series.ipynb) notebook

To keep your model in a central model registry you can register the model directly from Canvas UX:

![](img/canvas-add-to-model-registry.png)

Canvas also shows the model package details:

![](img/model-registry-details.png)

## Example 2: Amazon SageMaker Autopilot API

[Amazon SageMaker Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html)

Note: previous Autopilot UX in Studio Classic merged with Canvas as of re:Invent 2023. All AutoML functionality is moved to Canvas as of now.

[Time-series forecasting algorithms in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/timeseries-forecasting-algorithms.html)

Example notebook [Time-Series Forecasting with Amazon SageMaker Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/autopilot_time_series.ipynb)

## Example 3: Amazon SageMaker DeepAR

## Example 4: AutoGluon

## Example 5: Chronos

## Example 6: Amazon SageMaker JumpStart

## Example 7: Amazon QuickSight forecast

## Results

## Resources
- [Time-series forecasting with AWS services workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/caef4710-3721-4957-a2ce-33799920ef72/en-US)
- [Chronos forecasting GitHub repository](https://github.com/amazon-science/chronos-forecasting)
- [Adapting language model architectures for time series forecasting](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/pdf/2403.07815.pdf)
- [AutoGluon](https://github.com/autogluon/autogluon)


### Blog posts
- [Robust time series forecasting with MLOps on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/robust-time-series-forecasting-with-mlops-on-amazon-sagemaker/)
- [Deep demand forecasting with Amazon SageMaker]()

---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0