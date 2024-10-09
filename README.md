# Modern Time Series Forecasting on AWS

## Overview
This workshop demonstrates how to use AWS services to implement time series forecasting. It covers the following examples and AWS services:
1. Amazon SageMaker Canvas
2. Amazon SageMaker Autopilot API
3. Amazon SageMaker DeepAR
4. Chronos
5. AutoGluon
6. Amazon SageMaker custom algorithm
7. Amazon QuickSight forecast

The [workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/76a419ba-6303-4e7c-ac02-47112ed7cb3f/en-US) is available in AWS workshop catalog. You can run this workshop on an AWS-led event or in your own AWS account.

## How to use this workshop
To use this workshop, you need an Amazon SageMaker domain. All workshop content is in Jupyter notebooks running on Amazon SageMaker. To get started, follow the instructions in the **Getting started** section. To clean up resources, follow the instructions in the **Clean-up** section. You can execute the notebooks in any order, and you don't need to switch between the notebooks and the workshop web page.

## Required resources

**Ignore this section if you're using an AWS-provided account as a part of an AWS-led workshop.**

In order to be able to run notebooks and complete workshop labs you need access to the following resources in your AWS account. You can check quotas for all following resources in AWS console in [Service Quotas](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas) console.

**Studio JupyterLab app**  
Minimal required instance type is `ml.m5.2xlarge`. We recommend to use `ml.m5.4xlarge` as an instance to run all notebooks. If you have access to GPU-instances like `ml.g5.4xlarge` or `ml.g6.4xlarge`, use these instance to run the notebooks. 

To experiment with the full dataset with 370 time series in the lab 5 AutoGluon you need a GPU instance for the notebook - `ml.g5.4xlarge`/`ml.g6.4xlarge` or `ml.g5.8xlarge`/`ml.g6.8xlarge`.

- Check quota for [`ml.m5.2xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-7C9662F1)
- Check quota for [`ml.m5.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-2CA31BFA)
- Check quota for [`ml.g5.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-81940D85)
- Check quota for [`ml.g6.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-692B8304)
- Check quota for [`ml.g5.8xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-19B6BAFC)
- Check quota for [`ml.g6.8xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-804C2AFF)


**Number of concurrent AutoML Jobs**  
To follow the optimal flow of the workshop, you need to run at least three AutoML jobs in parallel. We recommend to have a quota set to six or more concurrent jobs.

- Check quota for [maximum number of concurrent AutoML Jobs](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-CFC2D5B6)

**Training jobs**  
To run a training job for DeepAR algorithm you need a `ml.c5.4xlarge` compute instance

- Check quota for [`ml.c5.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-E7898792)

**SageMaker real-time inference endpoints**  
DeepAR, Chronos, and AutoGluon notebooks deploy SageMaker real-time inference endpoints to test models. You need access to the following compute instances for endpoint use:
- Minimal for Autopilot and DeepAR endpoints: check [`ml.m5.xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-2F737F8D) 
- Recommended for Autopilot and DeepAR endpoints: check [`ml.m5.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-E2649D46)
- Minimal for Chronos Small endpoint: check [`ml.g5.xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-1928E07B)
- Optional for Chronos Base: check [`ml.g5.2xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-9614C779)
- Optional for Chronos Large: check [`ml.g5.4xlarge`](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-C1B9A48D)


## Workshop flow
The notebooks from Lab 1 to Lab 5 are self-sufficient. You can run them in any order. If you're unfamiliar with time series forecasting, we recommend starting with the Lab 1 notebook and continuing from there. Alternatively, you can run only the notebooks that interest you, such as `lab4_chronos` or `lab5_autogluon`.

The model training in Labs 1, 2, and 3 takes 15-40 minutes, depending on the algorithm. You don't need to wait for the training to complete before moving on to the next notebook. You can come back to the previous notebook once the training is done.

Executing all five notebooks will take 2-3 hours. If you're new to time series forecasting, Jupyter notebooks, or Python, it may take longer.

## Workshop costs
The notebooks in this workshop create cost-generating resources in your account. Make sure you always delete created SageMaker inference endpoints, log out of Canvas, and stop JupyterLab spaces if you don't use them.

If running all notebooks with all sections, including optional sections and training three models using **Standard** builds in Canvas, the estimated cost is approximately 90-100 USD. 

Please note that your actual costs may vary depending on the duration of the workshop, the number of inference endpoints created, and the time the endpoints remain in service.

To optimize costs, follow these recommendations:
1. Run only **Quick** builds in Canvas to minimize costs. Note that in this case you cannot download model performance JSON files
2. Use only a sample from the full dataset to train models and run all experiments. Each notebook contains code to create a small dataset with a sample from the time series
3. Promptly delete SageMaker inference endpoints after use
4. Use `ml.m5.xlarge` instance for JupyterLab app to balance performance and cost
5. Limit Chronos experiments to one endpoint and a sample of the time series in the notebook `lab4_chronos`

## Getting started
If you'd lke to create a new domain, you have two options:
1. Use the provided AWS CloudFormation [template](./cfn-templates/sagemaker-domain.yaml) that creates a SageMaker domain, a user profile, and adds the IAM roles required for executing the provided notebooks - this is the recommended approach
1. Follow the onboarding [instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) in the Developer Guide and create a new domain and a user profile via AWS Console

## Datasets

All examples and notebooks in this workshop using the same real-world dataset. It makes possible to compare performance and model metrics across different approaches. 

You use the [electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) from the repository of the University of California, Irvine:
> Trindade, Artur. (2015). ElectricityLoadDiagrams20112014. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86.


## Example 1: Amazon SageMaker Canvas

Open the [lab 1 notebook](./notebooks/lab1_sagemaker_canvas.ipynb) and follow the instructions.

Additional SageMaker Canvas links:
- [Time Series Forecasts in Amazon SageMaker Canvas](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-time-series.html)
- [Canvas Workshop - time series forecast lab](https://catalog.workshops.aws/canvas-immersion-day/en-US/1-use-cases/3-retail)
- [Time-Series Forecasting Using Amazon SageMaker Canvas](https://catalog.us-east-1.prod.workshops.aws/workshops/866925a4-cb5f-4a3d-9cd7-80edc0aa5f0c/en-US/4-0sagemakercanvas)


## Example 2: Amazon SageMaker Autopilot API

Open the [lab 2 notebook](./notebooks/lab2_sagemaker_autopilot_api.ipynb) and follow the instructions.

Note: previous Autopilot UX in Studio Classic merged with Canvas as of re:Invent 2023. All AutoML functionality is moved to Canvas as of now.

Additional SageMaker Autopilot API links:
- [Amazon SageMaker Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html)
- [Time series forecasting algorithms in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/timeseries-forecasting-algorithms.html)
- Example notebook [Time series Forecasting with Amazon SageMaker Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/autopilot_time_series.ipynb)
- [Lab 2 - Demand Forecasting with SageMaker Autopilot API](https://catalog.us-east-1.prod.workshops.aws/workshops/caef4710-3721-4957-a2ce-33799920ef72/en-US/40-sagemakerautopilot)
- [Time series forecasting with Amazon SageMaker AutoML](https://aws.amazon.com/it/blogs/machine-learning/time-series-forecasting-with-amazon-sagemaker-automl/)

## Example 3: Amazon SageMaker DeepAR

Open the [lab 3 notebook](./notebooks/lab3_sagemaker_deepar.ipynb) and follow the instructions.

Additional DeepAR links:
- [Use the SageMaker DeepAR forecasting algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
- [Deep AR Forecasting](https://sagemaker.readthedocs.io/en/stable/algorithms/time_series/deep_ar.html)
- [Example notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/deepar_electricity/DeepAR-Electricity.ipynb)
- [Deep Demand Forecasting with Amazon SageMaker notebook](https://github.com/awslabs/sagemaker-deep-demand-forecast/blob/mainline/src/deep-demand-forecast.ipynb)
- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
- [Predictive Analytics with Time-series Machine Learning on Amazon Timestream](https://aws.amazon.com/blogs/database/predictive-analytics-with-time-series-machine-learning-on-amazon-timestream/)
- [Bike-Share Demand Forecasting 2b: SageMaker DeepAR Algorithm](https://github.com/aws-samples/time-series-forecasting-on-aws/blob/main/2b_SageMaker_Built-In_DeepAR.ipynb)

## Example 4: Chronos

Open the [lab 4 notebook](./notebooks/lab4_chronos.ipynb) and follow the instructions.

Links to more Chronos content:
- [Chronos models on Huggingface](https://huggingface.co/amazon/chronos-t5-large)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Lot of Chronos-related content on Chronos GitHub](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file#-coverage)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/html/2403.07815v1)
- [Adapting language model architectures for time series forecasting](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting)
- [Evaluating Chronos models](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/README.md#evaluating-chronos-models)
- [Chronos-related content on Chronos GitHub](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file#-coverage)


## Example 5: AutoGluon

Open the [lab 5 notebook](./notebooks/lab5_autogluon.ipynb) and follow the instructions.

Links to AutoGluon content:
- AutoGluon time series
    - [AutoGluon time series forecasting](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- AutoGluon Chronos
    - [AutoGluon forecasting with Chronos](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
    - [Forecasting with Chronos notebook Colab](https://colab.research.google.com/github/autogluon/autogluon/blob/stable/docs/tutorials/timeseries/forecasting-chronos.ipynb)
- [AutoGluon Cloud](https://auto.gluon.ai/cloud/dev/tutorials/autogluon-cloud.html)


## Example 6: Amazon SageMaker custom algorithm

This example is under development.

Refer to the following resources to see how you can run custom algorithms on SageMaker:
- [Robust time series forecasting with MLOps on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/robust-time-series-forecasting-with-mlops-on-amazon-sagemaker/)
- [Deep demand forecast with Amazon SageMaker](https://github.com/awslabs/sagemaker-deep-demand-forecast)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI'21 Best Paper)](https://github.com/aws-samples/time-series-forecasting-on-aws/blob/main/3_SagaMaker_Custom_algorithm_Informer.ipynb)
- [GluonTS - Probabilistic Time Series Modeling in Python](https://github.com/awslabs/gluonts)


## Example 7: Amazon QuickSight forecast
[Amazon QuickSight](https://docs.aws.amazon.com/quicksight/latest/user/welcome.html) has ML features to give you hidden insights and trends in your data. One of these ML features is **ML-powered forecast**. The built-in ML forecast uses [Random Cut Forest (RCF) algorithm](https://docs.aws.amazon.com/quicksight/latest/user/concept-of-ml-algorithms.html) to detect seasonality, trends, exclude outliers, and impute missing values. For more details on how QuickSight uses RCF to generate forecasts, see the [developer guide](https://docs.aws.amazon.com/quicksight/latest/user/how-does-rcf-generate-forecasts.html).

![](img/quicksight_filter_item_101_store_001.png)

You can customize multiple settings on the **Forecast properties** pane, such as number of forecast periods, prediction interval, seasonality, and forecast boundaries.

For more details refer to the Developer Guide [Forecasting and creating what-if scenarios with Amazon QuickSight](https://docs.aws.amazon.com/quicksight/latest/user/forecasts-and-whatifs.html).

Besides a graphical forecasting, you can also add a forecast as a narrative in an insight widget. To learn more, see [Creating autonarratives with Amazon QuickSight](https://docs.aws.amazon.com/quicksight/latest/user/narratives-creating.html).

Additional resources for Amazon QuickSight forecasting:
- [ML-powered forecasting](https://docs.aws.amazon.com/quicksight/latest/user/forecast-function.html)

## Results and comparison

Open the [lab 6 notebook](./notebooks/lab_final_results.ipnyb) and follow the instructions.

Additional resources about time series forecast accuracy evaluation
- [Evaluating Predictor Accuracy](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html)
- [Evaluating Chronos models](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts#evaluating-chronos-models)
- [Forecasting time series - evaluation metrics](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html)

## Clean up
To avoid unnecessary costs, you must remove all project-provisioned and generated resources from your AWS account. 

### Shut down SageMaker resources
You must complete this section before deleting the SageMaker domain or the CloudFormation stack.

Complete the following activities to shut down your Amazon SageMaker resources:
- [Log out of Canvas](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-log-out.html)
- Make sure to delete all endpoints created by this workshop including Canvas asynchronous endpoints
- [Stop running applications and spaces in Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running.html#studio-updated-running-stop) > follow the instructions in the section **Use the Studio UI to delete your domain applications**

### Remove the SageMaker domain
You don't need to complete this section if you run an AWS-instructor led workshop in an AWS-provisioned account.

If you used the AWS Console to provision a Studio domain for this workshop, and don't need the domain, you can delete the domain by following the instructions in the [Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-delete-domain.html). 

If you provisioned a Studio domain with the provided CloudFormation template, you can delete the CloudFormation stack in the AWS console.

If you provisioned a new VPC for the domain, go to the [VPC console](https://console.aws.amazon.com/vpc/home?#vpcs) and delete the provisioned VPC.


## Resources

### Algorithms
- [References for machine learning and RCF](https://docs.aws.amazon.com/quicksight/latest/user/learn-more-about-machine-learning-and-rcf.html)
- [Chronos forecasting GitHub repository](https://github.com/amazon-science/chronos-forecasting)
- [Adapting language model architectures for time series forecasting](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/pdf/2403.07815.pdf)
- [AutoGluon](https://github.com/autogluon/autogluon)
- [AutoGluon Time series forecasting](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [GluonTS - Probabilistic Time Series Modeling in Python](https://github.com/awslabs/gluonts)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)

### Books and whitepapers
- [Time Series Analysis on AWS: Learn how to build forecasting models and detect anomalies in your time series data](https://www.amazon.com/Time-Analysis-AWS-forecasting-anomalies-ebook/dp/B09MMLLWDY)
- [Time Series Forecasting Principles with Amazon Forecast](https://docs.aws.amazon.com/whitepapers/latest/time-series-forecasting-principles-with-amazon-forecast/time-series-forecasting-principles-with-amazon-forecast.html)
- [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/pdf/2310.07820)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)

### Blog posts
- [Robust time series forecasting with MLOps on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/robust-time-series-forecasting-with-mlops-on-amazon-sagemaker/)
- [Deep demand forecasting with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/deep-demand-forecasting-with-amazon-sagemaker/)
- [Capture public health insights more quickly with no-code machine learning using Amazon SageMaker Canvas](https://aws.amazon.com/blogs/machine-learning/capture-public-health-insights-more-quickly-with-no-code-machine-learning-using-amazon-sagemaker-canvas/)
- [Speed up your time series forecasting by up to 50 percent with Amazon SageMaker Canvas UI and AutoML APIs](https://aws.amazon.com/blogs/machine-learning/speed-up-your-time-series-forecasting-by-up-to-50-percent-with-amazon-sagemaker-canvas-ui-and-automl-apis/)
- [Sagemaker Automated Model Tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/)
- [Time series forecasting with Amazon SageMaker AutoML](https://aws.amazon.com/it/blogs/machine-learning/time-series-forecasting-with-amazon-sagemaker-automl/)

### Workshops and notebooks
- [Time series forecasting with AWS services workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/caef4710-3721-4957-a2ce-33799920ef72/en-US)
- [Time series Forecasting with Amazon SageMaker Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/autopilot_time_series.ipynb)
- [Deep Demand Forecasting with Amazon SageMaker notebook](https://github.com/awslabs/sagemaker-deep-demand-forecast/blob/mainline/src/deep-demand-forecast.ipynb)
- [Timeseries Forecasting on AWS](https://github.com/aws-samples/time-series-forecasting-on-aws)
- [Inventory Forecasting using Amazon SageMaker](https://catalog.us-east-1.prod.workshops.aws/workshops/866925a4-cb5f-4a3d-9cd7-80edc0aa5f0c/en-US)


## QR codes and links

### This GitHub repository
Link: https://github.com/aws-samples/modern-time-series-forecasting-on-aws  
Short link: https://bit.ly/47hnKH6

![](./img/git-repo-qr-code.png)

### AWS workshop
Link: https://catalog.workshops.aws/modern-time-series-forecasting-on-aws/en-US  
Short link: https://bit.ly/4dBQ0G8

![](./img/workshop-qr-code.svg)
---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0