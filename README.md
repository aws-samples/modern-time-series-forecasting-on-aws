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

## Getting started
To run the notebooks in this project you must use [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) which requires a [SageMaker domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html).

If you'd lke to create a new domain, you have two options:
1. Use the provided AWS CloudFormation [template](./cfn-templates/sagemaker-domain.yaml) that creates a SageMaker domain, a user profile, and adds the IAM roles required for executing the provided notebooks - this is the recommended approach
1. Follow the onboarding [instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) in the Developer Guide and create a new domain and a user profile via AWS Console

## Workshop flow
You can run all notebooks independently of each other and in any order. We recommend to start with the notebook `lab1_sagemaker_canvas` if you'd like to run the whole workshop.

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
To avoid charges, you must remove all project-provisioned and generated resources from your AWS account. 

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

### Books and whitepapers
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

### Workshops and notebooks
- [Time series forecasting with AWS services workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/caef4710-3721-4957-a2ce-33799920ef72/en-US)
- [Time series Forecasting with Amazon SageMaker Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/autopilot_time_series.ipynb)
- [Deep Demand Forecasting with Amazon SageMaker notebook](https://github.com/awslabs/sagemaker-deep-demand-forecast/blob/mainline/src/deep-demand-forecast.ipynb)
- [Timeseries Forecasting on AWS](https://github.com/aws-samples/time-series-forecasting-on-aws)
- [Inventory Forecasting using Amazon SageMaker](https://catalog.us-east-1.prod.workshops.aws/workshops/866925a4-cb5f-4a3d-9cd7-80edc0aa5f0c/en-US)


## QR code for this GitHub repository

Short link: https://bit.ly/47hnKH6

![](./img/git-repo-qr-code.png)

---

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0