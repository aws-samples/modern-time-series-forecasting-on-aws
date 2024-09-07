#!/bin/bash

set -e  # Exit immediately if any command exits with a non-zero status

if [ ! -z "${SM_JOB_DEF_VERSION}" ]
then
   echo "Running in job mode, skip lcc"
else
   echo "Cloning repository..."
   git clone https://github.com/aws-samples/modern-time-series-forecasting-on-aws.git || { echo "Error: Failed to clone repository"; exit 0; }
   echo "Files cloned from GitHub repo"
fi
