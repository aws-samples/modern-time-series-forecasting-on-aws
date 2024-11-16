import os
from typing import Dict
from gluonts.dataset.jsonl import JsonLinesWriter
from pathlib import Path
from gluonts.dataset.split import DateSplitter
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import numpy as np
import boto3
import zipfile
import json

def preprocess(
    input_data_s3_path,
    output_s3_prefix,
    freq,
    prediction_length,
    data_start,
    data_end,
    backtest_windows=4,
    sample_size=0,
) -> Dict[str,str]:
    """
    Prepares time series data for training. 
    """
    
    # load raw dataset 
    print(f'Downloading from {input_data_s3_path}')

    os.makedirs("./data", exist_ok=True)
    s3 = boto3.client('s3')

    dataset_zip_filename = input_data_s3_path.split('/')[-1]
    s3.download_file(
        input_data_s3_path.split('/')[2], 
        '/'.join(input_data_s3_path.split('/')[3:]), 
        f'./data/{dataset_zip_filename}'
    )

    print(f'Unzipping {dataset_zip_filename}')
    zip_ref = zipfile.ZipFile(f'./data/{dataset_zip_filename}', 'r')
    zip_ref.extractall('./data')
    zip_ref.close()
    dataset_path = '.'.join(zip_ref.filename.split('.')[:-1])

    # load into DataFrame and resample
    # supported frequences for this example are 1h or 1d only
    print(f'Load dataset from {dataset_path} and resample to {freq} frequency')
    data_kw = pd.read_csv(
        dataset_path, 
        sep=';', 
        index_col=0,
        decimal=',',
        parse_dates=True,
    ).resample(freq).sum() / {'1h':4, '1d':'96'}[freq]

    # get the full dataset or a random sample of sample_size
    if sample_size != 0:
        print(f'Get a sample of {sample_size} time series out of the full dataset')
        ts_sample = data_kw[np.random.choice(data_kw.columns.to_list(), size=sample_size, replace=False)]
    else:
        print(f'Get the full dataset')
        ts_sample = data_kw

    # calculate the end of the training part based on backtest_windows
    end_training_date = pd.Period(data_end, freq=freq) - backtest_windows*prediction_length

    # convert to GluonTS format
    ts_dataset = PandasDataset(
        dict(ts_sample[(ts_sample.index > data_start) & (ts_sample.index <= data_end)])
    )
    # split to get the train dataset
    train_ds, _ = DateSplitter(date=end_training_date).split(ts_dataset)

    test_entry = next(iter(ts_dataset))
    train_entry = next(iter(train_ds))
    len_test = len(test_entry['target'])
    len_train = len(train_entry['target'])

    print(f'--------------------------------------------------------')
    print(f"The test dataset contains {len(ts_dataset)} time series")
    print(f"The test dataset starts {test_entry['start'].to_timestamp()} and ends {test_entry['start'] + len_test} and contains {len_test} data points")
    print(f"The train dataset starts {train_entry['start']} and ends {train_entry['start'] + len_train} and contains {len_train} data points")
    print(f"The backtest contains {len_test-len_train} data points and has {(len_test-len_train)/prediction_length} windows of {prediction_length} length")
    print(f'--------------------------------------------------------')

    # save train and test datasets
    train_file_name = 'train.jsonl.gz'
    test_file_name = 'test.jsonl.gz'
    train_file_path = Path(f'./data/{train_file_name}')
    test_file_path = Path(f'./data/{test_file_name}')
    train_file_s3_path = f'{output_s3_prefix}/train/{train_file_name}'
    test_file_s3_path = f'{output_s3_prefix}/test/{test_file_name}'
    
    JsonLinesWriter().write_to_file(train_ds, train_file_path)
    JsonLinesWriter().write_to_file(ts_dataset, test_file_path)

    # upload files to S3
    print(f'Upload train and test datasets to {train_file_s3_path} and {test_file_s3_path}')
    s3.upload_file(
        train_file_path, 
        train_file_s3_path.split('/')[2],
        '/'.join(train_file_s3_path.split('/')[3:])
    )
    s3.upload_file(
        test_file_path, 
        test_file_s3_path.split('/')[2],
        '/'.join(test_file_s3_path.split('/')[3:])
    )
    
    print('### Data processing completed. Exiting.')

    return {
        'train_data':train_file_s3_path,
        'test_data':test_file_s3_path,
    }













