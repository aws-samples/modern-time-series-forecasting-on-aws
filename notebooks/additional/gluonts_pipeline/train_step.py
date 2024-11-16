import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union
from gluonts.dataset.jsonl import JsonLinesFile
from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.torch import TemporalFusionTransformerEstimator
import boto3

def _upload_directory_to_s3(local_dir, bucket_name, s3_prefix):
    s3_client = boto3.client('s3')
    
    # Ensure the local directory path ends with a separator
    local_dir = os.path.join(local_dir, '')
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            # Get the full local path
            local_path = os.path.join(root, filename)
            
            # Calculate relative path from the local directory
            relative_path = os.path.relpath(local_path, local_dir)
            
            # Create S3 key with prefix
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            try:
                print(f"Uploading {local_path} to {bucket_name}/{s3_key}")
                s3_client.upload_file(local_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")


def _load_dataset(path, freq):
    return ListDataset(JsonLinesFile(path=path), freq=freq)


def _evaluate(
    predictor: Predictor,
    test_data: ListDataset,
    prediction_length: int,
    quantiles: List[float] = None,
    num_windows: int = 1,
    num_samples: int = 20,
) -> Tuple[Dict[str, float], pd.DataFrame]:

    # prepare test pairs
    # the testing windows are taken from the end of the dataset
    _, test_template = OffsetSplitter(offset=-num_windows*prediction_length).split(test_data)

    test_pairs = test_template.generate_instances(
        prediction_length=prediction_length, 
        windows=num_windows, 
    )

    # predict
    forecasts = predictor.predict(test_pairs.input, num_samples)
    
    # evaluate
    evaluator = Evaluator(quantiles=quantiles if quantiles else (np.arange(10) / 10.0)[1:])

    return evaluator([to_pandas(l) for l in test_pairs.label], forecasts)


def _train_predictor(
    dataset: ListDataset,
    trainer_hp,
    model_hp,
) -> Predictor:
    return TemporalFusionTransformerEstimator(
        **model_hp,
        trainer_kwargs={"max_epochs": trainer_hp['epochs']},
    ).train(dataset)

def _save_predictor(predictor: Predictor, model_dir: Path):
    predictor.serialize(model_dir)


def train(
    train_data_s3_path,
    test_data_s3_path,
    output_s3_prefix,
    hyperparameters,
    
)-> Dict[str, Union[str,float]]:
    """
    Trains the TFT predictor
    """
    
    freq = hyperparameters['freq']
    prediction_length = hyperparameters['prediction_length']
    
    # download datasets from S3
    print(f'Download datasets from {train_data_s3_path} and {test_data_s3_path}')
    os.makedirs("./data", exist_ok=True)
    s3 = boto3.client('s3')

    train_filename = train_data_s3_path.split('/')[-1]
    test_filename = test_data_s3_path.split('/')[-1]
    s3.download_file(
        train_data_s3_path.split('/')[2], 
        '/'.join(train_data_s3_path.split('/')[3:]), 
        f'./data/{train_filename}'
    )
    s3.download_file(
        test_data_s3_path.split('/')[2], 
        '/'.join(test_data_s3_path.split('/')[3:]), 
        f'./data/{test_filename}'
    )

    # load datasets into GluonTS format
    print(f'Load datasets into GluonTS format')
    train_ds = _load_dataset(Path(f'./data/{train_filename}'), freq)
    test_ds = _load_dataset(Path(f'./data/{test_filename}'), freq)

    train_entry = next(iter(train_ds))
    test_entry = next(iter(test_ds))
    len_train = train_entry['target'].shape[0]
    len_test = test_entry['target'].shape[0]
    
    print(f'--------------------------------------------------------')
    print(f"The test dataset contains {len(train_ds)} time series: {[e['item_id'] for e in train_ds]}")
    print(f"The test dataset starts {test_entry['start'].to_timestamp()} and ends {test_entry['start'] + len_test} and contains {len_test} data points")
    print(f"The train dataset starts {train_entry['start']} and ends {train_entry['start'] + len_train} and contains {len_train} data points")
    print(f"The backtest contains {len_test-len_train} data points and has {(len_test-len_train)/prediction_length} windows of {prediction_length} length")
    print(f'--------------------------------------------------------')

    # training
    print(f"Training the predictor for {hyperparameters['epochs']} epochs")
    predictor = _train_predictor(
        train_ds, 
        {
            'epochs':hyperparameters['epochs'],
        }, 
        {
            'freq':hyperparameters['freq'],
            'prediction_length':hyperparameters['prediction_length'],
            'context_length':hyperparameters['context_length'],
        },
    )

    # evaluation
    print(f"Evaluating the model on {hyperparameters['backtest_windows']} rolling windows")
    agg_metrics, item_metrics = _evaluate(
        predictor,
        test_ds,
        prediction_length,
        [float(x) for x in hyperparameters['quantiles'].split(',')],
        hyperparameters['backtest_windows'],
        hyperparameters['num_samples'],
    )

    # save predictor and results
    os.makedirs('./output/model', exist_ok=True)
    
    with open(os.path.join('./output', 'agg_metrics.json'), 'w', encoding="utf-8") as fout:
        json.dump(agg_metrics, fout)

    item_metrics.to_csv(
        os.path.join('./output', 'item_metrics.csv.gz'),
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    _save_predictor(predictor, Path('./output/model'))

    # upload artifacts to S3
    metrics_s3_path = f'{output_s3_prefix}'
    model_s3_path = f'{output_s3_prefix}/model'

    print(f'Upload metrics to {metrics_s3_path} and {model_s3_path}')
    _upload_directory_to_s3('./output', metrics_s3_path.split('/')[2], '/'.join(metrics_s3_path.split('/')[3:]))
    _upload_directory_to_s3('./output/model', model_s3_path.split('/')[2], '/'.join(model_s3_path.split('/')[3:]))

    print('### Training completed. Exiting.')
    
    return {
        'metrics_s3_path':metrics_s3_path,
        'model_s3_path':model_s3_path,
        'agg_metrics':agg_metrics,
    }
    
    
