import os
import argparse
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    aa = parser.add_argument

    # data, model, and output directories. Defaults are set in the environment variables.
    aa('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    aa('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    aa('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    aa('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    aa('--sm_training_env', type=str, default=os.environ.get('SM_TRAINING_ENV'))

    args, _ = parser.parse_known_args()
    print(f'Passed arguments: {args}')

    # get SageMaker enviroment setup
    sm_training_env = json.loads(args.sm_training_env)
    
    # hyperparameters
    hyperparameters = sm_training_env['hyperparameters']

    # load datasets into GluonTS format
    print(f'Load datasets into GluonTS format')
    train_ds = _load_dataset(Path(f'{args.train_dir}/train.jsonl.gz'), hyperparameters['freq'])
    test_ds = _load_dataset(Path(f'{args.test_dir}/test.jsonl.gz'), hyperparameters['freq'])

    train_entry = next(iter(train_ds))
    test_entry = next(iter(test_ds))
    len_train = train_entry['target'].shape[0]
    len_test = test_entry['target'].shape[0]
    
    print(f'--------------------------------------------------------')
    print(f"The test dataset contains {len(train_ds)} time series: {[e['item_id'] for e in train_ds]}")
    print(f"The test dataset starts {test_entry['start'].to_timestamp()} and ends {test_entry['start'] + len_test} and contains {len_test} data points")
    print(f"The train dataset starts {train_entry['start']} and ends {train_entry['start'] + len_train} and contains {len_train} data points")
    print(f"The backtest contains {len_test-len_train} data points and has {(len_test-len_train)/hyperparameters['prediction_length']} windows of {hyperparameters['prediction_length']} length")
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

    # emit training metrics - SageMaker collects them from the log stream
    # TBD
    
    # evaluation
    print(f"Evaluating the model on {hyperparameters['backtest_windows']} rolling windows")
    agg_metrics, item_metrics = _evaluate(
        predictor,
        test_ds,
        hyperparameters['prediction_length'],
        [float(x) for x in hyperparameters['quantiles'].split(',')],
        hyperparameters['backtest_windows'],
        hyperparameters['num_samples'],
    )

    # emit test metrics - SageMaker collects them from the log stream
    print(f"test_MSE:{agg_metrics['MSE']}")
    print(f"test_MAPE:{agg_metrics['MAPE']}")
    print(f"test_sMAPE:{agg_metrics['sMAPE']}")
    print(f"test_RMSE:{agg_metrics['RMSE']}")
    print(f"test_mean_wQuantileLoss:{agg_metrics['mean_wQuantileLoss']}")
    print(f"test_mean_absolute_QuantileLoss:{agg_metrics['mean_absolute_QuantileLoss']}")

    # save predictor and results
    # os.makedirs('./output/model', exist_ok=True)
    
    with open(os.path.join(args.output_data_dir, 'agg_metrics.json'), 'w', encoding="utf-8") as fout:
        json.dump(agg_metrics, fout)

    item_metrics.to_csv(
        os.path.join(args.output_data_dir, 'item_metrics.csv.gz'),
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    _save_predictor(predictor, Path(args.model_dir))

    print('### Training completed. Exiting.')