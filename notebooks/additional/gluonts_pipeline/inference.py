import os
import json
from typing import Any, List, Dict, Union
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import QuantileForecast
import numpy as np

class QuantileForecastEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, QuantileForecast):
            return {
                "__type__": "QuantileForecast",
                "forecast_arrays": obj.forecast_array.tolist(),
                "start_date": obj.start_date.to_timestamp().isoformat() if obj.start_date else None,
                "forecast_keys": obj.forecast_keys,
                "item_id": obj.item_id,
                "info": obj.info,
                "freq": obj.freq.freqstr,
                "prediction_length": obj.prediction_length,
                
            }
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def model_fn(model_dir: str) -> Predictor:
    print("loading model from {model_dir}")
    predictor = Predictor.deserialize(Path(model_dir))
    print("model was loaded successfully from {model_dir}")
    return predictor


def transform_fn(
    model: Predictor,
    request_body: Any, 
    content_type: Any, 
    accept_type: Any
):
    # print(f'get {request_body}')
    request_data = json.loads(request_body)

    parameters = request_data['parameters']
    request_list_data = ListDataset(
        request_data['inputs'],
        freq=parameters['freq'],
    )

    forecasts = list(model.predict(request_list_data, num_samples=parameters['num_samples']))
    return json.dumps(forecasts, cls=QuantileForecastEncoder), content_type