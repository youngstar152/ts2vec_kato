import os

import numpy as np
import pandas as pd


def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    assert 1 <= uuid <= 4
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")

    return data.to_numpy()
