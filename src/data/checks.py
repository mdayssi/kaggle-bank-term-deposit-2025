from typing import Dict, List

import numpy as np
import pandas as pd


def get_features_names(df: pd.DataFrame) -> Dict[str, List[str]]:
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return {'categorical': cat_features, 'numeric': num_features}






