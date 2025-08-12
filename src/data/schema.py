"""
Data schema and basic checks (to be filled in Milestone 1).
- column presence
- dtypes (numeric vs categorical)
- missing values report
"""
from typing import Dict, List
import pandas as pd

def infer_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    cat = [c for c in df.columns if df[c].dtype == "object"]
    num = [c for c in df.columns if c not in cat]
    return {"numeric": num, "categorical": cat}
