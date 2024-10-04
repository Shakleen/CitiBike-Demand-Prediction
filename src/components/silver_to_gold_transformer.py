import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from typing import Tuple

from src.utils import read_delta, write_delta


@dataclass
class SilverToGoldTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    silver_delta_path: str = os.path.join(root_delta_path, "silver")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
