from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather
from omegaconf import OmegaConf

from src.data.datatypes import Data
from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN


def get_code_system2code_counts(
    df: pd.DataFrame, code_systems: list[str]
) -> dict[str, dict[str, int]]:
    """

    Args:
        df (pd.DataFrame): The dataset in pandas dataframe format
        code_systems (list[str]): list of code systems to get counts for
    Returns:
        dict[str, dict[str, int]]: A dictionary with code systems as keys and a dictionary of code counts as values
    """
    code_system2code_counts = defaultdict(dict)
    for col in code_systems:
        # Flatten the list column and count occurrences
        all_codes = []
        for code_list in df[col]:
            # Skip None values
            if code_list is None:
                continue
            # Handle lists
            if isinstance(code_list, list):
                all_codes.extend(code_list)
            # Handle numpy arrays and other array-like objects (but not strings)
            elif hasattr(code_list, '__iter__') and not isinstance(code_list, str):
                try:
                    # Try to convert to list and extend
                    all_codes.extend(list(code_list))
                except (TypeError, ValueError):
                    # If conversion fails, skip or handle as single value
                    pass
            # Handle single values (skip NaN)
            else:
                try:
                    # Check if it's NaN using pd.isna (works for scalars)
                    if not pd.isna(code_list):
                        all_codes.append(code_list)
                except (TypeError, ValueError):
                    # If pd.isna fails, assume it's a valid value
                    all_codes.append(code_list)
        code_counts = Counter(all_codes)
        code_system2code_counts[col] = dict(code_counts)
    return code_system2code_counts


def data_pipeline(config: OmegaConf) -> Data:
    """The data pipeline.

    Args:
        config (OmegaConf): The config.

    Returns:
        Data: The data.
    """
    dir = Path(config.dir)
    # Read data from feather files
    df_table = pyarrow.feather.read_table(
        dir / config.data_filename,
        columns=[
            ID_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            "num_words",
            "num_targets",
        ]
        + config.code_column_names,
    )
    splits_table = pyarrow.feather.read_table(
        dir / config.split_filename,
    )
    
    # Convert to pandas for join operation
    df = df_table.to_pandas()
    splits = splits_table.to_pandas()
    
    # Join data with splits
    df = df.merge(splits, on=ID_COLUMN, how="inner")
    
    # Get code counts before filtering columns
    code_system2code_counts = get_code_system2code_counts(
        df, config.code_column_names
    )
    
    # Select only the required columns and convert back to arrow table
    df_selected = df[
        [
            ID_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            "split",
            "num_words",
            "num_targets",
        ]
    ]
    
    # Convert back to arrow table
    df_table = pa.Table.from_pandas(df_selected)
    
    # Define and cast to schema
    schema = pa.schema(
        [
            pa.field(ID_COLUMN, pa.int64()),
            pa.field(TEXT_COLUMN, pa.large_utf8()),
            pa.field(TARGET_COLUMN, pa.list_(pa.large_string())),
            pa.field("split", pa.large_string()),
            pa.field("num_words", pa.int64()),
            pa.field("num_targets", pa.int64()),
        ]
    )
    
    df_table = df_table.cast(schema)

    return Data(
        df_table,
        code_system2code_counts,
    )
