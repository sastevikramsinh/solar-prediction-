"""Data preprocessing utilities."""

import pandas as pd


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean solar dataset by filtering nighttime, outliers, and missing values.

    Args:
        df: Input merged dataframe.

    Returns:
        Tuple of cleaned dataframe and row-removal summary statistics.
    """
    working_df = df.copy()
    original_rows = len(working_df)

    working_df = working_df[working_df["IRRADIATION"] > 0.0]
    working_df = working_df[working_df["AC_POWER"] > 0.0]
    print(
        f"Rows after removing nighttime: {len(working_df)} "
        "(nighttime rows are expected to be ~50% of data)"
    )

    q1 = working_df["AC_POWER"].quantile(0.25)
    q3 = working_df["AC_POWER"].quantile(0.75)
    iqr = q3 - q1
    working_df = working_df[
        (working_df["AC_POWER"] >= q1 - 1.5 * iqr)
        & (working_df["AC_POWER"] <= q3 + 1.5 * iqr)
    ]

    working_df = working_df.ffill(limit=3)
    working_df = working_df.dropna()

    final_rows = len(working_df)
    stats = {
        "original_rows": int(original_rows),
        "final_rows": int(final_rows),
        "removed_rows": int(original_rows - final_rows),
    }

    return working_df, stats

