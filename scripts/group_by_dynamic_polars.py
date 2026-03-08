import polars as pl
from polars import *
from polars._typing import *
from datetime import timedelta

def str_to_col(column_name: str) -> IntoExpr:
    return col(column_name)

def group_by_dynamic_right_aligned(
        df: DataFrame,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        include_boundaries: bool = False,
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        check_sorted: bool = True,
        include_windows_ending_after_last_index: bool = False
):
    """
    Wrapper for polars group_by_dynamic that allows the rolling operation to look backwards instead of the standard forward behaviour.
    Source: https://github.com/pola-rs/polars/issues/6669#issuecomment-1825605731.
    """
    
    # First pass at the groups, with no offset
    labels_series = (df
      .group_by_dynamic(index_column, every=every, period=period, include_boundaries=include_boundaries, by=by, closed='right', label='right', start_by='window')
      .agg()
      .select(index_column)
      .to_series(0)
    )

    max_date = df.select(index_column).to_series(0).max()
    end_of_first_window_extending_beyond_data=labels_series.filter(labels_series >= max_date)[0]
    # The negative offset to shift windows by such that the last window ends exactly on max_date
    offset = max_date - end_of_first_window_extending_beyond_data

    # Redo the group_by_dynamic with the offset, so we get a window ending exactly on max_date
    groups = df.group_by_dynamic(index_column, every=every, period=period, by=by, closed='right', label='right', start_by='window', offset=offset)

    if include_windows_ending_after_last_index:
        return groups
    
    # Monkey patch the agg function to filter out the groups that extend beyond the last date, which contain only subsets of the last full window
    def wrapped_agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> DataFrame:
        return groups.__class__.agg(self, *aggs, **named_aggs).filter(str_to_col(index_column) <= max_date)
    
    groups.agg = wrapped_agg.__get__(groups, groups.__class__)
    return groups