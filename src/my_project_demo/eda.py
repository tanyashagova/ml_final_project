from pandas_profiling import ProfileReport
from pathlib import Path
from typing import Tuple

import click
import pandas as pd


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def get_report(
    dataset_path: Path
    )-> None:
    dataset = pd.read_csv(dataset_path).set_index('Id')
    profile = ProfileReport(dataset, title="Pandas Profiling Report")
    profile.to_file("eda_report.html")   
    #return profile
