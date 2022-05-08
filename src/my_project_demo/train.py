from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import  cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
#from sklearn.metrics import log_loss

from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n_neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default='uniform',
    type=str,
    show_default=True,
)
@click.option(
    "--algorithm",
    default='auto',
    type=str,
    show_default=True,
)
@click.option(
    "--leaf_size",
    default=30,
    type=int,
    show_default=True,
)
@click.option(
    "--p",
    default=2,
    type=int,
    show_default=True,
)
@click.option(
    "--metric",
    default='minkowski',
    type=str,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    n_neighbors: int,
    weights: str,
    algorithm: str,
    leaf_size: int,
    p: int,
    metric: str
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, n_neighbors, weights, algorithm,  leaf_size, p, metric)
        #pipeline.fit(features_train, target_train)
        cross_validate(pipeline, features_train, target_train, cv=5)
        pred_val = pipeline.predict(features_val)
        accuracy = accuracy_score(target_val, pred_val)
        f1score = f1_score(target_val, pred_val, average='weighted')
        precision = precision_score(target_val, pred_val, average='weighted')
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("leaf_size", leaf_size)
        mlflow.log_param("p", p)
        mlflow.log_param("metric", metric)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1score", f1score)
        mlflow.log_metric("precision", precision)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"f1score: {f1score}.")
        click.echo(f"precision: {precision}")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")