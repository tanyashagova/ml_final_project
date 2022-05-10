from pathlib import Path
from joblib import dump

import numpy as np
import click
import mlflow
import mlflow.sklearn
from scipy.__config__ import show
from sklearn.model_selection import  cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

from .data import get_dataset
from .pipeline import create_pipeline, create_pipeline_reg


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
    "--model-type",
    default='KNN',
    type=str,
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
    default='distance',
    type=str,
    show_default=True,
)
@click.option(
    "--metric",
    default='minkowski',
    type=str,
    show_default=True,
)
@click.option(
    "--max_iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logregc",
    default=1.0,
    type=float,
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
    metric: str,
    logregc: float, 
    max_iter: int,
    model_type: str,  
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if model_type == 'KNN':
            pipeline = create_pipeline(use_scaler, n_neighbors, weights, metric)
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("weights", weights)
            mlflow.log_param("metric", metric)
        elif model_type == 'LogReg':
            pipeline = create_pipeline_reg(use_scaler, max_iter, random_state, logregc)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_C", logregc)
            mlflow.log_param("random_state", random_state)
        #pipeline.fit(features_train, target_train)
        feature_eng = 2

        if feature_eng == 0:
            cv_results = cross_validate(pipeline, features_train, target_train, 
                            cv=5,
                            scoring=('accuracy', 'f1_weighted', 'precision_weighted')
                            )
        elif feature_eng == 1:
            selection_model = RandomForestClassifier(random_state=42)
            pipe_selection = make_pipeline(SelectFromModel(selection_model), pipeline)
            cv_results = cross_validate(pipe_selection, features_train, target_train, 
                            cv=5,
                            scoring=('accuracy', 'f1_weighted', 'precision_weighted')
                            )
        elif feature_eng == 2:
            train_tranc = PCA(n_components=35, random_state=42).fit_transform(features_train)
            click.echo(f"tranc shape: {train_tranc.shape}.")
            cv_results = cross_validate(pipeline, train_tranc, target_train, 
                            cv=5,
                            scoring=('accuracy', 'f1_weighted', 'precision_weighted')
                            )        
        
        
        accuracy = np.mean(cv_results['test_accuracy'])
        f1score = np.mean(cv_results['test_f1_weighted'])
        precision = np.mean(cv_results['test_precision_weighted'])
        mlflow.log_param("feat_engeen", feature_eng)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1score", f1score)
        mlflow.log_metric("precision", precision)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"f1score: {f1score}.")
        click.echo(f"precision: {precision}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
