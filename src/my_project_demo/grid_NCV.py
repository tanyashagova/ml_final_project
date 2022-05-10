from pathlib import Path
import click
from multimethod import distance

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold

from .data import get_dataset
#from .pipeline import create_pipeline, create_pipeline_reg


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
def grid_ncv(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
) ->  None:
    features_train, features_val, target_train, target_val = get_dataset(
         dataset_path,
         random_state,
         test_split_ratio,
     )
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=6, shuffle=True, random_state=42)
    # enumerate splits
    outer_results = list()
    X = features_train.values
    y = target_train.values
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        # define the model
        model = KNeighborsClassifier(weights='distance', p=1)
        # define search space
        space = dict()
        space['n_neighbors'] = [3, 4, 5, 7]
        space['metric'] = ['minkowski', 'euclidean']
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, y_pred)
        # store the result
        outer_results.append(acc)
        # report progress
        click.echo('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        # summarize the estimated performance of the model
    click.echo(f'Accuracy: {np.mean(outer_results)}')
    click.echo(f'Trained model with best parameters')
    model = KNeighborsClassifier(n_neighbors=4, metric='minkowski', weights='distance', p=1)
    model.fit(features_train, target_train)
    pred_val = model.predict(features_val)
    accuracy = accuracy_score(target_val, pred_val)
    f1score = f1_score(target_val, pred_val, average='weighted')
    precision = precision_score(target_val, pred_val, average='weighted')
    click.echo(f'Accuracy: {accuracy}')
    click.echo(f'f1score: {f1score}')
    click.echo(f'precision: {precision}')


