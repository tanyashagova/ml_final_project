#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, n_neighbors:int, weights: str,  metric:str
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, metric=metric, p=1
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
