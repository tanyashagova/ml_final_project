#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, n_neighbors:int, weights: str, algorithm: str,  leaf_size: int, p:int, metric: str
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                leaf_size=leaf_size, p=p, metric=metric
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
