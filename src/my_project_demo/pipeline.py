#from sklearn.linear_model import LogisticRegression
from distutils.log import Log
from scipy import rand
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, n_neighbors:int, weights: str,  metric:str,
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

def create_pipline_reg(
    use_scaler:bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                max_iter=max_iter, logreg_C=logreg_C, random_state=random_state
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)