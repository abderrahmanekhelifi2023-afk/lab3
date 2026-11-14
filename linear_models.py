import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Вычисляет сигмоиду без переполнения экспоненты."""
    probabilities = np.empty_like(scores, dtype=float)
    nonnegative = scores >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
    exponent = np.exp(scores[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    return probabilities


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучает регрессию пакетным спуском и сохраняет MSE после каждого шага."""
    features = np.asarray(data.train_features, dtype=float)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    test_features = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    loss_history = []
    errors = features @ weights + bias - target

    for _ in range(data.epochs):
        # Веса и смещение обновляются по ошибкам всей обучающей выборки.
        weights -= data.learning_rate * (2.0 / len(target)) * (features.T @ errors)
        bias -= data.learning_rate * 2.0 * errors.mean()
        errors = features @ weights + bias - target
        loss_history.append(float(np.mean(errors ** 2)))

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_features @ weights + bias,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучает бинарный классификатор пакетным спуском, сохраняя BCE по шагам."""
    features = np.asarray(data.train_features, dtype=float)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    test_features = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    loss_history = []
    scores = features @ weights + bias

    for _ in range(data.epochs):
        errors = _sigmoid(scores) - target
        weights -= data.learning_rate * (features.T @ errors) / len(target)
        bias -= data.learning_rate * errors.mean()
        scores = features @ weights + bias
        # BCE по логитам через softplus не требует log(0) или обрезки вероятностей.
        signed_scores = (2.0 * target - 1.0) * scores
        loss_history.append(float(np.mean(np.logaddexp(0.0, -signed_scores))))

    probabilities = _sigmoid(test_features @ weights + bias)
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=(probabilities >= 0.5).astype(int),
    )


def _classification_metrics(target, probabilities, predictions) -> ClassificationMetrics:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнивает готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    features = np.asarray(data.train_features, dtype=float)
    target = np.asarray(data.train_target, dtype=int).reshape(-1)
    test_features = np.asarray(data.test_features, dtype=float)
    test_target = np.asarray(data.test_target, dtype=int).reshape(-1)
    models = [
        LogisticRegression(max_iter=1000, random_state=42),
        DecisionTreeClassifier(random_state=42),
        KNeighborsClassifier(n_neighbors=min(5, len(target))),
    ]
    metrics = []
    for model in models:
        model.fit(features, target)
        positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
        probabilities = model.predict_proba(test_features)[:, positive_column]
        predictions = model.predict(test_features)
        metrics.append(_classification_metrics(test_target, probabilities, predictions))
    return BaselineComparisonResult(*metrics)


def run_experiments() -> dict:
    """Воспроизводит эксперименты на двух CSV и сохраняет требуемые метрики."""
    from dataclasses import asdict
    import json
    from pathlib import Path

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    folder = Path(__file__).resolve().parent
    students = pd.read_csv(folder / "Student_Performance.csv")
    student_features = students.drop(columns="Performance Index").copy()
    student_features["Extracurricular Activities"] = student_features[
        "Extracurricular Activities"
    ].map({"Yes": 1, "No": 0})
    train_x, test_x, train_y, test_y = train_test_split(
        student_features, students["Performance Index"], test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    regression_data = RegressionInput(
        scaler.fit_transform(train_x), train_y.to_numpy(),
        scaler.transform(test_x), test_y.to_numpy(),
    )
    regression = train_linear_regression(regression_data)

    german = pd.read_csv(folder / "german.csv", sep=";")
    train_x, test_x, train_y, test_y = train_test_split(
        german.iloc[:, 1:], german.iloc[:, 0], test_size=0.2,
        random_state=42, stratify=german.iloc[:, 0],
    )
    scaler = StandardScaler()
    classification_data = ClassificationInput(
        scaler.fit_transform(train_x), train_y.to_numpy(),
        scaler.transform(test_x), test_y.to_numpy(),
    )
    own_logistic = train_logistic_classifier(classification_data)
    baselines = compare_classical_models(classification_data)
    results = {
        "linear_regression": {
            "test_mse": float(np.mean((regression.predictions - regression_data.test_target) ** 2)),
        },
        "classification": {
            "own_logistic": asdict(_classification_metrics(
                classification_data.test_target,
                own_logistic.probabilities, own_logistic.predictions,
            )),
            **asdict(baselines),
        },
    }
    (folder / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    for name, model in [("MSE", regression), ("BCE", own_logistic)]:
        checkpoints = [1, 10, 100, len(model.loss_history)]
        print(name, {step: model.loss_history[step - 1] for step in checkpoints})
    return results


if __name__ == "__main__":
    run_experiments()
