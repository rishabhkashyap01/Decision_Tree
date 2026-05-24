import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.logistic_regression import LogisticRegression


def test_binary_classification():
    """Test binary logistic regression on breast cancer dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = lr.accuracy(y_test, preds)
    print(f"  Accuracy: {acc:.4f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_loss_decreases():
    """Verify that loss decreases during training."""
    X, y = make_classification(n_samples=300, n_features=5, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=300)
    lr.fit(X, y)

    assert lr.loss_history[-1] < lr.loss_history[0], "Loss should decrease"
    print(f"  Loss: {lr.loss_history[0]:.4f} → {lr.loss_history[-1]:.4f}")
    print("  ✓ Passed")


def test_l2_regularization():
    """Test L2-regularized logistic regression."""
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500, regularization="l2", alpha=0.1)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = lr.accuracy(y_test, preds)
    print(f"  Accuracy: {acc:.4f}")
    assert acc > 0.88, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_l1_regularization():
    """Test L1-regularized logistic regression (sparse weights)."""
    X, y = make_classification(n_samples=300, n_features=15, n_informative=4,
                                n_redundant=5, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500, regularization="l1", alpha=0.5)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = lr.accuracy(y_test, preds)
    n_near_zero = np.sum(np.abs(lr.weights) < 0.01)
    print(f"  Accuracy: {acc:.4f}  Near-zero weights: {n_near_zero}/{len(lr.weights)}")
    assert acc > 0.75, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_multiclass_ovr():
    """Test One-vs-Rest multi-class on Iris dataset."""
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500, multi_class="ovr")
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = lr.accuracy(y_test, preds)
    print(f"  Accuracy: {acc:.4f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_multiclass_softmax():
    """Test Softmax (multinomial) logistic regression on Iris dataset."""
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500, multi_class="softmax")
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = lr.accuracy(y_test, preds)
    print(f"  Accuracy: {acc:.4f}  Loss (final): {lr.loss_history[-1]:.4f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_metrics():
    """Test precision, recall, and F1 on a binary problem."""
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(learning_rate=0.1, n_iterations=500)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    p = lr.precision(y_test, preds)
    r = lr.recall(y_test, preds)
    f1 = lr.f1_score(y_test, preds)
    print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    assert f1 > 0.90, f"F1 too low: {f1}"
    print("  ✓ Passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Logistic Regression Tests")
    print("=" * 50)

    print("\n[1] Binary Classification (Breast Cancer):")
    test_binary_classification()

    print("\n[2] Loss Decreases:")
    test_loss_decreases()

    print("\n[3] L2 Regularization:")
    test_l2_regularization()

    print("\n[4] L1 Regularization:")
    test_l1_regularization()

    print("\n[5] Multi-class OvR (Iris):")
    test_multiclass_ovr()

    print("\n[6] Multi-class Softmax (Iris):")
    test_multiclass_softmax()

    print("\n[7] Metrics (Precision, Recall, F1):")
    test_metrics()

    print("\n✅ All logistic regression tests passed!")
