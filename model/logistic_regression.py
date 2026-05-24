import numpy as np


class LogisticRegression:
    """Logistic Regression via gradient descent with optional L1/L2 regularization
    and multi-class support (One-vs-Rest or Softmax)."""

    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization=None,
                 alpha=0.01, threshold=0.5, multi_class="binary"):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # None, "l1", or "l2"
        self.alpha = alpha                    # regularization strength
        self.threshold = threshold            # decision boundary for binary
        self.multi_class = multi_class        # "binary", "ovr", or "softmax"
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.classes_ = None
        self._ovr_classifiers = []
        self._class_to_idx = {}

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if self.multi_class == "ovr" and len(self.classes_) > 2:
            self._fit_ovr(X, y)
        elif self.multi_class == "softmax" and len(self.classes_) > 2:
            self._fit_softmax(X, y)
        else:
            self._fit_binary(X, y)

    def _fit_binary(self, X, y):
        """Binary logistic regression via gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            if self.regularization == "l2":
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == "l1":
                dw += (self.alpha / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

    def _fit_ovr(self, X, y):
        """One-vs-Rest multi-class logistic regression."""
        self._ovr_classifiers = []
        self.loss_history = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(float)
            clf = LogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                regularization=self.regularization,
                alpha=self.alpha,
                threshold=self.threshold,
                multi_class="binary"
            )
            clf.fit(X, y_binary)
            self._ovr_classifiers.append(clf)

        self.loss_history = np.mean(
            [clf.loss_history for clf in self._ovr_classifiers], axis=0
        ).tolist()

    def _fit_softmax(self, X, y):
        """Softmax (multinomial) logistic regression via gradient descent."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self._class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_idx[label] for label in y])
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_idx] = 1

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias          # (n_samples, n_classes)
            probs = self._softmax(z)                  # (n_samples, n_classes)

            error = probs - Y_onehot
            dw = (1 / n_samples) * (X.T @ error)     # (n_features, n_classes)
            db = (1 / n_samples) * np.sum(error, axis=0)

            if self.regularization == "l2":
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == "l1":
                dw += (self.alpha / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-15), axis=1))
            if self.regularization == "l2":
                loss += (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == "l1":
                loss += (self.alpha / n_samples) * np.sum(np.abs(self.weights))
            self.loss_history.append(loss)

    # ----------------------------
    # Predict
    # ----------------------------
    def predict_proba(self, X):
        """Return class probabilities."""
        if self.multi_class == "softmax" and self.weights is not None and self.weights.ndim == 2:
            return self._softmax(X @ self.weights + self.bias)
        elif self.multi_class == "ovr" and self._ovr_classifiers:
            probs = np.column_stack([clf.predict_proba(X) for clf in self._ovr_classifiers])
            return probs / probs.sum(axis=1, keepdims=True)
        else:
            return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        """Return predicted class labels."""
        if self.multi_class == "softmax" and self.weights is not None and self.weights.ndim == 2:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        elif self.multi_class == "ovr" and self._ovr_classifiers:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        else:
            return (self.predict_proba(X) >= self.threshold).astype(int)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, y, y_pred):
        """Binary cross-entropy loss with optional regularization penalty."""
        n = len(y)
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

        if self.regularization == "l2":
            loss += (self.alpha / (2 * n)) * np.sum(self.weights ** 2)
        elif self.regularization == "l1":
            loss += (self.alpha / n) * np.sum(np.abs(self.weights))

        return loss

    # ----------------------------
    # Metrics
    # ----------------------------
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def log_loss(self, y_true, y_pred_proba):
        return -np.mean(y_true * np.log(y_pred_proba + 1e-15) +
                        (1 - y_true) * np.log(1 - y_pred_proba + 1e-15))

    def precision(self, y_true, y_pred):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp + 1e-15)

    def recall(self, y_true, y_pred):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn + 1e-15)

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * p * r / (p + r + 1e-15)
