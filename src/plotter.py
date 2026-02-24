import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns
import numpy as np
import pandas as pd
import os

from typing import Dict

class Plotter:
    """Generates and saves every figure required by the project."""

    def __init__(self, output_dir: str):
        self.out = output_dir
        os.makedirs(self.out, exist_ok=True)

    def _save(self, name: str) -> str:
        path = os.path.join(self.out, name)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
        return path

    def confusion_matrix(
        self, y_true, y_pred, labels, title: str, fname: str, cmap: str = "Blues"
    ) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=labels, yticklabels=labels,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

        return self._save(fname)

    def roc_curves(
        self, y_test_bin, y_prob, classes, title: str, fname: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

        for i, (cls, col) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"Type {cls} (AUC = {auc(fpr, tpr):.3f})")
            
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)

        return self._save(fname)

    def roc_comparison(
        self, models_probs: Dict[str, np.ndarray],
        y_test_bin, classes, fname: str,
    ) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for idx, (name, y_prob) in enumerate(models_probs.items()):
            ax = axes[idx // 2, idx % 2]
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, lw=1.5,
                        label=f"Type {cls} (AUC={auc(fpr, tpr):.2f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.5)
            ax.set_title(name)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=7, loc="lower right")

        plt.suptitle("ROC Curves — All Models", fontsize=14, y=1.01)
        return self._save(fname)

    def threshold_sweep(
        self, thresholds, f1_scores, best_t: float, fname: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(thresholds, f1_scores, "b-o", markersize=4)
        ax.axvline(best_t, color="r", linestyle="--",
                   label=f"Best = {best_t:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("LDA: F1 Score vs Decision Threshold")
        ax.legend()

        return self._save(fname)

    def lda_projection(self, X_lda, y_test, fname: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 7))

        sc = ax.scatter(
            X_lda[:, 0], X_lda[:, 1], c=y_test, cmap="Set1", alpha=0.5, s=10
        )

        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA Projection — First 2 Discriminant Components")
        plt.colorbar(sc, label="Cover Type")

        return self._save(fname)

    def model_comparison_bar(self, comp_df: pd.DataFrame, fname: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(comp_df))
        w = 0.25

        ax.bar(x - w, comp_df["Accuracy"], w, label="Accuracy", color="#2196F3")
        ax.bar(x, comp_df["F1_Weighted"], w, label="F1 Weighted", color="#4CAF50")
        ax.bar(x + w, comp_df["F1_Macro"], w, label="F1 Macro", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["Model"], rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Model Comparison: LR vs LDA vs QDA vs Naive Bayes")
        ax.legend()

        for i, row in comp_df.iterrows():
            ax.text(
                i - w, row["Accuracy"] + 0.01,
                f"{row['Accuracy']:.3f}", ha="center", fontsize=8,
            )

        return self._save(fname)

    def regression_residuals(
        self, y_true, y_pred_lin, y_pred_poi,
        metrics_lin: dict, metrics_poi: dict, fname: str,
    ) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_pred_lin, y_true - y_pred_lin, alpha=0.3, s=5, color="#2196F3")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title(
            f"Linear Regression Residuals\n"
            f"MSE={metrics_lin['mse']:.1f}, R²={metrics_lin['r2']:.3f}"
        )

        axes[1].scatter(y_pred_poi, y_true - y_pred_poi, alpha=0.3, s=5, color="#4CAF50")
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title(
            f"Poisson Regression Residuals\n"
            f"MSE={metrics_poi['mse']:.1f}, D²={metrics_poi['d2']:.3f}"
        )

        return self._save(fname)

    def regression_actual_vs_pred(
        self, y_true, y_pred_lin, y_pred_poi, fname: str
    ) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, yp, c, t in [
            (axes[0], y_pred_lin, "#2196F3", "Linear"),
            (axes[1], y_pred_poi, "#4CAF50", "Poisson"),
        ]:
            ax.scatter(y_true, yp, alpha=0.3, s=5, color=c)
            ax.plot([0, y_true.max()], [0, y_true.max()], "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{t}: Actual vs Predicted")

        return self._save(fname)

    def regression_distributions(
        self, y_true, y_pred_lin, y_pred_poi, fname: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(y_true, bins=50, alpha=0.5, label="Actual", color="gray")
        ax.hist(y_pred_lin, bins=50, alpha=0.5, label="Linear Pred", color="#2196F3")
        ax.hist(y_pred_poi, bins=50, alpha=0.5, label="Poisson Pred", color="#4CAF50")
        ax.set_xlabel("Horizontal Distance to Hydrology")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution: Actual vs Predicted")
        ax.legend()

        return self._save(fname)

    def correlation_heatmap(
        self, df: pd.DataFrame, features: list, fname: str
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))

        corr = df[features].corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            xticklabels=[f.replace("_", "\n") for f in features],
            yticklabels=[f.replace("_", "\n") for f in features],
        )
        ax.set_title("Feature Correlation Matrix")

        return self._save(fname)