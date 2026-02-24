import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

from src.config import Config
from src.dataloader import DataLoader
from src.preprocessor import DataPreprocessor
from src.plotter import Plotter
from src.selection import ForwardSelection, BackwardSelection

from src.model.logistic import LogisticRegressionAnalysis
from src.model.l_discriminant import DiscriminantAnalysis
from src.model.bayes import NaiveBayesAnalysis
from src.model.comparison import RegressionComparison

import warnings

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titlesize": 12})


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loader = DataLoader(cfg)
        self.preprocessor = DataPreprocessor(cfg)
        self.plotter = Plotter(cfg.OUTPUT_DIR)
        self.lr = LogisticRegressionAnalysis(cfg)
        self.da = DiscriminantAnalysis()
        self.nb = NaiveBayesAnalysis()
        self.reg = RegressionComparison()
        self._results: dict = {}
        self._models: Dict[str, Tuple] = {}

    def run(self) -> None:
        """Execute the full analysis pipeline."""
        self._banner("FOREST COVER TYPE PREDICTION — CLASSIFICATION PROJECT")

        # Load & preprocess
        df = self.loader.load()
        X_tr, X_te, y_tr, y_te = self.preprocessor.prepare_multiclass(df)
        X_trb, X_teb, y_trb, y_teb = self.preprocessor.prepare_binary(df)
        X_trr, X_ter, y_trr, y_ter = self.preprocessor.prepare_regression(df)

        print(f"\nMulti-class split : train={X_tr.shape[0]},  test={X_te.shape[0]}")
        print(f"Binary split        : train={X_trb.shape[0]}, test={X_teb.shape[0]}")
        # print(f"Regression split    : train={X_trr.shape[0]}, test={X_ter.shape[0]}")

        # Correlation heatmap
        self.plotter.correlation_heatmap(
            df, self.cfg.CONTINUOUS_FEATURES, "correlation_heatmap.png"
        )

        # Task 1
        self._run_logistic_regression(
            df, X_tr, X_te, y_tr, y_te, X_trb, X_teb, y_trb, y_teb
        )

        # Task 2
        self._run_discriminant_analysis(
            X_tr, X_te, y_tr, y_te, X_trb, X_teb, y_trb, y_teb
        )

        # Task 3
        self._run_naive_bayes_and_comparison(X_tr, X_te, y_tr, y_te)

        # Task 4
        self._run_regression(X_trr, X_ter, y_trr, y_ter)

        # Summary
        self._print_summary()

    # TASK 1
    def _run_logistic_regression(
        self, df, X_tr, X_te, y_tr, y_te, X_trb, X_teb, y_trb, y_teb
    ):
        self._banner("SECTION 1: LOGISTIC REGRESSION")

        # 1a — Binary
        print("\n--- 1a) Binary Logistic Regression (Type 1 vs 2) ---")
        model_b, pred_b, acc_b = self.lr.fit_binary(X_trb, X_teb, y_trb, y_teb)
        self._results["lr_bin_acc"] = acc_b

        print(f"Accuracy: {acc_b:.4f}")
        print(classification_report(y_teb, pred_b, target_names=self.cfg.LABELS[:2]))

        # 1b — Coefficient statistics
        print("--- 1b) Coefficient Statistics ---")
        coef_tbl = self.lr.coefficient_statistics(model_b, X_trb)
        print(coef_tbl.to_string(index=False, float_format="%.4f"))

        coef_tbl.to_csv(
            os.path.join(self.cfg.OUTPUT_DIR, "logistic_coefficients.csv"), index=False
        )

        # 1c — Confounding
        print("\n--- 1c) Confounding Variable Analysis ---")
        mask_bin = df[self.cfg.TARGET].isin([1, 2])
        pairs, vif_df = self.lr.confounding_analysis(
            df.loc[mask_bin], self.cfg.CONTINUOUS_FEATURES
        )

        if pairs:
            print("Highly correlated pairs (|r| > 0.5):")
            for f1, f2, c in pairs:
                print(f"  {f1} <-> {f2}: r = {c}")
        else:
            print("  No pairs with |r| > 0.5.")

        print("\nVariance Inflation Factors:")
        print(vif_df.to_string(index=False, float_format="%.2f"))
        confounders = vif_df[vif_df["VIF"] > 5]["Feature"].tolist()

        if confounders:
            print(f"  Potential confounders (VIF>5): {confounders}")
        else:
            print("  No strong confounding (all VIF < 5)")

        # 1d — Multi-class
        print("\n--- 1d) Multi-class Logistic Regression (7 Classes) ---")

        model_m, pred_m, acc_m = self.lr.fit_multiclass(X_tr, X_te, y_tr, y_te)
        self._results["lr_multi_acc"] = acc_m
        self._models["Logistic Regression"] = (model_m, pred_m)

        print(f"Accuracy: {acc_m:.4f}")
        print(classification_report(y_te, pred_m, target_names=self.cfg.LABELS))

        self.plotter.confusion_matrix(
            y_te, pred_m, self.cfg.LABELS,
            "Logistic Regression — Confusion Matrix", "lr_confusion_matrix.png",
        )

        # Forward Selection
        # print("\n=== FORWARD SELECTION ===")
        # fwd = ForwardSelection(model_m, scoring="accuracy", cv=5)
        # fwd.fit(X_tr, y_tr, max_features=10)

        # print(f"\nBest features considering all class types: {fwd.get_best_features()}")

        # Backward Selection
        # print("\n=== BACKWARD SELECTION ===")
        # bwd = BackwardSelection(model_b, scoring="accuracy", cv=5)
        # bwd.fit(X_tr, y_tr, min_features=1)

        # print(f"\nBest features considering all class types: {bwd.get_best_features()}")

    #  TASK 2
    def _run_discriminant_analysis(
        self, X_tr, X_te, y_tr, y_te, X_trb, X_teb, y_trb, y_teb
    ):
        self._banner("SECTION 2: DISCRIMINANT ANALYSIS")

        # 2a — LDA
        print("\n--- 2a) Linear Discriminant Analysis ---")
        lda_m, pred_lda, acc_lda = self.da.fit_lda(X_tr, X_te, y_tr, y_te)
        self._results["lda_acc"] = acc_lda
        self._models["LDA"] = (lda_m, pred_lda)

        print(f"Accuracy: {acc_lda:.4f}")
        print(classification_report(y_te, pred_lda, target_names=self.cfg.LABELS))

        X_proj = lda_m.transform(X_te)
        self.plotter.lda_projection(X_proj, y_te, "lda_projection.png")

        # 2b — Threshold
        print("--- 2b) Effective Threshold ---")
        lda_bin = LinearDiscriminantAnalysis()
        lda_bin.fit(X_trb, y_trb)
        
        best_t, best_f1, threshs, f1s = self.da.find_best_threshold(
            lda_bin, X_teb, y_teb
        )
        print(f"Best threshold: {best_t:.2f}  (F1 = {best_f1:.4f})")
        self.plotter.threshold_sweep(threshs, f1s, best_t, "lda_threshold.png")

        # 2c — ROC
        print("--- 2c) ROC Curves ---")
        classes = sorted(y_te.unique())
        y_test_bin = label_binarize(y_te, classes=classes)
        y_prob_lda = lda_m.predict_proba(X_te)
        self.plotter.roc_curves(
            y_test_bin, y_prob_lda, self.cfg.LABELS,
            "ROC Curves — LDA (One-vs-Rest)", "roc_curve_lda.png",
        )

        # 2d — QDA
        print("--- 2d) Quadratic Discriminant Analysis ---")
        qda_m, pred_qda, acc_qda = self.da.fit_qda(X_tr, X_te, y_tr, y_te)
        self._results["qda_acc"] = acc_qda
        self._models["QDA"] = (qda_m, pred_qda)

        print(f"Accuracy: {acc_qda:.4f}")
        print(classification_report(y_te, pred_qda, target_names=self.cfg.LABELS))
        self.plotter.confusion_matrix(
            y_te, pred_qda, self.cfg.LABELS,
            "QDA — Confusion Matrix", "qda_confusion_matrix.png", cmap="Greens",
        )

    #  TASK 3
    def _run_naive_bayes_and_comparison(self, X_tr, X_te, y_tr, y_te):
        self._banner("SECTION 3: NAIVE BAYES & MODEL COMPARISON")

        nb_m, pred_nb, acc_nb = self.nb.fit(X_tr, X_te, y_tr, y_te)
        self._results["nb_acc"] = acc_nb
        self._models["Naive Bayes"] = (nb_m, pred_nb)
        print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
        print(classification_report(y_te, pred_nb, target_names=self.cfg.LABELS))

        # Comparison
        comp_df = self.nb.compare_models(self._models, y_te)
        print("\n--- Model Comparison ---")
        print(comp_df.to_string(index=False, float_format="%.4f"))
        comp_df.to_csv(
            os.path.join(self.cfg.OUTPUT_DIR, "model_comparison.csv"), index=False
        )
        self.plotter.model_comparison_bar(comp_df, "model_comparison.png")

        # ROC comparison
        classes = sorted(y_te.unique())
        y_test_bin = label_binarize(y_te, classes=classes)
        probs_dict = {n: m.predict_proba(X_te) for n, (m, _) in self._models.items()}
        self.plotter.roc_comparison(
            probs_dict, y_test_bin, classes, "roc_comparison_all.png"
        )

    #  TASK 4
    def _run_regression(self, X_tr, X_te, y_tr, y_te):
        self._banner("SECTION 4: LINEAR vs POISSON REGRESSION")

        lin = self.reg.fit_linear(X_tr, X_te, y_tr, y_te)
        poi = self.reg.fit_poisson(X_tr, X_te, y_tr, y_te)

        print(
            f"Linear  : MSE={lin['mse']:.2f}, MAE={lin['mae']:.2f}, "
            f"R²={lin['r2']:.4f}"
        )
        print(
            f"Poisson : MSE={poi['mse']:.2f}, MAE={poi['mae']:.2f}, "
            f"D²={poi['d2']:.4f}"
        )

        self._results.update(
            {
                "lin_mse": lin["mse"],
                "lin_r2": lin["r2"],
                "poi_mse": poi["mse"],
                "poi_d2": poi["d2"],
            }
        )

        comp = pd.DataFrame(
            {
                "Metric": ["MSE", "MAE", "R²/D²"],
                "Linear": [lin["mse"], lin["mae"], lin["r2"]],
                "Poisson": [poi["mse"], poi["mae"], poi["d2"]],
            }
        )
        print(f"\n{comp.to_string(index=False, float_format='%.4f')}")

        y_np = y_te.values if hasattr(y_te, "values") else y_te
        self.plotter.regression_residuals(
            y_np, lin["y_pred"], poi["y_pred"],
            {"mse": lin["mse"], "r2": lin["r2"]},
            {"mse": poi["mse"], "d2": poi["d2"]},
            "regression_residuals.png",
        )
        self.plotter.regression_actual_vs_pred(
            y_np, lin["y_pred"], poi["y_pred"], "regression_actual_vs_pred.png"
        )
        self.plotter.regression_distributions(
            y_np, lin["y_pred"], poi["y_pred"], "regression_distributions.png"
        )

    @staticmethod
    def _banner(text: str):
        print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")

    def _print_summary(self):
        self._banner("FINAL SUMMARY")
        r = self._results
        print(f"\nClassification (Multi-class, 7 Cover Types):")
        print(f"  Logistic Regression : {r.get('lr_multi_acc', 0):.4f}")
        print(f"  LDA                 : {r.get('lda_acc', 0):.4f}")
        print(f"  QDA                 : {r.get('qda_acc', 0):.4f}")
        print(f"  Naive Bayes         : {r.get('nb_acc', 0):.4f}")
        print(f"\nBinary LR (Type {self.cfg.LABELS[0]} vs {self.cfg.LABELS[1]}): {r.get('lr_bin_acc', 0):.4f}")
        print(f"\nRegression:")
        print(f"  Linear  : R²={r.get('lin_r2', 0):.4f}, MSE={r.get('lin_mse', 0):.1f}")
        print(f"  Poisson : D²={r.get('poi_d2', 0):.4f}, MSE={r.get('poi_mse', 0):.1f}")
        print(f"\nAll plots saved to {self.cfg.OUTPUT_DIR}/")
        print("=" * 70)

if __name__ == "__main__":
    pipeline = Pipeline(Config())
    pipeline.run()