import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

from src.config import Config
from src.dataloader import DataLoader
from src.preprocessor import DataPreprocessor

from src.model.logistic import LogisticRegressionAnalysis
from src.model.l_discriminant import DiscriminantAnalysis
from src.model.bayes import NaiveBayesAnalysis
from src.model.comparison import RegressionComparison

# Page Config & Custom Theme
st.set_page_config(
    page_title="Forest Cover Type Classification",
    layout="wide",
    page_icon="🌲",
    initial_sidebar_state="expanded",
)

# Cover type name mapping
COVER_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}
COVER_COLORS = ["#E74C3C", "#3498DB", "#9B59B6", "#F39C12", "#2ECC71", "#E91E8A", "#95A5A6"]

# Premium CSS
st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 1.5rem; }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1e293b;
        font-weight: 700;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #162d4a 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        font-weight: 500;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2b5e9e 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .section-header h3 { color: white !important; margin: 0; }
    .section-header p { color: #cbd5e1; margin: 4px 0 0 0; font-size: 0.9rem; }

    /* ── Insight cards ── */
    .insight-card {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.92rem;
    }
    .insight-card strong { color: #92400e; }

    .finding-card {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    .warning-card {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    /* ── Dataframes ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Separator ── */
    .section-sep {
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, transparent 100%);
        margin: 24px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions
def section_header(title, subtitle=None):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="section-header"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def insight_box(text):
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)


def finding_box(text):
    st.markdown(f'<div class="finding-card">{text}</div>', unsafe_allow_html=True)


def warning_box(text):
    st.markdown(f'<div class="warning-card">{text}</div>', unsafe_allow_html=True)


def separator():
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)


def styled_plot(fig):
    """Apply consistent styling to matplotlib figures."""
    for ax in fig.get_axes():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.xaxis.label.set_color("#475569")
        ax.yaxis.label.set_color("#475569")
        if ax.get_title():
            ax.title.set_color("#1e293b")
            ax.title.set_fontweight("bold")
    fig.patch.set_facecolor("white")
    return fig


# Data Loading & Splits
cfg = Config()


@st.cache_data
def load_data():
    loader = DataLoader(cfg)
    return loader.load()


df = load_data()
FEATS = cfg.CONTINUOUS_FEATURES

# ── Sidebar ──
st.sidebar.markdown("## 🌲 Forest Cover Type")
st.sidebar.markdown("**Classification & Regression**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Overview",
        "1️⃣ Logistic Regression",
        "2️⃣ Discriminant Analysis",
        "3️⃣ Naive Bayes & Comparison",
        "4️⃣ Linear vs Poisson Regression",
        "📈 Results Summary",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25, 0.05)
st.sidebar.caption(f"Train: {1 - test_size:.0%} | Test: {test_size:.0%}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<small style='color:#94a3b8'>Dataset: {len(df):,} samples<br>"
    f"Features: {len(FEATS)} continuous<br>"
    f"Classes: {df[cfg.TARGET].nunique()}</small>",
    unsafe_allow_html=True,
)


@st.cache_data
def prepare_splits(test_sz):
    # Multi-class
    scaler_m = StandardScaler()
    X_m = pd.DataFrame(scaler_m.fit_transform(df[FEATS]), columns=FEATS)
    y_m = df[cfg.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X_m, y_m, test_size=test_sz, random_state=42, stratify=y_m
    )

    # Binary (Type 1 vs 2)
    mask = y_m.isin([1, 2])
    X_b = pd.DataFrame(
        StandardScaler().fit_transform(df.loc[mask, FEATS]), columns=FEATS
    )
    y_b = (y_m[mask].reset_index(drop=True) == 1).astype(int)
    X_b = X_b.reset_index(drop=True)
    Xtrb, Xteb, ytrb, yteb = train_test_split(
        X_b, y_b, test_size=test_sz, random_state=42, stratify=y_b
    )

    return Xtr, Xte, ytr, yte, Xtrb, Xteb, ytrb, yteb


X_train, X_test, y_train, y_test, X_train_b, X_test_b, y_train_b, y_test_b = (
    prepare_splits(test_size)
)

# Instantiate analysis objects
lr_analysis = LogisticRegressionAnalysis(cfg)
da_analysis = DiscriminantAnalysis()
nb_analysis = NaiveBayesAnalysis()
reg_analysis = RegressionComparison()


# Cache model results to avoid redundant re-fitting
@st.cache_data
def run_binary_lr(_lr, Xtr, Xte, ytr, yte):
    model, pred, acc = _lr.fit_binary(Xtr, Xte, ytr, yte)
    coefs = _lr.coefficient_statistics(model, Xtr)
    return model, pred, acc, coefs


@st.cache_data
def run_multiclass_lr(_lr, Xtr, Xte, ytr, yte):
    return _lr.fit_multiclass(Xtr, Xte, ytr, yte)


@st.cache_data
def run_lda(_da, Xtr, Xte, ytr, yte):
    return _da.fit_lda(Xtr, Xte, ytr, yte)


@st.cache_data
def run_qda(_da, Xtr, Xte, ytr, yte):
    return _da.fit_qda(Xtr, Xte, ytr, yte)


@st.cache_data
def run_nb(_nb, Xtr, Xte, ytr, yte):
    return _nb.fit(Xtr, Xte, ytr, yte)


#  PAGE: DATA OVERVIEW
if page == "📊 Data Overview":
    section_header(
        "📊 Dataset Overview",
        "Forest Cover Type Prediction — Predicting forest cover from cartographic variables"
    )

    # ── Key stats ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Cover Types", f"{df[cfg.TARGET].nunique()}")
    c4.metric("Continuous Feats", f"{len(FEATS)}")

    separator()

    # ── Class distribution ──
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("#### Cover Type Distribution")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        counts = df[cfg.TARGET].value_counts().sort_index()
        bars = ax.bar(
            [COVER_NAMES.get(i, str(i)) for i in counts.index],
            counts.values,
            color=COVER_COLORS[: len(counts)],
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.005,
                f"{val:,}\n({val / len(df) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8, color="#475569", fontweight="bold",
            )
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Class Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        plt.xticks(rotation=25, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

    with col_r:
        st.markdown("#### Class Breakdown")
        dist_df = pd.DataFrame({
            "Cover Type": [COVER_NAMES.get(i, str(i)) for i in counts.index],
            "Count": counts.values,
            "Percentage": [f"{v / len(df) * 100:.1f}%" for v in counts.values],
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        insight_box(
            "<strong>Class Imbalance:</strong> Lodgepole Pine (49%) and Spruce/Fir (36%) "
            "dominate the dataset, comprising 85% of all samples. "
            "Cottonwood/Willow represents only 0.5%."
        )

    separator()

    # ── Correlation heatmap ──
    st.markdown("#### Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATS].corr()
    mask_tri = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask_tri, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(styled_plot(fig))

    separator()

    # ── Feature stats ──
    with st.expander("📋 Feature Statistics", expanded=False):
        st.dataframe(df[FEATS].describe().round(2).T, use_container_width=True)

    with st.expander("🔍 Sample Data (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)


#  PAGE: LOGISTIC REGRESSION
elif page == "1️⃣ Logistic Regression":
    section_header(
        "1️⃣ Logistic Regression",
        "Binary and multi-class classification with coefficient analysis"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔢 Binary LR", "📊 Coefficients", "🔗 Confounding", "🎯 Multi-class"]
    )

    # ── Tab 1: Binary ──
    with tab1:
        st.markdown("#### Binary Logistic Regression — Spruce/Fir vs Lodgepole Pine")

        model_b, pred_b, acc_b, coef_tbl = run_binary_lr(
            lr_analysis, X_train_b, X_test_b, y_train_b, y_test_b
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc_b:.4f}")
        report_dict = classification_report(
            y_test_b, pred_b,
            target_names=["Lodgepole Pine", "Spruce/Fir"],
            output_dict=True,
        )
        c2.metric("Spruce/Fir F1", f"{report_dict['Spruce/Fir']['f1-score']:.4f}")
        c3.metric("Lodgepole Pine F1", f"{report_dict['Lodgepole Pine']['f1-score']:.4f}")

        separator()

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("##### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test_b, pred_b)
            sns.heatmap(
                cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Lodgepole Pine", "Spruce/Fir"],
                yticklabels=["Lodgepole Pine", "Spruce/Fir"],
                linewidths=1, linecolor="white",
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title("Binary Classification", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(styled_plot(fig))

        with col_r:
            st.markdown("##### Classification Report")
            report_df = pd.DataFrame(report_dict).T.round(4)
            report_df = report_df.drop(["accuracy"], errors="ignore")
            st.dataframe(report_df, use_container_width=True)

    # ── Tab 2: Coefficients ──
    with tab2:
        st.markdown("#### Coefficient Statistics — Standard Error, Z-statistic, P-value")

        _, _, _, coef_tbl = run_binary_lr(
            lr_analysis, X_train_b, X_test_b, y_train_b, y_test_b
        )

        # Styled table
        def highlight_sig(row):
            if row.get("Significance") == "***":
                return ["background-color: #dcfce7"] * len(row)
            elif row.get("Significance") == "":
                return ["background-color: #fef9c3"] * len(row)
            return [""] * len(row)

        st.dataframe(
            coef_tbl.style
            .format({
                "Coefficient": "{:.4f}",
                "Std_Error": "{:.4f}",
                "Z_Statistic": "{:.2f}",
                "P_Value": "{:.6f}",
            })
            .apply(highlight_sig, axis=1),
            use_container_width=True,
        )

        separator()

        # Coefficient bar chart
        st.markdown("##### Coefficient Magnitudes")
        plot_df = coef_tbl[coef_tbl["Feature"] != "Intercept"].copy()
        plot_df = plot_df.sort_values("Coefficient")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#ef4444" if c < 0 else "#22c55e" for c in plot_df["Coefficient"]]
        ax.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coefficient Value", fontsize=10)
        ax.set_title("Feature Coefficients (Binary LR)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

        insight_box(
            "<strong>Elevation</strong> is by far the strongest predictor (coef = +1.73, Z = 285). "
            "Higher elevations strongly favor Spruce/Fir. "
            "<strong>Hillshade_9am</strong> is the only non-significant feature (p = 0.076)."
        )

    # ── Tab 3: Confounding ──
    with tab3:
        st.markdown("#### Confounding Variable Analysis")
        st.markdown("Examining multicollinearity via correlation pairs and Variance Inflation Factors (VIF).")

        mask_bin = df[cfg.TARGET].isin([1, 2])
        pairs, vif_df = lr_analysis.confounding_analysis(df.loc[mask_bin], FEATS)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Highly Correlated Pairs (|r| > 0.5)")
            if pairs:
                pair_data = []
                for f1, f2, c in pairs:
                    pair_data.append({"Feature 1": f1, "Feature 2": f2, "Correlation": c})
                st.dataframe(
                    pd.DataFrame(pair_data).style.format({"Correlation": "{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("No highly correlated pairs found.")

        with col_r:
            st.markdown("##### Variance Inflation Factors")
            vif_styled = vif_df.copy()
            vif_styled["Status"] = vif_styled["VIF"].apply(
                lambda v: "🔴 Severe" if v > 10 else ("🟡 Moderate" if v > 5 else "🟢 OK")
            )
            st.dataframe(
                vif_styled.style.format({"VIF": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )

        confounders = vif_df[vif_df["VIF"] > 5]["Feature"].tolist()
        if confounders:
            warning_box(
                f"<strong>Confounders Detected (VIF > 5):</strong> {', '.join(confounders)}. "
                "The Hillshade features exhibit severe multicollinearity (VIF > 100) due to "
                "overlapping geometric derivations from Aspect and Slope."
            )
        else:
            finding_box("All VIF values < 5 — no severe multicollinearity detected.")

    # ── Tab 4: Multi-class ──
    with tab4:
        st.markdown("#### Multi-class Logistic Regression — 7 Cover Types")

        model_m, pred_m, acc_m = run_multiclass_lr(
            lr_analysis, X_train, X_test, y_train, y_test
        )

        labels = sorted(y_test.unique())
        label_names = [COVER_NAMES.get(l, str(l)) for l in labels]
        report_m = classification_report(y_test, pred_m, target_names=label_names, output_dict=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc_m:.4f}")
        c2.metric("Weighted F1", f"{report_m['weighted avg']['f1-score']:.4f}")
        c3.metric("Macro F1", f"{report_m['macro avg']['f1-score']:.4f}")

        separator()

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("##### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, pred_m, labels=labels)
            sns.heatmap(
                cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor="white",
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title("7-Class Confusion Matrix", fontsize=12, fontweight="bold")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(styled_plot(fig))

        with col_r:
            st.markdown("##### Per-Class Performance")
            perf_data = []
            for name in label_names:
                r = report_m[name]
                perf_data.append({
                    "Class": name,
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "F1-Score": r["f1-score"],
                    "Support": int(r["support"]),
                })
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(
                perf_df.style.format({
                    "Precision": "{:.3f}", "Recall": "{:.3f}",
                    "F1-Score": "{:.3f}", "Support": "{:,}",
                }).background_gradient(subset=["F1-Score"], cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True, hide_index=True,
            )

            insight_box(
                "<strong>Aspen</strong> (F1 ≈ 0.00) is almost never correctly classified — "
                "samples are overwhelmingly misclassified as Lodgepole Pine. "
                "This reflects both class imbalance and feature overlap."
            )


#  PAGE: DISCRIMINANT ANALYSIS
elif page == "2️⃣ Discriminant Analysis":
    section_header(
        "2️⃣ Discriminant Analysis",
        "Linear (LDA) and Quadratic (QDA) Discriminant Analysis with threshold tuning"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📐 LDA", "🎚️ Threshold", "📈 ROC Curves", "📐 QDA"]
    )

    # ── Tab 1: LDA ──
    with tab1:
        st.markdown("#### Linear Discriminant Analysis — 7 Classes")

        lda_m, pred_lda, acc_lda = run_lda(da_analysis, X_train, X_test, y_train, y_test)

        labels = sorted(y_test.unique())
        label_names = [COVER_NAMES.get(l, str(l)) for l in labels]

        c1, c2, c3 = st.columns(3)
        c1.metric("LDA Accuracy", f"{acc_lda:.4f}")
        lda_report = classification_report(y_test, pred_lda, target_names=label_names, output_dict=True)
        c2.metric("Weighted F1", f"{lda_report['weighted avg']['f1-score']:.4f}")
        c3.metric("Macro F1", f"{lda_report['macro avg']['f1-score']:.4f}")

        separator()

        st.markdown("##### LDA Projection — First 2 Discriminant Components")
        X_proj = lda_m.transform(X_test)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cls in enumerate(labels):
            mask = y_test.values == cls
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                c=COVER_COLORS[i], label=COVER_NAMES.get(cls, str(cls)),
                alpha=0.4, s=8, edgecolors="none",
            )
        ax.set_xlabel("LD1", fontsize=11)
        ax.set_ylabel("LD2", fontsize=11)
        ax.set_title("LDA Projection", fontsize=14, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9, markerscale=3)
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

        finding_box(
            "Spruce/Fir and Lodgepole Pine overlap heavily in discriminant space, "
            "explaining the difficulty in distinguishing them. Ponderosa Pine and Krummholz "
            "form more distinct clusters."
        )

    # ── Tab 2: Threshold ──
    with tab2:
        st.markdown("#### Effective Threshold — Binary LDA (Spruce/Fir vs Lodgepole Pine)")

        lda_bin = LinearDiscriminantAnalysis()
        lda_bin.fit(X_train_b, y_train_b)
        best_t, best_f1, threshs, f1s = da_analysis.find_best_threshold(
            lda_bin, X_test_b, y_test_b
        )

        c1, c2 = st.columns(2)
        c1.metric("Optimal Threshold", f"{best_t:.2f}")
        c2.metric("Best F1 Score", f"{best_f1:.4f}")

        separator()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(threshs, f1s, "o-", color="#3b82f6", markersize=7, linewidth=2, markerfacecolor="white", markeredgewidth=2)
        ax.axvline(best_t, color="#ef4444", linestyle="--", linewidth=2, label=f"Optimal = {best_t:.2f}")
        ax.fill_between(threshs, f1s, alpha=0.08, color="#3b82f6")
        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title("F1 Score vs Decision Threshold", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

        insight_box(
            f"The optimal threshold ({best_t:.2f}) is below the default 0.50, "
            "indicating that slightly favoring Spruce/Fir predictions improves overall F1 balance."
        )

    # ── Tab 3: ROC ──
    with tab3:
        st.markdown("#### ROC Curves — LDA (One-vs-Rest)")

        lda_m2, _, _ = run_lda(da_analysis, X_train, X_test, y_train, y_test)
        classes = sorted(y_test.unique())
        y_tb = label_binarize(y_test, classes=classes)
        y_prob = lda_m2.predict_proba(X_test)

        fig, ax = plt.subplots(figsize=(9, 6))
        auc_scores = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_tb[:, i], y_prob[:, i])
            auc_val = auc(fpr, tpr)
            auc_scores[COVER_NAMES.get(cls, str(cls))] = auc_val
            ax.plot(fpr, tpr, lw=2, color=COVER_COLORS[i],
                    label=f"{COVER_NAMES.get(cls, str(cls))} (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves — LDA", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

        # AUC summary
        auc_df = pd.DataFrame({"Class": auc_scores.keys(), "AUC": auc_scores.values()})
        auc_df = auc_df.sort_values("AUC", ascending=False)
        st.dataframe(
            auc_df.style.format({"AUC": "{:.3f}"}).background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.7, vmax=1.0),
            use_container_width=True, hide_index=True,
        )

    # ── Tab 4: QDA ──
    with tab4:
        st.markdown("#### Quadratic Discriminant Analysis")

        qda_m, pred_qda, acc_qda = run_qda(da_analysis, X_train, X_test, y_train, y_test)

        labels = sorted(y_test.unique())
        label_names = [COVER_NAMES.get(l, str(l)) for l in labels]

        c1, c2, c3 = st.columns(3)
        c1.metric("QDA Accuracy", f"{acc_qda:.4f}")
        qda_report = classification_report(y_test, pred_qda, target_names=label_names, output_dict=True)
        c2.metric("Weighted F1", f"{qda_report['weighted avg']['f1-score']:.4f}")
        c3.metric("Macro F1", f"{qda_report['macro avg']['f1-score']:.4f}")

        separator()

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("##### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, pred_qda, labels=labels)
            sns.heatmap(
                cm, annot=True, fmt=",d", cmap="Greens", ax=ax,
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor="white",
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title("QDA Confusion Matrix", fontsize=12, fontweight="bold")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(styled_plot(fig))

        with col_r:
            st.markdown("##### LDA vs QDA — Minority Class Recall")
            lda_report_full = classification_report(
                y_test,
                run_lda(da_analysis, X_train, X_test, y_train, y_test)[1],
                target_names=label_names, output_dict=True,
            )
            comparison_data = []
            for name in label_names:
                comparison_data.append({
                    "Class": name,
                    "LDA Recall": lda_report_full[name]["recall"],
                    "QDA Recall": qda_report[name]["recall"],
                    "Δ Recall": qda_report[name]["recall"] - lda_report_full[name]["recall"],
                })
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comp_df.style.format({
                    "LDA Recall": "{:.3f}", "QDA Recall": "{:.3f}", "Δ Recall": "{:+.3f}"
                }).background_gradient(subset=["Δ Recall"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
                use_container_width=True, hide_index=True,
            )

            insight_box(
                "<strong>QDA significantly improves minority class recall</strong> — "
                "Cottonwood/Willow jumps from ~14% to ~60%, and Douglas-fir from ~25% to ~56%. "
                "This comes at the cost of lower overall accuracy."
            )


#  PAGE: NAIVE BAYES & COMPARISON
elif page == "3️⃣ Naive Bayes & Comparison":
    section_header(
        "3️⃣ Naive Bayes & Model Comparison",
        "Gaussian Naive Bayes and head-to-head comparison of all four classifiers"
    )

    nb_m, pred_nb, acc_nb = run_nb(nb_analysis, X_train, X_test, y_train, y_test)

    labels = sorted(y_test.unique())
    label_names = [COVER_NAMES.get(l, str(l)) for l in labels]

    # ── NB Results ──
    st.markdown("#### Gaussian Naive Bayes Results")
    nb_report = classification_report(y_test, pred_nb, target_names=label_names, output_dict=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc_nb:.4f}")
    c2.metric("Weighted F1", f"{nb_report['weighted avg']['f1-score']:.4f}")
    c3.metric("Macro F1", f"{nb_report['macro avg']['f1-score']:.4f}")

    with st.expander("📋 Full Classification Report"):
        st.text(classification_report(y_test, pred_nb, target_names=label_names))

    separator()

    # ── Full comparison ──
    st.markdown("#### Model Comparison — All Four Classifiers")

    models_dict = {
        "Logistic Regression": run_multiclass_lr(lr_analysis, X_train, X_test, y_train, y_test),
        "LDA": run_lda(da_analysis, X_train, X_test, y_train, y_test),
        "QDA": run_qda(da_analysis, X_train, X_test, y_train, y_test),
        "Naive Bayes": (nb_m, pred_nb, acc_nb),
    }
    comp_input = {n: (m, p) for n, (m, p, _) in models_dict.items()}
    comp_df = nb_analysis.compare_models(comp_input, y_test)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comp_df))
        w = 0.22
        bars1 = ax.bar(x - w, comp_df["Accuracy"], w, label="Accuracy", color="#3b82f6", edgecolor="white")
        bars2 = ax.bar(x, comp_df["F1_Weighted"], w, label="F1 Weighted", color="#22c55e", edgecolor="white")
        bars3 = ax.bar(x + w, comp_df["F1_Macro"], w, label="F1 Macro", color="#f59e0b", edgecolor="white")

        # Value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color="#475569", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["Model"], fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        st.pyplot(styled_plot(fig))

    with col_r:
        st.markdown("##### Metrics Table")
        st.dataframe(
            comp_df.style
            .format({c: "{:.4f}" for c in ["Accuracy", "F1_Weighted", "F1_Macro"]})
            .highlight_max(axis=0, subset=["Accuracy", "F1_Weighted", "F1_Macro"], color="#dcfce7")
            .highlight_min(axis=0, subset=["Accuracy", "F1_Weighted", "F1_Macro"], color="#fef2f2"),
            use_container_width=True, hide_index=True,
        )

        insight_box(
            "<strong>Key Trade-off:</strong> Logistic Regression wins on overall accuracy "
            "(best for majority-class predictions), while QDA achieves the highest macro F1 "
            "(best for balanced performance across all classes)."
        )

    separator()

    # ── ROC comparison ──
    st.markdown("#### ROC Curves — All Models")
    classes = sorted(y_test.unique())
    y_tb = label_binarize(y_test, classes=classes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_names = list(models_dict.keys())

    for idx, (name, (model, pred, _)) in enumerate(models_dict.items()):
        ax = axes[idx // 2][idx % 2]
        try:
            y_prob = model.predict_proba(X_test)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_tb[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, lw=1.5, color=COVER_COLORS[i],
                        label=f"Type {cls} (AUC={auc(fpr, tpr):.2f})")
        except Exception:
            ax.text(0.5, 0.5, "Probabilities\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.grid(alpha=0.15)

    fig.suptitle("ROC Curves — All Models", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(styled_plot(fig))


#  PAGE: REGRESSION
elif page == "4️⃣ Linear vs Poisson Regression":
    section_header(
        "4️⃣ Linear vs Poisson Regression",
        "Comparing linear and Poisson models for predicting Horizontal Distance to Hydrology"
    )

    target = cfg.REGRESSION_TARGET
    feats_r = [f for f in FEATS if f != target]
    Xr = pd.DataFrame(StandardScaler().fit_transform(df[feats_r]), columns=feats_r)
    yr = df[target]
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        Xr, yr, test_size=test_size, random_state=42
    )

    lin = reg_analysis.fit_linear(Xtr_r, Xte_r, ytr_r, yte_r)
    poi = reg_analysis.fit_poisson(Xtr_r, Xte_r, ytr_r, yte_r)

    # ── Metric cards ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Linear R²", f"{lin['r2']:.4f}")
    c2.metric("Linear MSE", f"{lin['mse']:,.0f}")
    c3.metric("Linear MAE", f"{lin['mae']:.1f}")
    c4.metric("Poisson D²", f"{poi['d2']:.4f}")
    c5.metric("Poisson MSE", f"{poi['mse']:,.0f}")
    c6.metric("Poisson MAE", f"{poi['mae']:.1f}")

    winner = "Linear" if lin['r2'] > poi['d2'] else "Poisson"
    finding_box(
        f"<strong>{winner} Regression</strong> outperforms on all metrics. "
        f"R² = {lin['r2']:.4f} vs D² = {poi['d2']:.4f} (higher is better), "
        f"MSE = {lin['mse']:,.0f} vs {poi['mse']:,.0f} (lower is better)."
    )

    separator()

    # ── Residuals ──
    st.markdown("#### Residual Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (data, name, color) in enumerate([(lin, "Linear", "#3b82f6"), (poi, "Poisson", "#22c55e")]):
        ax = axes[i]
        residuals = yte_r.values - data["y_pred"]
        ax.scatter(data["y_pred"], residuals, alpha=0.15, s=4, color=color, edgecolors="none")
        ax.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
        metric_label = f"R²={lin['r2']:.3f}" if i == 0 else f"D²={poi['d2']:.3f}"
        ax.set_title(f"{name} Regression Residuals\nMSE={data['mse']:,.0f}, {metric_label}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
    plt.tight_layout()
    st.pyplot(styled_plot(fig))

    separator()

    # ── Actual vs Predicted ──
    st.markdown("#### Actual vs Predicted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (data, name, color) in enumerate([(lin, "Linear", "#3b82f6"), (poi, "Poisson", "#22c55e")]):
        ax = axes[i]
        ax.scatter(yte_r, data["y_pred"], alpha=0.15, s=4, color=color, edgecolors="none")
        max_val = max(yte_r.max(), np.max(data["y_pred"]))
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, alpha=0.7)
        ax.set_title(f"{name}: Actual vs Predicted", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
    plt.tight_layout()
    st.pyplot(styled_plot(fig))

    separator()

    # ── Distribution comparison ──
    st.markdown("#### Prediction Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(yte_r, bins=80, alpha=0.4, label="Actual", color="#94a3b8", edgecolor="white")
    ax.hist(lin["y_pred"], bins=80, alpha=0.5, label="Linear Pred", color="#3b82f6", edgecolor="white")
    ax.hist(poi["y_pred"], bins=80, alpha=0.5, label="Poisson Pred", color="#22c55e", edgecolor="white")
    ax.set_xlabel("Horizontal Distance to Hydrology", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution: Actual vs Predicted", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-200, 1500)
    plt.tight_layout()
    st.pyplot(styled_plot(fig))

    insight_box(
        "The Poisson model concentrates predictions in a narrow range, failing to capture the "
        "actual distribution's spread. The Linear model better approximates the shape but "
        "generates some negative predictions, which are physically impossible for a distance variable."
    )


#  PAGE: RESULTS SUMMARY
elif page == "📈 Results Summary":
    section_header("📈 Results Summary", "Comprehensive overview of all models and key findings")

    # ── Classification metrics ──
    st.markdown("#### Classification Results — 7 Cover Types")
    models_all = {
        "Logistic Regression": run_multiclass_lr(lr_analysis, X_train, X_test, y_train, y_test),
        "LDA": run_lda(da_analysis, X_train, X_test, y_train, y_test),
        "QDA": run_qda(da_analysis, X_train, X_test, y_train, y_test),
        "Naive Bayes": run_nb(nb_analysis, X_train, X_test, y_train, y_test),
    }

    c1, c2, c3, c4 = st.columns(4)
    accs = {}
    for (name, (_, pred, acc)), col in zip(models_all.items(), [c1, c2, c3, c4]):
        accs[name] = acc
        col.metric(name, f"{acc:.4f}")

    best = max(accs, key=accs.get)
    st.success(f"🏆 **Best Overall Accuracy: {best}** — {accs[best]:.4f}")

    separator()

    # ── Key Findings ──
    st.markdown("#### Key Findings")

    col_l, col_r = st.columns(2)
    with col_l:
        finding_box(
            "<strong>🎯 Best Accuracy:</strong> Logistic Regression achieves the highest overall accuracy, "
            "confirming that linear decision boundaries work well for the dominant classes."
        )
        finding_box(
            "<strong>⚖️ Best Balance:</strong> QDA achieves the highest macro F1 score, "
            "providing significantly better recall on minority classes (Cottonwood/Willow, Aspen, Douglas-fir)."
        )
        finding_box(
            "<strong>📊 Elevation Dominance:</strong> Elevation is the most significant predictor "
            "across all models (Z = 285 in binary LR), with higher elevations favoring Spruce/Fir."
        )

    with col_r:
        warning_box(
            "<strong>🔗 Multicollinearity:</strong> Hillshade features show severe confounding "
            "(VIF > 100) due to shared geometric derivation from Aspect and Slope."
        )
        finding_box(
            "<strong>📈 Regression:</strong> Linear Regression outperforms Poisson "
            "(R² = 0.447 vs D² = 0.364) for predicting Horizontal Distance to Hydrology. "
            "Neither model fully captures the right-skewed distribution."
        )
        warning_box(
            "<strong>⚠️ Class Imbalance:</strong> The dataset is heavily imbalanced — "
            "Aspen and Cottonwood/Willow are nearly invisible to most models, "
            "with Aspen achieving 0% recall under Logistic Regression and LDA."
        )

    separator()

    # ── Methodology ──
    st.markdown("#### Methodology")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""
        **Dataset:** Covertype ({len(df):,} samples, {df[cfg.TARGET].nunique()} cover types, {len(FEATS)} continuous features)

        **Preprocessing:** StandardScaler normalization on continuous features, stratified train/test split ({1-test_size:.0%}/{test_size:.0%})

        **Classification Models:** Logistic Regression, LDA, QDA, Gaussian Naive Bayes
        """)
    with col_r:
        st.markdown(f"""
        **Evaluation Metrics:** Accuracy, F1 (weighted & macro), ROC-AUC, confusion matrices

        **Regression Models:** Ordinary Least Squares vs. Poisson GLM

        **Target (Regression):** Horizontal Distance to Hydrology
        """)