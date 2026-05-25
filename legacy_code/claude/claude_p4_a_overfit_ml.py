"""
CRT Lead Detection Project — Step 4: Overfit ML Experiment
===========================================================
PURPOSE OF THIS SCRIPT
-----------------------
Before building a "real" model, we deliberately overfit — train and
evaluate on the same data — to answer one question:

    "Can a model at all learn the mapping from RV lead position
     to LV lead position using this dataset?"

If the training error is near-zero, the data pipeline is correct and
the signal exists. If it's high, something upstream is wrong.

WHAT THIS SCRIPT DOES
----------------------
1. Loads normalized_results.json (86 patients, Step 3 output)
2. Builds a feature matrix X and target matrix y:
     X (inputs)  = RV lead positions + heart axis length
                   [RL1_t, RL1_ang, RL2_t, RL2_ang, axis_len_mm]
     y (targets) = LV lead positions (what we want to predict)
                   [LL1_t, LL1_ang, LL2_t, LL2_ang,
                    LL3_t, LL3_ang, LL4_t, LL4_ang]
3. Trains three models of increasing complexity:
     - Linear Regression  (baseline, low capacity)
     - Random Forest      (medium capacity, good for tabular data)
     - MLP Neural Net     (high capacity, should fully overfit)
4. Reports training error (overfit score) AND a leave-one-out
   cross-validation score (first honest peek at generalisation).
5. Saves all results + a feature importance plot (Random Forest).

Requirements:
    pip install scikit-learn matplotlib

Usage:
    python claude_p4_overfit_ml.py
"""

import json
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works everywhere
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import Ridge
from sklearn.ensemble       import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput    import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")   # suppress sklearn convergence noise


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURE YOUR PATHS
# ─────────────────────────────────────────────────────────────────────────────

NORMALIZED_JSON = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\normalized_results.json")
OUTPUT_DIR      = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")

OUTPUT_JSON     = OUTPUT_DIR / "ml_results.json"
OUTPUT_REPORT   = OUTPUT_DIR / "ml_report.txt"
OUTPUT_PLOT     = OUTPUT_DIR / "ml_overfit_plot.png"


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE / TARGET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Inputs: things known before/independently of LV lead placement
FEATURE_NAMES = [
    "RL1_t",       # RV distal lead — longitudinal position
    "RL1_ang",     # RV distal lead — radial angle
    "RL2_t",       # RV proximal lead — longitudinal position
    "RL2_ang",     # RV proximal lead — radial angle
    "axis_len_mm", # total heart axis length (patient size proxy)
]

# Targets: LV lead positions (what we want to eventually predict)
TARGET_NAMES = [
    "LL1_t",   "LL1_ang",
    "LL2_t",   "LL2_ang",
    "LL3_t",   "LL3_ang",
    "LL4_t",   "LL4_ang",
]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(json_path: Path):
    """
    Parse normalized_results.json into numpy arrays X, y
    and return also the patient IDs for reference.

    Skips patients that are missing any required feature or target.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    X_rows, y_rows, pids = [], [], []

    for pid, rec in sorted(data.items()):
        electrodes   = rec.get("electrodes", {})
        axis_len_mm  = rec.get("axis_len_mm", None)

        # ── build feature row ─────────────────────────────────────────────────
        row_x = []
        ok = True
        for feat in FEATURE_NAMES:
            if feat == "axis_len_mm":
                if axis_len_mm is None:
                    ok = False; break
                row_x.append(axis_len_mm)
            else:
                lead, attr = feat.rsplit("_", 1)       # e.g. "RL1", "t"
                e = electrodes.get(lead)
                if e is None:
                    ok = False; break
                key = "longitudinal_t" if attr == "t" else "radial_angle_deg"
                row_x.append(e[key])
        if not ok:
            print(f"  [{pid}] skipped — missing feature(s)")
            continue

        # ── build target row ──────────────────────────────────────────────────
        row_y = []
        for targ in TARGET_NAMES:
            lead, attr = targ.rsplit("_", 1)
            e = electrodes.get(lead)
            if e is None:
                ok = False; break
            key = "longitudinal_t" if attr == "t" else "radial_angle_deg"
            row_y.append(e[key])
        if not ok:
            print(f"  [{pid}] skipped — missing target(s)")
            continue

        X_rows.append(row_x)
        y_rows.append(row_y)
        pids.append(pid)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y, pids


# ─────────────────────────────────────────────────────────────────────────────
#  ANGULAR MEAN ABSOLUTE ERROR
# ─────────────────────────────────────────────────────────────────────────────

def angular_mae(y_true_ang: np.ndarray, y_pred_ang: np.ndarray) -> float:
    """
    Circular mean absolute error for angle columns (in degrees).
    Handles the 0°/360° wraparound — e.g. error between 5° and 355° is 10°,
    not 350°.
    """
    diff = np.abs(y_true_ang - y_pred_ang) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    return float(diff.mean())


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                      target_names: list) -> dict:
    """
    Compute per-target and aggregate MAE.
    Angle columns use circular MAE; longitudinal columns use plain MAE.
    """
    results = {}
    t_maes, ang_maes = [], []

    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if name.endswith("_ang"):
            mae = angular_mae(yt, yp)
            ang_maes.append(mae)
        else:
            mae = mean_absolute_error(yt, yp)
            t_maes.append(mae)
        results[name] = round(mae, 4)

    results["_mean_t_mae"]   = round(float(np.mean(t_maes)),   4)
    results["_mean_ang_mae"] = round(float(np.mean(ang_maes)), 4)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_models():
    """
    Return a dict of model_name → sklearn Pipeline.
    All pipelines include StandardScaler so features are normalised
    before fitting — important for MLP and Ridge.
    """
    return {
        # Baseline: linear — low capacity, won't overfit much
        "LinearRidge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=0.01)),
        ]),

        # Medium capacity: tree ensemble — handles non-linearity well
        # n_estimators=500, min_samples_leaf=1 → will memorise training data
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        # High capacity: MLP — with enough hidden units will fully overfit
        "MLP_Neural": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=5000,
                random_state=42,
                learning_rate_init=1e-3,
                early_stopping=False,   # no early stopping → full overfit
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LEAVE-ONE-OUT CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_loo(model_pipeline, X, y, target_names):
    """
    Leave-One-Out CV: train on n-1 patients, predict the held-out one.
    Gives an honest estimate of generalisation with small datasets.

    Returns per-target MAE dict (same format as score_predictions).
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        # clone and refit each fold (LeaveOneOut doesn't share state)
        from sklearn.base import clone
        pipe = clone(model_pipeline)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_true_all.append(y[test_idx[0]])
        y_pred_all.append(pred[0])

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    return score_predictions(y_true_arr, y_pred_arr, target_names)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(all_model_results: dict, X: np.ndarray, y: np.ndarray,
              pids: list, rf_pipeline):
    """
    4-panel figure:
      [0,0] Train vs LOO MAE bar chart (longitudinal t)
      [0,1] Train vs LOO MAE bar chart (angle)
      [1,0] Predicted vs actual scatter for LL1_t (RF, train set)
      [1,1] Feature importances (Random Forest)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Step 4 — Overfit ML Experiment\nCRT Lead Position Prediction",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    model_names = list(all_model_results.keys())
    colors = {"LinearRidge": "#4e79a7", "RandomForest": "#f28e2b",
              "MLP_Neural": "#e15759"}

    # ── [0,0] Longitudinal t MAE ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(model_names))
    width = 0.35
    train_t = [all_model_results[m]["train"]["_mean_t_mae"] for m in model_names]
    loo_t   = [all_model_results[m]["loo"]["_mean_t_mae"]   for m in model_names]
    bars1 = ax0.bar(x_pos - width/2, train_t, width, label="Train (overfit)",
                    color=[colors[m] for m in model_names], alpha=0.9)
    bars2 = ax0.bar(x_pos + width/2, loo_t,   width, label="LOO (honest)",
                    color=[colors[m] for m in model_names], alpha=0.45,
                    edgecolor="black", linewidth=0.8)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax0.set_ylabel("Mean Absolute Error  (t units)")
    ax0.set_title("Longitudinal Position Error\n(t: 0=Apex, 1=Base)")
    ax0.legend(fontsize=8)
    ax0.axhline(0.05, color="green", linestyle="--", linewidth=0.8,
                label="5% axis goal")
    for bar in bars1:
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # ── [0,1] Angle MAE ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    train_ang = [all_model_results[m]["train"]["_mean_ang_mae"] for m in model_names]
    loo_ang   = [all_model_results[m]["loo"]["_mean_ang_mae"]   for m in model_names]
    ax1.bar(x_pos - width/2, train_ang, width, label="Train (overfit)",
            color=[colors[m] for m in model_names], alpha=0.9)
    ax1.bar(x_pos + width/2, loo_ang,   width, label="LOO (honest)",
            color=[colors[m] for m in model_names], alpha=0.45,
            edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Absolute Error  (degrees)")
    ax1.set_title("Radial Angle Error\n(0°=Anterior, clockwise)")
    ax1.legend(fontsize=8)
    ax1.axhline(10, color="green", linestyle="--", linewidth=0.8,
                label="10° goal")
    for bar in ax1.patches[:len(model_names)]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}°", ha="center", va="bottom", fontsize=7)

    # ── [1,0] Predicted vs Actual scatter — LL1_t, Random Forest ─────────────
    ax2 = fig.add_subplot(gs[1, 0])
    rf_pipe = rf_pipeline
    rf_pipe.fit(X, y)
    y_pred_train = rf_pipe.predict(X)
    ll1_t_idx = TARGET_NAMES.index("LL1_t")
    ax2.scatter(y[:, ll1_t_idx], y_pred_train[:, ll1_t_idx],
                alpha=0.75, edgecolors="k", linewidths=0.4,
                color="#f28e2b", s=60)
    lo = min(y[:, ll1_t_idx].min(), y_pred_train[:, ll1_t_idx].min()) - 0.05
    hi = max(y[:, ll1_t_idx].max(), y_pred_train[:, ll1_t_idx].max()) + 0.05
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax2.set_xlabel("Actual  LL1_t")
    ax2.set_ylabel("Predicted  LL1_t")
    ax2.set_title("Predicted vs Actual — LL1 Longitudinal\n(Random Forest, train set)")
    ax2.legend(fontsize=8)
    mae_val = all_model_results["RandomForest"]["train"]["LL1_t"]
    ax2.text(0.05, 0.92, f"Train MAE = {mae_val:.4f}",
             transform=ax2.transAxes, fontsize=9, color="#f28e2b")

    # ── [1,1] Feature importances (RF) ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    rf_model = rf_pipe.named_steps["model"]
    importances = rf_model.feature_importances_   # shape (n_features,)
    sorted_idx  = np.argsort(importances)[::-1]
    feat_labels = [FEATURE_NAMES[i] for i in sorted_idx]
    feat_vals   = importances[sorted_idx]
    bar_colors  = ["#4e79a7" if "RL" in f else "#59a14f" for f in feat_labels]
    ax3.barh(feat_labels[::-1], feat_vals[::-1], color=bar_colors[::-1])
    ax3.set_xlabel("Mean Impurity Decrease (importance)")
    ax3.set_title("Random Forest Feature Importances\n(all LV targets combined)")

    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅  Plot saved          → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CRT Lead — Step 4: Overfit ML Experiment")
    print("=" * 60)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {NORMALIZED_JSON.name} ...")
    X, y, pids = load_dataset(NORMALIZED_JSON)
    print(f"  Dataset: {len(pids)} patients × "
          f"{len(FEATURE_NAMES)} features → {len(TARGET_NAMES)} targets")
    print(f"  Features : {FEATURE_NAMES}")
    print(f"  Targets  : {TARGET_NAMES}\n")

    if len(pids) < 5:
        print("⚠  Need at least 5 patients. Check normalized_results.json.")
        return

    # ── train all models ──────────────────────────────────────────────────────
    models = make_models()
    all_results = {}
    report_lines = []

    header = (
        f"\n{'─'*60}\n"
        f"{'Model':<16}  {'Mode':<8}  "
        + "  ".join(f"{n:>9}" for n in TARGET_NAMES)
        + f"  {'AvgT':>7}  {'AvgAng':>8}\n"
        + "─" * 60
    )
    print(header)
    report_lines.append(header)

    for model_name, pipe in models.items():
        print(f"\n[{model_name}]")
        model_result = {}

        # ── TRAIN score (deliberate overfit) ─────────────────────────────────
        pipe.fit(X, y)
        y_pred_train = pipe.predict(X)
        train_scores = score_predictions(y, y_pred_train, TARGET_NAMES)
        model_result["train"] = train_scores

        row_train = (
            f"  {model_name:<16}  {'TRAIN':<8}  "
            + "  ".join(f"{train_scores[n]:>9.4f}" for n in TARGET_NAMES)
            + f"  {train_scores['_mean_t_mae']:>7.4f}"
            + f"  {train_scores['_mean_ang_mae']:>8.2f}°"
        )
        print(row_train)
        report_lines.append(row_train)

        # ── LOO score (honest generalisation estimate) ────────────────────────
        print(f"  Running Leave-One-Out CV ({len(pids)} folds) ...", end="", flush=True)
        loo_scores = run_loo(pipe, X, y, TARGET_NAMES)
        model_result["loo"] = loo_scores
        print(" done.")

        row_loo = (
            f"  {model_name:<16}  {'LOO':8}  "
            + "  ".join(f"{loo_scores[n]:>9.4f}" for n in TARGET_NAMES)
            + f"  {loo_scores['_mean_t_mae']:>7.4f}"
            + f"  {loo_scores['_mean_ang_mae']:>8.2f}°"
        )
        print(row_loo)
        report_lines.append(row_loo)

        all_results[model_name] = model_result

    # ── summary table ─────────────────────────────────────────────────────────
    summary = [
        "",
        "=" * 60,
        "SUMMARY — Mean errors across all LV lead targets",
        "=" * 60,
        f"  {'Model':<18}  {'Train t-MAE':>12}  {'LOO t-MAE':>10}  "
        f"{'Train ang-MAE':>14}  {'LOO ang-MAE':>12}",
        "─" * 74,
    ]
    for m, r in all_results.items():
        gap_t   = r['loo']['_mean_t_mae']   - r['train']['_mean_t_mae']
        gap_ang = r['loo']['_mean_ang_mae'] - r['train']['_mean_ang_mae']
        summary.append(
            f"  {m:<18}  {r['train']['_mean_t_mae']:>12.4f}"
            f"  {r['loo']['_mean_t_mae']:>10.4f}"
            f"  {r['train']['_mean_ang_mae']:>12.2f}°"
            f"  {r['loo']['_mean_ang_mae']:>10.2f}°"
            f"  (gap: {gap_t:+.3f} / {gap_ang:+.1f}°)"
        )
    summary += [
        "─" * 74,
        "",
        "HOW TO READ THIS:",
        "  Train MAE ≈ 0          → model memorised the data ✅ (expected)",
        "  LOO MAE << random      → real signal exists in the features ✅",
        "  LOO MAE >> Train MAE   → overfitting gap (normal with n=86)",
        "  LOO t-MAE < 0.10       → within ±10% of heart axis → clinically useful",
        "  LOO ang-MAE < 20°      → within 1 clock-hour → clinically useful",
        "",
        "NEXT STEPS:",
        "  • Add more features (raw image patches around RV lead)",
        "  • Try leave-10%-out or 5-fold CV for stabler estimates",
        "  • Once Dataset 2 (308 pts) is labelled: retrain with more data",
        "=" * 60,
    ]

    for line in summary:
        print(line)
    report_lines.extend(summary)

    # ── save JSON + report ────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅  ML results saved    → {OUTPUT_JSON}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved        → {OUTPUT_REPORT}")

    # ── plot ──────────────────────────────────────────────────────────────────
    print("  Generating plot ...")
    rf_pipe = models["RandomForest"]
    make_plot(all_results, X, y, pids, rf_pipe)


if __name__ == "__main__":
    # friendly install check
    missing = []
    try: import sklearn
    except ImportError: missing.append("scikit-learn")
    try: import matplotlib
    except ImportError: missing.append("matplotlib")

    if missing:
        print(f"⚠  Missing: {missing}\n   Run:  pip install {' '.join(missing)}")
    else:
        run()