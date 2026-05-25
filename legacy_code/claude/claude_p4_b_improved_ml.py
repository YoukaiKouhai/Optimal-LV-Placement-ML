"""
CRT Lead Detection Project — Step 4b: Improved ML Experiment
=============================================================
Improvements over Step 4:
  1. Richer features  — adds RV lead spread, ANT angle, axis length
  2. Single-target    — predicts LL1_t and LL1_ang separately
                        (distal LV lead, most clinically important)
  3. Cleaner comparison — feature ablation table shows which features
                          actually help vs hurt
  4. Baseline comparison — always compare against "just predict the mean"
                           so we know if ML is actually adding value

Usage:
    python claude_p4b_improved_ml.py

Input:
    normalized_results.json   (Step 3 output)

Output:
    ml_4b_results.json
    ml_4b_report.txt
    ml_4b_plot.png
"""

import json
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.base            import clone
from sklearn.metrics         import mean_absolute_error

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
NORMALIZED_JSON = BASE_DIR / "normalized_results.json"
OUTPUT_JSON     = BASE_DIR / "ml_4b_results.json"
OUTPUT_REPORT   = BASE_DIR / "ml_4b_report.txt"
OUTPUT_PLOT     = BASE_DIR / "ml_4b_plot.png"


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
#
#  We build features in named groups so we can run ablation tests
#  (test which group of features actually helps).
#
#  GROUP A — basic RV position (same as Step 4)
#    RL1_t, RL1_ang, RL2_t, RL2_ang
#
#  GROUP B — RV lead geometry (new)
#    rv_spread_t    : RL2_t - RL1_t  (how spread out the RV lead is)
#    rv_spread_ang  : angular diff between RL1 and RL2 (should be small)
#    rv_mean_t      : mean of RL1_t and RL2_t
#    rv_mean_ang    : circular mean of RL1_ang and RL2_ang
#
#  GROUP C — heart size and ANT reference (new)
#    axis_len_mm    : apex-to-base distance
#    ant_ang        : radial angle of ANT marker (should be ~0 by definition,
#                     but small deviations encode asymmetry info)
#
#  FULL FEATURE SET = A + B + C  (11 features)

def circular_mean_deg(a1: float, a2: float) -> float:
    """Mean of two angles, handling 0°/360° wraparound."""
    a1r, a2r = math.radians(a1), math.radians(a2)
    mean_sin  = (math.sin(a1r) + math.sin(a2r)) / 2
    mean_cos  = (math.cos(a1r) + math.cos(a2r)) / 2
    return math.degrees(math.atan2(mean_sin, mean_cos)) % 360


def angular_diff(a1: float, a2: float) -> float:
    """Shortest angular distance between two angles (always positive, 0–180)."""
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff


def build_features(rec: dict) -> dict | None:
    """
    Extract all engineered features for one patient record.
    Returns None if any required field is missing.
    """
    elec = rec.get("electrodes", {})

    rl1 = elec.get("RL1")
    rl2 = elec.get("RL2")
    if rl1 is None or rl2 is None:
        return None

    rl1_t   = rl1["longitudinal_t"]
    rl1_ang = rl1["radial_angle_deg"]
    rl2_t   = rl2["longitudinal_t"]
    rl2_ang = rl2["radial_angle_deg"]

    # GROUP B
    rv_spread_t   = rl2_t - rl1_t
    rv_spread_ang = angular_diff(rl1_ang, rl2_ang)
    rv_mean_t     = (rl1_t + rl2_t) / 2
    rv_mean_ang   = circular_mean_deg(rl1_ang, rl2_ang)

    # GROUP C
    axis_len      = rec.get("axis_len_mm")
    if axis_len is None:
        return None

    # ANT angle: the script stores ANT in centroids but not in electrodes.
    # We reconstruct it: ANT is always at ~0° by construction of our frame,
    # but small deviations exist due to the projection math. We leave it as 0
    # here unless you add it to the normalisation output in Step 3.
    ant_ang = 0.0   # placeholder — see note above

    return {
        # Group A
        "RL1_t":         rl1_t,
        "RL1_ang":       rl1_ang,
        "RL2_t":         rl2_t,
        "RL2_ang":       rl2_ang,
        # Group B
        "rv_spread_t":   rv_spread_t,
        "rv_spread_ang": rv_spread_ang,
        "rv_mean_t":     rv_mean_t,
        "rv_mean_ang":   rv_mean_ang,
        # Group C
        "axis_len_mm":   axis_len,
        "ant_ang":       ant_ang,
    }


# Feature group definitions for ablation
FEATURE_GROUPS = {
    "A_rv_pos":    ["RL1_t", "RL1_ang", "RL2_t", "RL2_ang"],
    "B_rv_geom":   ["rv_spread_t", "rv_spread_ang", "rv_mean_t", "rv_mean_ang"],
    "C_heart":     ["axis_len_mm", "ant_ang"],
}
ALL_FEATURES = (FEATURE_GROUPS["A_rv_pos"]
              + FEATURE_GROUPS["B_rv_geom"]
              + FEATURE_GROUPS["C_heart"])

# Targets — we now predict EACH separately for cleaner analysis
TARGETS = {
    "LL1_t":   ("longitudinal_t",  "LL1"),
    "LL1_ang": ("radial_angle_deg", "LL1"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(json_path: Path, feature_list: list, target_name: str):
    """
    Build X (n_patients × n_features) and y (n_patients,) arrays
    for a single target variable.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    attr, lead = TARGETS[target_name]

    X_rows, y_vals, pids = [], [], []
    for pid in sorted(data.keys()):
        rec  = data[pid]
        feat = build_features(rec)
        if feat is None:
            continue

        elec = rec["electrodes"].get(lead)
        if elec is None:
            continue

        X_rows.append([feat[f] for f in feature_list])
        y_vals.append(elec[attr])
        pids.append(pid)

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_vals,  dtype=np.float32),
            pids)


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mae_for_target(target_name: str, y_true: np.ndarray,
                   y_pred: np.ndarray) -> float:
    """Use circular MAE for angle targets, plain MAE for longitudinal."""
    if target_name.endswith("_ang"):
        diff = np.abs(y_true - y_pred) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        return float(diff.mean())
    return float(mean_absolute_error(y_true, y_pred))


def units(target_name: str) -> str:
    return "°" if target_name.endswith("_ang") else "t-units"


def mm_equiv(t_mae: float, mean_axis_mm: float) -> float:
    """Convert a longitudinal t-MAE to millimetres using mean heart axis."""
    return t_mae * mean_axis_mm


# ─────────────────────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────────────────────

def make_models():
    return {
        "MeanBaseline":    None,          # special case — predict training mean
        "LinearRidge":     Pipeline([("sc", StandardScaler()),
                                     ("m",  Ridge(alpha=0.1))]),
        "Lasso":           Pipeline([("sc", StandardScaler()),
                                     ("m",  Lasso(alpha=0.01, max_iter=5000))]),
        "KNN_5":           Pipeline([("sc", StandardScaler()),
                                     ("m",  KNeighborsRegressor(n_neighbors=5))]),
        "KNN_3":           Pipeline([("sc", StandardScaler()),
                                     ("m",  KNeighborsRegressor(n_neighbors=3))]),
        "RandomForest":    Pipeline([("sc", StandardScaler()),
                                     ("m",  RandomForestRegressor(
                                                n_estimators=300,
                                                min_samples_leaf=1,
                                                random_state=42, n_jobs=-1))]),
        "GradientBoost":   Pipeline([("sc", StandardScaler()),
                                     ("m",  GradientBoostingRegressor(
                                                n_estimators=200,
                                                max_depth=3,
                                                learning_rate=0.05,
                                                random_state=42))]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LOO EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_loo(model_name: str, pipe, X: np.ndarray,
                 y: np.ndarray, target_name: str) -> tuple:
    """
    Returns (train_mae, loo_mae).
    MeanBaseline predicts the training-set mean each fold.
    """
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        if model_name == "MeanBaseline":
            y_pred_loo[test_idx[0]] = y[train_idx].mean()
        else:
            m = clone(pipe)
            m.fit(X[train_idx], y[train_idx])
            y_pred_loo[test_idx[0]] = m.predict(X[test_idx])[0]

    # train score (fit on all data)
    if model_name == "MeanBaseline":
        y_pred_train = np.full_like(y, y.mean())
        train_mae    = mae_for_target(target_name, y, y_pred_train)
    else:
        pipe.fit(X, y)
        y_pred_train = pipe.predict(X)
        train_mae    = mae_for_target(target_name, y, y_pred_train)

    loo_mae = mae_for_target(target_name, y, y_pred_loo)
    return train_mae, loo_mae, y_pred_loo


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ABLATION
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(json_path: Path, target_name: str,
                 mean_axis_mm: float) -> list:
    """
    Test four feature-set combinations with the best model (LinearRidge).
    Returns list of dicts for the report.
    """
    combos = {
        "A only":   FEATURE_GROUPS["A_rv_pos"],
        "A + B":    FEATURE_GROUPS["A_rv_pos"] + FEATURE_GROUPS["B_rv_geom"],
        "A + C":    FEATURE_GROUPS["A_rv_pos"] + FEATURE_GROUPS["C_heart"],
        "A + B + C": ALL_FEATURES,
    }
    results = []
    for combo_name, feats in combos.items():
        X, y, _ = load_dataset(json_path, feats, target_name)
        pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=0.1))])
        _, loo_mae, _ = evaluate_loo("LinearRidge", pipe, X, y, target_name)
        extra = ""
        if not target_name.endswith("_ang"):
            extra = f"  ≈ {mm_equiv(loo_mae, mean_axis_mm):.1f} mm"
        results.append({
            "combo":   combo_name,
            "n_feat":  len(feats),
            "loo_mae": loo_mae,
            "extra":   extra,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(results_by_target: dict, ablation_by_target: dict):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "Step 4b — Improved ML: Single-Target Prediction\n"
        "Predicting Distal LV Lead (LL1) Position from RV Lead + Heart Features",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    target_labels = list(results_by_target.keys())
    model_names   = list(next(iter(results_by_target.values())).keys())

    palette = {
        "MeanBaseline":  "#bbbbbb",
        "LinearRidge":   "#4e79a7",
        "Lasso":         "#76b7b2",
        "KNN_5":         "#f28e2b",
        "KNN_3":         "#ffbe7d",
        "RandomForest":  "#e15759",
        "GradientBoost": "#b07aa1",
    }

    for t_idx, target_name in enumerate(target_labels):
        res  = results_by_target[target_name]
        abl  = ablation_by_target[target_name]
        unit = "°" if target_name.endswith("_ang") else "t-units"

        # ── bar chart: train vs LOO per model ────────────────────────────────
        ax = fig.add_subplot(gs[0, t_idx])
        x_pos = np.arange(len(model_names))
        width = 0.38
        train_vals = [res[m]["train_mae"] for m in model_names]
        loo_vals   = [res[m]["loo_mae"]   for m in model_names]
        colors     = [palette[m] for m in model_names]

        ax.bar(x_pos - width/2, train_vals, width,
               color=colors, alpha=0.9, label="Train")
        ax.bar(x_pos + width/2, loo_vals,   width,
               color=colors, alpha=0.4, edgecolor="k", lw=0.6, label="LOO")

        # mark baseline
        baseline_loo = res["MeanBaseline"]["loo_mae"]
        ax.axhline(baseline_loo, color="red", linestyle="--",
                   linewidth=1.2, label=f"Mean baseline ({baseline_loo:.3f})")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_title(f"Target: {target_name}")
        ax.legend(fontsize=7)

        # ── ablation bar chart ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, t_idx])
        combo_names = [a["combo"] for a in abl]
        loo_maes    = [a["loo_mae"] for a in abl]
        n_feats     = [a["n_feat"] for a in abl]
        bar_colors  = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

        bars = ax2.bar(combo_names, loo_maes, color=bar_colors, alpha=0.85)
        ax2.axhline(baseline_loo, color="red", linestyle="--",
                    linewidth=1.0, label=f"Mean baseline")
        ax2.set_ylabel(f"LOO MAE ({unit})")
        ax2.set_title(f"Feature Ablation — {target_name}\n(LinearRidge)")
        ax2.set_xticklabels(combo_names, rotation=20, ha="right", fontsize=8)
        ax2.legend(fontsize=7)
        for bar, nf in zip(bars, n_feats):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"n={nf}", ha="center", va="bottom", fontsize=7)

    # ── right panel: improvement over baseline ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = {}
    for tname in target_labels:
        res      = results_by_target[tname]
        baseline = res["MeanBaseline"]["loo_mae"]
        for mname in model_names:
            if mname == "MeanBaseline":
                continue
            pct = 100 * (baseline - res[mname]["loo_mae"]) / baseline
            improvements.setdefault(mname, []).append(pct)

    model_order  = [m for m in model_names if m != "MeanBaseline"]
    mean_impr    = [np.mean(improvements[m]) for m in model_order]
    bar_c        = [palette[m] for m in model_order]
    y_pos        = np.arange(len(model_order))
    ax3.barh(y_pos, mean_impr, color=bar_c, alpha=0.85)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(model_order, fontsize=8)
    ax3.set_xlabel("% improvement over Mean Baseline\n(avg across LL1_t and LL1_ang)")
    ax3.set_title("Which Model Beats the Baseline\nby the Most?")
    for i, v in enumerate(mean_impr):
        ax3.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # ── right bottom: scatter — best model LL1_t LOO predictions ─────────────
    ax4 = fig.add_subplot(gs[1, 2])
    # re-extract best model's LOO predictions for LL1_t
    best_model = model_order[int(np.argmax(mean_impr))]
    X_full, y_full, pids = load_dataset(NORMALIZED_JSON, ALL_FEATURES, "LL1_t")
    best_pipe = make_models()[best_model]
    _, _, y_loo = evaluate_loo(best_model, best_pipe, X_full, y_full, "LL1_t")
    ax4.scatter(y_full, y_loo, alpha=0.75, edgecolors="k",
                linewidths=0.4, color=palette[best_model], s=55, zorder=3)
    lo = min(y_full.min(), y_loo.min()) - 0.05
    hi = max(y_full.max(), y_loo.max()) + 0.05
    ax4.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect")
    ax4.set_xlabel("Actual LL1_t")
    ax4.set_ylabel("LOO Predicted LL1_t")
    ax4.set_title(f"Best Model LOO Scatter — LL1_t\n({best_model})")
    ax4.legend(fontsize=8)
    loo_err = mae_for_target("LL1_t", y_full, y_loo)
    ax4.text(0.05, 0.92, f"LOO MAE = {loo_err:.3f} t-units",
             transform=ax4.transAxes, fontsize=9)

    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅  Plot saved → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  CRT Lead — Step 4b: Improved ML Experiment")
    print("=" * 65)

    # quick pass to get mean axis length for mm conversion
    with open(NORMALIZED_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    axis_lengths  = [v["axis_len_mm"] for v in raw.values()
                     if v.get("axis_len_mm")]
    mean_axis_mm  = float(np.mean(axis_lengths))
    print(f"\n  Mean heart axis length across dataset: {mean_axis_mm:.1f} mm")
    print(f"  So t-MAE of 0.10 ≈ {0.10 * mean_axis_mm:.1f} mm along axis\n")

    models = make_models()
    results_by_target  = {}
    ablation_by_target = {}
    report_lines       = []

    for target_name in TARGETS:
        print(f"\n{'━'*65}")
        print(f"  TARGET: {target_name}  ({units(target_name)})")
        print(f"{'━'*65}")

        X, y, pids = load_dataset(NORMALIZED_JSON, ALL_FEATURES, target_name)
        print(f"  {len(pids)} patients, {X.shape[1]} features\n")

        target_results = {}
        hdr = (f"  {'Model':<16}  {'Train MAE':>10}  {'LOO MAE':>10}"
               f"  {'vs Baseline':>12}  {'mm equiv':>10}")
        print(hdr)
        print("  " + "─" * 62)
        report_lines += [f"\nTARGET: {target_name}", hdr, "  " + "─"*62]

        baseline_loo = None

        for model_name, pipe in models.items():
            train_mae, loo_mae, y_pred_loo = evaluate_loo(
                model_name, pipe, X, y, target_name
            )

            if model_name == "MeanBaseline":
                baseline_loo = loo_mae
                vs_str  = "  (reference)"
                mm_str  = ""
            else:
                diff   = loo_mae - baseline_loo
                sign   = "+" if diff >= 0 else ""
                vs_str = f"  {sign}{diff:.4f}"
                if not target_name.endswith("_ang"):
                    mm_str = f"  ≈{mm_equiv(loo_mae, mean_axis_mm):.1f} mm"
                else:
                    mm_str = ""

            target_results[model_name] = {
                "train_mae": round(train_mae, 4),
                "loo_mae":   round(loo_mae, 4),
            }

            unit_str = units(target_name)
            row = (f"  {model_name:<16}  {train_mae:>9.4f}{unit_str}"
                   f"  {loo_mae:>9.4f}{unit_str}"
                   f"{vs_str}{mm_str}")
            print(row)
            report_lines.append(row)

        results_by_target[target_name] = target_results

        # ── feature ablation ──────────────────────────────────────────────────
        print(f"\n  Feature Ablation (LinearRidge, LOO):")
        print(f"  {'Feature Set':<14}  {'N feats':>8}  {'LOO MAE':>10}")
        print("  " + "─" * 38)
        report_lines += [f"\n  Ablation:", f"  {'Feature Set':<14}  N  LOO MAE"]

        abl = run_ablation(NORMALIZED_JSON, target_name, mean_axis_mm)
        for a in abl:
            row = (f"  {a['combo']:<14}  {a['n_feat']:>6}  "
                   f"  {a['loo_mae']:>8.4f} {units(target_name)}{a['extra']}")
            print(row)
            report_lines.append(row)
        ablation_by_target[target_name] = abl

    # ── overall summary ───────────────────────────────────────────────────────
    print(f"\n{'━'*65}")
    print("  OVERALL SUMMARY")
    print(f"{'━'*65}")

    summary = [
        "",
        "━"*65,
        "OVERALL SUMMARY — best LOO performance per target",
        "━"*65,
    ]

    for target_name, res in results_by_target.items():
        baseline = res["MeanBaseline"]["loo_mae"]
        non_base = {m: v for m, v in res.items() if m != "MeanBaseline"}
        best_m   = min(non_base, key=lambda m: non_base[m]["loo_mae"])
        best_loo = non_base[best_m]["loo_mae"]
        impr_pct = 100 * (baseline - best_loo) / baseline
        mm_str   = (f"  ≈ {mm_equiv(best_loo, mean_axis_mm):.1f} mm"
                    if not target_name.endswith("_ang") else "")

        line = (f"  {target_name:<10}  best={best_m:<16}"
                f"  LOO={best_loo:.4f} {units(target_name)}{mm_str}"
                f"  ({impr_pct:+.1f}% vs mean-baseline)")
        print(line)
        summary.append(line)

    interpretation = [
        "",
        "INTERPRETATION:",
        f"  Mean heart axis = {mean_axis_mm:.1f} mm",
        "  LOO t-MAE 0.10 → predictions off by ~1 electrode spacing along axis",
        "  LOO ang-MAE 15° → roughly half a clock-segment off",
        "",
        "WHAT TO DO NEXT:",
        "  • If LOO ang-MAE > 20°: the signal for angle is weak with these",
        "    features alone — consider adding image-based features (texture",
        "    around the RV lead, wall motion data) in a future step.",
        "  • If best model beats baseline by > 15%: signal is real and worth",
        "    pursuing with more labeled data from Dataset 2.",
        "  • The ablation table shows which feature groups actually helped —",
        "    drop any group that made LOO MAE worse.",
        "━"*65,
    ]
    summary.extend(interpretation)
    for line in interpretation:
        print(line)

    report_lines.extend(summary)

    # ── save outputs ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "mean_axis_mm":      mean_axis_mm,
            "results_by_target": results_by_target,
            "ablation":          {t: abl for t, abl in ablation_by_target.items()},
        }, f, indent=2)
    print(f"\n✅  Results saved → {OUTPUT_JSON}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved  → {OUTPUT_REPORT}")

    print("  Generating plot ...")
    make_plot(results_by_target, ablation_by_target)


if __name__ == "__main__":
    missing = []
    try: import sklearn
    except ImportError: missing.append("scikit-learn")
    try: import matplotlib
    except ImportError: missing.append("matplotlib")

    if missing:
        print(f"⚠  Missing: {missing}\n   Run: pip install {' '.join(missing)}")
    else:
        run()