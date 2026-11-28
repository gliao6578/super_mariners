#Random Classification Forest part A

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

PROJECT_ROOT = Path("/content/super_mariners-main")
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded 30-minute windows: {len(df):,}")

df = df[df["usable_for_model"] == True].copy()
df = df[~df["vessel_group"].isin(["Not Available", "Other"])].dropna(subset=["mmsi", "vessel_group"]).copy()

print(f"Usable classified windows: {len(df):,}")
print(f"Unique vessels (MMSI): {df['mmsi'].nunique():,}\n")

agg_spec = {
    "vessel_group": ("vessel_group", lambda x: x.mode().iloc[0]),
    "active_windows": ("window_start", "count"),
    "mean_speed_knots": ("avg_speed_knots", "mean"),
    "std_speed_knots": ("avg_speed_knots", "std"),
    "peak_speed_knots": ("max_speed_knots", "max"),
    "mean_accel_mps2": ("avg_acceleration_mps2", "mean"),
    "max_abs_accel_mps2": ("max_abs_acceleration_mps2", "max"),
    "mean_turn_rate": ("avg_turn_rate_deg_min", "mean"),
    "max_turn_rate": ("max_turn_rate_deg_min", "max"),
    "mean_heading_cog_diff": ("avg_heading_cog_difference", "mean"),
    "mean_movement_efficiency": ("movement_efficiency", "mean"),
    "mean_low_speed_frac": ("low_speed_fraction", "mean"),
    "avg_turns_per_window": ("significant_turn_count", "mean"),
    "pct_time_in_bay": ("inside_bay", "mean"),
    "pct_time_in_anchorage": ("inside_anchorage", "mean"),
    "pct_time_in_restricted": ("inside_restricted_area", "mean"),
    "pct_time_in_military": ("inside_military_area", "mean"),
    "mean_dist_bay_entrance_nm": ("distance_to_bay_entrance_nm", "mean"),
    "mean_dist_channel_nm": ("distance_to_shipping_channel_nm", "mean"),
    "mean_dist_anchorage_nm": ("distance_to_anchorage_nm", "mean"),
    "mean_dist_restricted_nm": ("distance_to_restricted_area_nm", "mean"),
}

valid_agg_spec = {
    out_col: (src_col, func)
    for out_col, (src_col, func) in agg_spec.items()
    if src_col in df.columns
}

vessel_df = df.groupby("mmsi", as_index=False).agg(**valid_agg_spec)

if "std_speed_knots" in vessel_df.columns:
    vessel_df["std_speed_knots"] = vessel_df["std_speed_knots"].fillna(0)

feature_cols = [c for c in vessel_df.columns if c not in ["mmsi", "vessel_group"]]
vessel_df = vessel_df.dropna(subset=feature_cols).copy()

counts = vessel_df["vessel_group"].value_counts()
valid_groups = counts[counts >= 8].index
vessel_df = vessel_df[vessel_df["vessel_group"].isin(valid_groups)].copy()

print("--- UNIQUE VESSEL COUNTS PER BEHAVIORAL CLASS ---")
print(vessel_df["vessel_group"].value_counts(), "\n")

print("--- KRUSKAL-WALLIS STATISTICAL SIGNIFICANCE ACROSS VESSEL TYPES ---")
kw_results = []
for feat in feature_cols:
    groups = [grp[feat].values for _, grp in vessel_df.groupby("vessel_group")]
    if len(set(vessel_df[feat])) <= 1:
        continue
    h_stat, p_val = kruskal(*groups)
    kw_results.append({
        "Feature": feat,
        "H_Statistic": round(h_stat, 2),
        "p_value": f"{p_val:.2e}",
        "Significant (p<0.05)": p_val < 0.05
    })

kw_df = pd.DataFrame(kw_results).sort_values(by="H_Statistic", ascending=False)
print(kw_df.to_string(index=False), "\n")

X = vessel_df[feature_cols]
y = vessel_df["vessel_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
cv_acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

print("--- RANDOM FOREST VALIDATION METRICS (BEHAVIORAL ONLY) ---")
print(f"5-Fold CV Accuracy : {cv_acc.mean():.3f} (+/- {cv_acc.std():.3f})")
print(f"5-Fold CV Macro F1 : {cv_f1.mean():.3f} (+/- {cv_f1.std():.3f})\n")

y_pred = rf.predict(X_test)
print("--- TEST SET CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

probs = rf.predict_proba(X)
classes = list(rf.classes_)

vessel_df["predicted_behavior_class"] = rf.predict(X)
vessel_df["reported_class_probability"] = [
    probs[i, classes.index(actual)] for i, actual in enumerate(y)
]
vessel_df["behavioral_anomaly_flag"] = vessel_df["reported_class_probability"] < 0.15

out_csv = OUTPUT_DIR / "san_diego_behavioral_classification_anomalies.csv"
vessel_df.to_csv(out_csv, index=False)
print(f"Saved behavioral profiles & anomaly scores to: {out_csv}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=axes[0], cmap="Blues", xticks_rotation=45
)
axes[0].set_title("Behavioral Random Forest Confusion Matrix")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
importances.tail(15).plot(kind="barh", ax=axes[1], color="teal")
axes[1].set_title("Top 15 Movement & Spatial Features Separating Ship Types")
axes[1].set_xlabel("Gini Importance")

plt.tight_layout()
plt.show()


#Random Classification Forest Part B,  Anomoly thresholds
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/content/super_mariners-main")
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "san_diego_behavioral_classification_anomalies.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "san_diego_categorized_anomalies.csv"

df = pd.read_csv(INPUT_FILE)
print(f"Loaded total unique vessels: {len(df):,}")

metrics_to_score = [
    "mean_speed_knots",
    "peak_speed_knots",
    "mean_turn_rate",
    "mean_movement_efficiency",
    "mean_low_speed_frac",
    "pct_time_in_restricted",
    "pct_time_in_military",
    "mean_dist_channel_nm",
    "mean_heading_cog_diff",
]
available_metrics = [m for m in metrics_to_score if m in df.columns]

for col in available_metrics:
    grouped = df.groupby("vessel_group")[col]
    med = grouped.transform("median")
    q75 = grouped.transform(lambda x: x.quantile(0.75))
    q25 = grouped.transform(lambda x: x.quantile(0.25))
    iqr = (q75 - q25).replace(0, np.nan)
    df[f"{col}_z"] = ((df[col] - med) / (iqr * 0.7413)).fillna(0)

def classify_anomaly_type(row):
    reasons = []
    in_mil = row.get("pct_time_in_military", 0) - 0.30
       # Was 0.05 (now requires >30% of vessel's time)
    in_res = row.get("pct_time_in_restricted", 0) - 0.50
    # Was 0.15 (now requires >50% of vessel's time)
    if row["vessel_group"] != "Military" and (in_mil > 0 or in_res > 0):
        reasons.append("Restricted/Military Zone Intrusion")

    is_ct = {row["vessel_group"], row["predicted_behavior_class"]} == {"Cargo", "Tanker"}
    if (0.15 - row["reported_class_probability"]) > 0 and row["predicted_behavior_class"] != row["vessel_group"] and not is_ct:
        reasons.append(f"AIS Identity Mismatch (Acts like {row['predicted_behavior_class']})")

    c1 = (row.get("mean_turn_rate_z", 0) - 2.5) > 0
    c2 = (-2.0 - row.get("mean_movement_efficiency_z", 0)) > 0
    c3 = (row.get("mean_heading_cog_diff_z", 0) - 2.5) > 0
    if c1 or c2 or c3:
        reasons.append("Erratic Maneuvering / Loitering")

    if (row.get("peak_speed_knots_z", 0) - 2.5) > 0 or (row.get("mean_speed_knots_z", 0) - 2.5) > 0:
        reasons.append("Excessive Speed for Vessel Class")
    elif (row.get("mean_low_speed_frac_z", 0) - 2.5) > 0 and (0.20 - row.get("pct_time_in_anchorage", 0)) > 0:
        reasons.append("Unexplained Offshore Loitering/Stalling")

    if row["vessel_group"] in ["Cargo", "Tanker"] and (row.get("mean_dist_channel_nm_z", 0) - 2.5) > 0 and (0.20 - row.get("pct_time_in_anchorage", 0)) > 0:
        reasons.append("Shipping Channel Deviation")

    if not reasons:
        return "Normal"
    if len(reasons) == 1:
        return reasons
    return "Multi-Factor: " + " + ".join(reasons)

df["anomaly_category"] = df.apply(classify_anomaly_type, axis=1)
df["is_anomalous"] = df["anomaly_category"] != "Normal"

anomalies_only = df[df["is_anomalous"]].copy()

print("\n--- BREAKDOWN OF DETECTED ANOMALY CATEGORIES ---")
print(anomalies_only["anomaly_category"].value_counts().to_string())

print("\n--- ANOMALY RATE BY DECLARED VESSEL GROUP ---")
summary_by_type = (
    df.groupby("vessel_group")["is_anomalous"]
    .agg(total_ships="count", anomalous_ships="sum", anomaly_pct=lambda x: round(x.mean() * 100, 1))
    .sort_values("anomalous_ships", ascending=False)
)
print(summary_by_type.to_string())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved categorized anomaly dataset to: {OUTPUT_FILE}")

if not anomalies_only.empty:
    plt.figure(figsize=(12, 6))
    top_cats = anomalies_only["anomaly_category"].value_counts().head(10).sort_values()
    top_cats.plot(kind="barh", color="crimson")
    plt.title("Top Categorized Maritime Behavioral Anomalies (San Diego)")
    plt.xlabel("Number of Unique Vessels (MMSI)")
    plt.tight_layout()
    plt.show()