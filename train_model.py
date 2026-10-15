"""
train_model.py — Machine Learning models for Classic Models dataset.

Three models are trained:
  1. Revenue Forecaster     — Predicts next-month order-line revenue
                              (Random Forest Regressor on time-series features)
  2. Customer Churn Model   — Predicts if a customer will NOT place another order
                              (Random Forest Classifier)
  3. Customer CLV Model     — Predicts total lifetime payment amount per customer
                              (Gradient Boosting Regressor)

Usage:
    python train_model.py

Requirements:
    pip install -r requirements.txt

Outputs (saved to ./models/):
    revenue_forecaster.pkl
    churn_classifier.pkl
    clv_regressor.pkl
    model_report.txt
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(ROOT, "cleaned_data")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

REPORT_PATH = os.path.join(MODEL_DIR, "model_report.txt")


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(name: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run clean_data.py first — missing: {path}")
    return pd.read_csv(path)


def save_model(model, filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def section(title: str, lines: list[str]) -> str:
    sep = "═" * 60
    body = "\n".join(f"  {l}" for l in lines)
    return f"\n{sep}\n  {title}\n{sep}\n{body}\n"


# ════════════════════════════════════════════════════════════════
# MODEL 1 — Revenue Forecaster
# ════════════════════════════════════════════════════════════════
def build_revenue_features() -> pd.DataFrame:
    """
    Build a monthly time-series feature table.
    Features: month index, month-of-year, lag-1, lag-2, lag-3,
              rolling-3-mean, rolling-6-mean, year, quarter.
    Target: revenue for that month.
    """
    od     = load("order_details")
    orders = load("orders")

    od["lineTotal"]   = pd.to_numeric(od["quantityOrdered"], errors="coerce") * \
                        pd.to_numeric(od["priceEach"],        errors="coerce")
    orders["orderDate"] = pd.to_datetime(orders["orderDate"], errors="coerce")

    merged = od.merge(orders[["orderNumber", "orderDate"]], on="orderNumber", how="left")
    merged["month"] = merged["orderDate"].dt.to_period("M")

    monthly = (
        merged.groupby("month")["lineTotal"]
        .sum()
        .sort_index()
        .reset_index()
        .rename(columns={"lineTotal": "revenue"})
    )
    monthly["month_idx"] = range(len(monthly))
    monthly["month_num"] = monthly["month"].dt.month
    monthly["year"]      = monthly["month"].dt.year
    monthly["quarter"]   = monthly["month"].dt.quarter
    monthly["lag1"]      = monthly["revenue"].shift(1)
    monthly["lag2"]      = monthly["revenue"].shift(2)
    monthly["lag3"]      = monthly["revenue"].shift(3)
    monthly["roll3"]     = monthly["revenue"].shift(1).rolling(3).mean()
    monthly["roll6"]     = monthly["revenue"].shift(1).rolling(6).mean()

    return monthly.dropna().reset_index(drop=True)


def train_revenue_model(report_lines: list) -> dict:
    print("▶  Training Revenue Forecaster …")
    df = build_revenue_features()

    features = ["month_idx", "month_num", "year", "quarter",
                "lag1", "lag2", "lag3", "roll3", "roll6"]
    X, y = df[features], df["revenue"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds, squared=False)
    r2   = r2_score(y_te, preds)

    path = save_model(model, "revenue_forecaster.pkl")

    report_lines += [
        f"Algorithm      : Random Forest Regressor (200 trees)",
        f"Train rows     : {len(X_tr)}   |   Test rows: {len(X_te)}",
        f"MAE            : ${mae:,.2f}",
        f"RMSE           : ${rmse:,.2f}",
        f"R² Score       : {r2:.4f}",
        f"Saved to       : {path}",
        "",
        "Feature Importances (top 5):",
    ]
    importances = sorted(zip(features, model.feature_importances_),
                         key=lambda x: -x[1])
    for feat, imp in importances[:5]:
        report_lines.append(f"  {feat:<12} {imp:.4f}")

    print(f"   MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.3f}")
    return {"model": model, "mae": mae, "rmse": rmse, "r2": r2}


# ════════════════════════════════════════════════════════════════
# MODEL 2 — Customer Churn Classifier
# ════════════════════════════════════════════════════════════════
def build_churn_features() -> pd.DataFrame:
    """
    Features per customer computed from the first 60% of the timeline.
    Label = 1 (churned) if the customer placed no order in the last 40%.
    Features: recency (days since last order), frequency (# orders),
              monetary (total spend), avg order value, distinct products,
              avg days between orders, country-encoded, credit limit.
    """
    orders   = load("orders")
    od       = load("order_details")
    customers = load("customers")
    payments  = load("payments")

    orders["orderDate"]  = pd.to_datetime(orders["orderDate"],  errors="coerce")
    payments["paymentDate"] = pd.to_datetime(payments["paymentDate"], errors="coerce")
    od["lineTotal"] = pd.to_numeric(od["quantityOrdered"], errors="coerce") * \
                      pd.to_numeric(od["priceEach"],        errors="coerce")
    customers["creditLimit"] = pd.to_numeric(customers["creditLimit"], errors="coerce").fillna(0)

    cutoff = orders["orderDate"].quantile(0.60)
    orders_train = orders[orders["orderDate"] <= cutoff]
    orders_test  = orders[orders["orderDate"] >  cutoff]

    # Customers active in train period
    active_custs = set(orders_train["customerNumber"].astype(str))
    # Customers who ordered in test period (not churned)
    retained = set(orders_test["customerNumber"].astype(str))

    # Merge order details into train orders
    train_merged = orders_train.merge(od, on="orderNumber", how="left")

    # RFM features
    ref_date = cutoff
    rfm = (
        orders_train.groupby("customerNumber")
        .agg(
            frequency=("orderNumber", "count"),
            recency_days=("orderDate", lambda x: (ref_date - x.max()).days),
            first_order=("orderDate", "min"),
        )
        .reset_index()
    )
    rfm["tenure_days"] = (ref_date - rfm["first_order"]).dt.days

    monetary = (
        train_merged.groupby("customerNumber")["lineTotal"]
        .sum()
        .reset_index()
        .rename(columns={"lineTotal": "monetary"})
    )
    avg_order = (
        train_merged.groupby(["customerNumber", "orderNumber"])["lineTotal"]
        .sum()
        .reset_index()
        .groupby("customerNumber")["lineTotal"]
        .mean()
        .reset_index()
        .rename(columns={"lineTotal": "avg_order_value"})
    )
    distinct_products = (
        train_merged.groupby("customerNumber")["productCode"]
        .nunique()
        .reset_index()
        .rename(columns={"productCode": "distinct_products"})
    )

    # Payment features
    pay_feat = (
        payments[payments["paymentDate"] <= cutoff]
        .groupby("customerNumber")["amount"]
        .agg(total_payments="sum", payment_count="count")
        .reset_index()
    )

    # Merge all
    feat = rfm.merge(monetary, on="customerNumber", how="left") \
              .merge(avg_order, on="customerNumber", how="left") \
              .merge(distinct_products, on="customerNumber", how="left") \
              .merge(pay_feat, on="customerNumber", how="left") \
              .merge(customers[["customerNumber", "country", "creditLimit"]], on="customerNumber", how="left")

    feat["customerNumber"] = feat["customerNumber"].astype(str)
    feat["churned"] = (~feat["customerNumber"].isin(retained)).astype(int)

    # Encode country
    le = LabelEncoder()
    feat["country_enc"] = le.fit_transform(feat["country"].fillna("Unknown"))
    feat = feat.fillna(0)

    return feat


def train_churn_model(report_lines: list) -> dict:
    print("▶  Training Customer Churn Classifier …")
    df = build_churn_features()

    features = ["frequency", "recency_days", "tenure_days", "monetary",
                "avg_order_value", "distinct_products", "total_payments",
                "payment_count", "creditLimit", "country_enc"]
    X = df[features]
    y = df["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    preds      = model.predict(X_te)
    preds_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, preds)
    try:
        auc = roc_auc_score(y_te, preds_prob)
    except Exception:
        auc = float("nan")
    cr = classification_report(y_te, preds, target_names=["Retained", "Churned"])

    path = save_model(model, "churn_classifier.pkl")

    report_lines += [
        f"Algorithm      : Random Forest Classifier (200 trees, balanced)",
        f"Train rows     : {len(X_tr)}   |   Test rows: {len(X_te)}",
        f"Accuracy       : {acc:.4f}",
        f"ROC-AUC        : {auc:.4f}",
        f"Saved to       : {path}",
        "",
        "Classification Report:",
    ]
    for line in cr.strip().split("\n"):
        report_lines.append(f"  {line}")
    report_lines.append("")
    report_lines.append("Feature Importances (top 5):")
    importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:5]:
        report_lines.append(f"  {feat:<22} {imp:.4f}")

    print(f"   Accuracy={acc:.3f}  ROC-AUC={auc:.3f}")
    return {"model": model, "accuracy": acc, "auc": auc}


# ════════════════════════════════════════════════════════════════
# MODEL 3 — Customer Lifetime Value (CLV)
# ════════════════════════════════════════════════════════════════
def build_clv_features() -> pd.DataFrame:
    """
    Predict total payment amount per customer.
    Features derived from orders, order details, and customer master.
    """
    orders    = load("orders")
    od        = load("order_details")
    customers = load("customers")
    payments  = load("payments")

    orders["orderDate"]     = pd.to_datetime(orders["orderDate"],     errors="coerce")
    payments["paymentDate"] = pd.to_datetime(payments["paymentDate"], errors="coerce")
    od["lineTotal"] = pd.to_numeric(od["quantityOrdered"], errors="coerce") * \
                      pd.to_numeric(od["priceEach"],        errors="coerce")
    customers["creditLimit"] = pd.to_numeric(customers["creditLimit"], errors="coerce").fillna(0)

    # Target: total payments per customer
    clv_target = (
        payments.groupby("customerNumber")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "clv"})
    )

    # Order features
    order_feat = (
        orders.groupby("customerNumber")
        .agg(
            order_count=("orderNumber", "count"),
            first_order=("orderDate", "min"),
            last_order=("orderDate", "max"),
        )
        .reset_index()
    )
    order_feat["tenure_days"] = (order_feat["last_order"] - order_feat["first_order"]).dt.days

    # Line-item features
    merged = orders.merge(od, on="orderNumber", how="left")
    line_feat = (
        merged.groupby("customerNumber")
        .agg(
            total_revenue=("lineTotal", "sum"),
            avg_order_value=("lineTotal", "mean"),
            total_units=("quantityOrdered", "sum"),
            distinct_products=("productCode", "nunique"),
        )
        .reset_index()
    )

    # Country encoding
    le = LabelEncoder()
    customers["country_enc"] = le.fit_transform(customers["country"].fillna("Unknown"))

    feat = (
        clv_target
        .merge(order_feat[["customerNumber", "order_count", "tenure_days"]], on="customerNumber", how="left")
        .merge(line_feat, on="customerNumber", how="left")
        .merge(customers[["customerNumber", "creditLimit", "country_enc"]], on="customerNumber", how="left")
        .fillna(0)
    )
    return feat


def train_clv_model(report_lines: list) -> dict:
    print("▶  Training Customer CLV Regressor …")
    df = build_clv_features()

    features = ["order_count", "tenure_days", "total_revenue",
                "avg_order_value", "total_units", "distinct_products",
                "creditLimit", "country_enc"]
    X, y = df[features], df["clv"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds, squared=False)
    r2   = r2_score(y_te, preds)

    # Cross-validated R²
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    path = save_model(model, "clv_regressor.pkl")

    report_lines += [
        f"Algorithm      : Gradient Boosting Regressor (300 trees, lr=0.05)",
        f"Train rows     : {len(X_tr)}   |   Test rows: {len(X_te)}",
        f"MAE            : ${mae:,.2f}",
        f"RMSE           : ${rmse:,.2f}",
        f"R² Score       : {r2:.4f}",
        f"CV R² (5-fold) : {cv_r2:.4f}",
        f"Saved to       : {path}",
        "",
        "Feature Importances (top 5):",
    ]
    importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:5]:
        report_lines.append(f"  {feat:<22} {imp:.4f}")

    print(f"   MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.3f}  CV-R²={cv_r2:.3f}")
    return {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "cv_r2": cv_r2}


# ════════════════════════════════════════════════════════════════
# PREDICTION HELPERS (use saved models)
# ════════════════════════════════════════════════════════════════
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_next_month_revenue() -> float:
    """Predict revenue for the month after the last known month."""
    model = load_model("revenue_forecaster.pkl")
    df = build_revenue_features()
    last = df.iloc[-1]
    next_features = pd.DataFrame([{
        "month_idx": last["month_idx"] + 1,
        "month_num": (int(last["month_num"]) % 12) + 1,
        "year":      int(last["year"]) + (1 if last["month_num"] == 12 else 0),
        "quarter":   ((int(last["month_num"]) % 12) // 3) + 1,
        "lag1":      last["revenue"],
        "lag2":      last["lag1"],
        "lag3":      last["lag2"],
        "roll3":     df["revenue"].iloc[-3:].mean(),
        "roll6":     df["revenue"].iloc[-6:].mean(),
    }])
    return float(model.predict(next_features)[0])


def score_customer_churn() -> pd.DataFrame:
    """Return churn probability for every customer."""
    model = load_model("churn_classifier.pkl")
    df = build_churn_features()
    features = ["frequency", "recency_days", "tenure_days", "monetary",
                "avg_order_value", "distinct_products", "total_payments",
                "payment_count", "creditLimit", "country_enc"]
    probs = model.predict_proba(df[features])[:, 1]
    result = df[["customerNumber"]].copy()
    result["churn_probability"] = probs.round(4)
    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def predict_clv() -> pd.DataFrame:
    """Return predicted CLV for every customer."""
    model = load_model("clv_regressor.pkl")
    df = build_clv_features()
    features = ["order_count", "tenure_days", "total_revenue",
                "avg_order_value", "total_units", "distinct_products",
                "creditLimit", "country_enc"]
    preds = model.predict(df[features])
    result = df[["customerNumber", "clv"]].copy()
    result["predicted_clv"] = preds.round(2)
    result["error"] = (result["predicted_clv"] - result["clv"]).round(2)
    return result.sort_values("predicted_clv", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  🏎️  Classic Models — Model Training")
    print("═" * 60)

    all_report_lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Models dir: {MODEL_DIR}",
    ]

    # ── Model 1 ──
    rev_lines = []
    train_revenue_model(rev_lines)
    all_report_lines.append(section("MODEL 1 — Revenue Forecaster", rev_lines))

    # ── Model 2 ──
    churn_lines = []
    train_churn_model(churn_lines)
    all_report_lines.append(section("MODEL 2 — Customer Churn Classifier", churn_lines))

    # ── Model 3 ──
    clv_lines = []
    train_clv_model(clv_lines)
    all_report_lines.append(section("MODEL 3 — Customer CLV Regressor", clv_lines))

    # ── Predictions ──
    next_rev = predict_next_month_revenue()
    churn_df = score_customer_churn()
    clv_df   = predict_clv()

    pred_lines = [
        f"Next-Month Revenue Forecast : ${next_rev:,.2f}",
        "",
        "Top 5 Highest Churn-Risk Customers:",
    ]
    for _, row in churn_df.head(5).iterrows():
        pred_lines.append(f"  Customer {row['customerNumber']:<8} churn_prob={row['churn_probability']:.3f}")
    pred_lines += [
        "",
        "Top 5 Highest Predicted CLV Customers:",
    ]
    for _, row in clv_df.head(5).iterrows():
        pred_lines.append(f"  Customer {row['customerNumber']:<8} predicted_clv=${row['predicted_clv']:>10,.2f}")

    all_report_lines.append(section("PREDICTIONS", pred_lines))

    # ── Write report ──
    report_text = "\n".join(str(l) for l in all_report_lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"\n✅  All models saved to: {MODEL_DIR}")
    print(f"📄  Report saved to:     {REPORT_PATH}")
    print(f"\n   Next-month revenue forecast: ${next_rev:,.2f}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
