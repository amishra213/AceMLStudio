"""
AceML Studio – Industry Templates Engine
==========================================
Pre-built ML pipeline templates for common industry verticals.

Each template provides:
  • Recommended target variables (if discoverable)
  • Feature engineering steps tailored to the domain
  • Model recommendations with rationale
  • Evaluation metrics to prioritise
  • Domain-specific data-quality checks
  • Preprocessing pipeline config

Supported industries:
  retail_ecommerce, finance_banking, healthcare, manufacturing,
  marketing, hr_people_analytics, real_estate, energy_utilities,
  logistics_supply_chain, insurance

Usage:
    config = IndustryTemplates.get_template("retail_ecommerce")
    result = IndustryTemplates.apply_template(df, "retail_ecommerce")
    recs   = IndustryTemplates.recommend_industry(df)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aceml.industry_templates")


# ════════════════════════════════════════════════════════════════════
#  Template definitions
# ════════════════════════════════════════════════════════════════════

_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ── Retail / E-Commerce ──────────────────────────────────────────
    "retail_ecommerce": {
        "name": "Retail & E-Commerce",
        "icon": "🛍️",
        "description": "Customer behaviour, product recommendations, churn, and sales forecasting.",
        "use_cases": [
            "Customer churn prediction",
            "Product recommendation",
            "Sales forecasting",
            "Customer lifetime value (CLV)",
            "Price optimisation",
            "Inventory demand forecasting",
        ],
        "target_column_hints": ["churn", "clv", "revenue", "quantity", "sales", "purchase",
                                  "returned", "converted", "lifetime_value"],
        "key_features": {
            "datetime": ["order_date", "purchase_date", "signup_date", "last_login"],
            "monetary": ["price", "revenue", "total_spend", "discount", "margin"],
            "behavioural": ["page_views", "sessions", "clicks", "cart_adds", "returns"],
            "categorical": ["category", "product_type", "channel", "region", "segment"],
        },
        "feature_engineering": [
            {"step": "recency_frequency_monetary",
             "description": "Compute RFM (Recency, Frequency, Monetary) features from transaction data",
             "columns_needed": ["customer_id", "order_date", "revenue"],
             "output_features": ["recency_days", "frequency", "monetary_value"]},
            {"step": "datetime_decomposition",
             "description": "Extract year, month, day-of-week, is_weekend, quarter from order dates",
             "columns_needed": ["order_date"],
             "output_features": ["order_year", "order_month", "order_dow", "order_is_weekend"]},
            {"step": "price_elasticity",
             "description": "Compute log(price) and price × discount interaction",
             "columns_needed": ["price", "discount"],
             "output_features": ["log_price", "price_discount_interaction"]},
            {"step": "clv_segments",
             "description": "Segment customers by spend percentile (Low / Medium / High)",
             "columns_needed": ["total_spend"],
             "output_features": ["spend_segment"]},
            {"step": "category_encoding",
             "description": "Target-encode high-cardinality product categories",
             "columns_needed": ["category"],
             "output_features": ["category_encoded"]},
        ],
        "recommended_models": {
            "churn": ["xgboost", "random_forest", "logistic_regression"],
            "forecasting": ["gradient_boosting", "prophet", "lstm"],
            "recommendation": ["matrix_factorization", "als", "lightgbm"],
            "clv": ["gradient_boosting", "random_forest", "linear_regression"],
        },
        "primary_metrics": {
            "churn": ["roc_auc", "f1_score", "precision_at_k"],
            "forecasting": ["mae", "rmse", "mape"],
            "recommendation": ["ndcg", "map", "hit_rate"],
        },
        "data_quality_checks": [
            "Check for duplicate customer IDs",
            "Validate purchase dates are not in the future",
            "Check for negative prices or quantities",
            "Detect inactive customers (no purchase in 12+ months)",
            "Validate target encoding to avoid leakage",
        ],
        "column_name_patterns": [
            r"customer", r"order", r"product", r"sku", r"basket",
            r"revenue", r"price", r"discount", r"cart", r"purchase",
        ],
    },

    # ── Finance / Banking ────────────────────────────────────────────
    "finance_banking": {
        "name": "Finance & Banking",
        "icon": "💰",
        "description": "Credit risk, fraud detection, loan default, and financial forecasting.",
        "use_cases": [
            "Credit default prediction",
            "Fraud detection",
            "Credit scoring",
            "Loan approval",
            "Customer segmentation",
            "Stock price forecasting",
        ],
        "target_column_hints": ["default", "fraud", "chargeback", "approved", "risk_score",
                                  "delinquent", "charged_off"],
        "key_features": {
            "financial": ["income", "balance", "credit_score", "debt", "loan_amount", "interest_rate"],
            "behavioural": ["missed_payments", "late_payments", "utilisation_rate", "transactions_count"],
            "demographic": ["age", "employment_status", "years_employed", "education"],
        },
        "feature_engineering": [
            {"step": "debt_to_income",
             "description": "Compute Debt-to-Income ratio (DTI = total_debt / annual_income)",
             "columns_needed": ["total_debt", "annual_income"],
             "output_features": ["debt_to_income_ratio"]},
            {"step": "credit_utilisation",
             "description": "Credit utilisation = balance / credit_limit",
             "columns_needed": ["balance", "credit_limit"],
             "output_features": ["credit_utilisation_pct"]},
            {"step": "payment_history_features",
             "description": "Count missed_payments, create binary flag: ever_missed_payment",
             "columns_needed": ["missed_payments"],
             "output_features": ["ever_missed_payment", "log_missed_payments"]},
            {"step": "log_income",
             "description": "Log-transform income and loan amount (right-skewed distributions)",
             "columns_needed": ["income", "loan_amount"],
             "output_features": ["log_income", "log_loan_amount"]},
            {"step": "age_buckets",
             "description": "Bucket age into lifecycle stages (e.g. 18–25, 26–40, 41–60, 60+)",
             "columns_needed": ["age"],
             "output_features": ["age_bucket"]},
        ],
        "recommended_models": {
            "default": ["xgboost", "gradient_boosting", "logistic_regression"],
            "fraud": ["isolation_forest", "xgboost", "random_forest"],
            "scoring": ["logistic_regression", "gradient_boosting", "calibrated_svm"],
        },
        "primary_metrics": {
            "default": ["roc_auc", "ks_statistic", "gini"],
            "fraud": ["precision", "recall", "f1_score", "roc_auc"],
            "scoring": ["roc_auc", "ks_statistic", "calibration"],
        },
        "data_quality_checks": [
            "Check for class imbalance (fraud is typically < 1% of transactions)",
            "Validate income/loan amounts are positive",
            "Check for data leakage in payment history features",
            "Ensure credit scores are within valid range (300–850)",
            "Detect temporal leakage in time-series splits",
        ],
        "column_name_patterns": [
            r"income", r"loan", r"credit", r"debt", r"balance",
            r"fraud", r"default", r"payment", r"interest", r"bank",
        ],
    },

    # ── Healthcare ───────────────────────────────────────────────────
    "healthcare": {
        "name": "Healthcare & Clinical",
        "icon": "🏥",
        "description": "Patient outcome prediction, readmission risk, and clinical analytics.",
        "use_cases": [
            "Patient readmission prediction",
            "Disease risk stratification",
            "Length-of-stay prediction",
            "Medication adherence",
            "Diagnostic support",
            "Claims fraud detection",
        ],
        "target_column_hints": ["readmitted", "diagnosis", "mortality", "los", "outcome",
                                  "complication", "discharge", "risk_level"],
        "key_features": {
            "clinical": ["age", "diagnosis_code", "procedure_code", "medication",
                          "lab_value", "vital_sign"],
            "administrative": ["insurance", "los", "admission_type", "discharge_disposition"],
            "social_determinants": ["zip_code", "income_level", "education"],
        },
        "feature_engineering": [
            {"step": "age_risk_buckets",
             "description": "Segment patients into clinical age risk groups (<18, 18–40, 41–65, 65+)",
             "columns_needed": ["age"],
             "output_features": ["age_risk_group"]},
            {"step": "comorbidity_index",
             "description": "Sum binary comorbidity flags to create a composite comorbidity score",
             "columns_needed": ["diagnosis_*", "condition_*"],
             "output_features": ["comorbidity_score"]},
            {"step": "los_log_transform",
             "description": "Log-transform length-of-stay (right-skewed)",
             "columns_needed": ["los"],
             "output_features": ["log_los"]},
            {"step": "vital_sign_anomaly",
             "description": "Flag abnormal vital signs (out-of-range BP, HR, SpO2)",
             "columns_needed": ["blood_pressure", "heart_rate", "spo2"],
             "output_features": ["vital_anomaly_flag"]},
            {"step": "prior_visits",
             "description": "Count prior admissions / ER visits per patient",
             "columns_needed": ["patient_id", "admission_date"],
             "output_features": ["prior_visit_count", "days_since_last_visit"]},
        ],
        "recommended_models": {
            "readmission": ["xgboost", "random_forest", "logistic_regression"],
            "los": ["gradient_boosting", "random_forest", "linear_regression"],
            "mortality": ["xgboost", "neural_network", "gradient_boosting"],
        },
        "primary_metrics": {
            "readmission": ["roc_auc", "recall", "f1_score"],
            "risk": ["roc_auc", "calibration", "brier_score"],
            "regression": ["mae", "rmse", "r2"],
        },
        "data_quality_checks": [
            "Verify PHI is removed or de-identified before modelling",
            "Check for impossible lab values (e.g. negative age)",
            "Validate diagnosis codes against ICD-10 codeset",
            "Check for temporal leakage (future labels leaked into training)",
            "Assess class imbalance for rare outcomes",
        ],
        "column_name_patterns": [
            r"patient", r"diagnosis", r"icd", r"admission", r"discharge",
            r"los", r"readmit", r"hospital", r"medication", r"lab",
        ],
    },

    # ── Manufacturing ────────────────────────────────────────────────
    "manufacturing": {
        "name": "Manufacturing & Quality",
        "icon": "🏭",
        "description": "Predictive maintenance, defect detection, yield optimisation, and process control.",
        "use_cases": [
            "Predictive maintenance (PdM)",
            "Defect / quality prediction",
            "Yield optimisation",
            "Energy consumption forecasting",
            "Supply chain disruption prediction",
            "Equipment failure detection",
        ],
        "target_column_hints": ["failure", "defect", "downtime", "yield", "quality_flag",
                                  "fault", "anomaly", "maintenance"],
        "key_features": {
            "sensor": ["temperature", "vibration", "pressure", "rpm", "current", "voltage"],
            "operational": ["run_time", "cycle_count", "load", "speed"],
            "maintenance": ["last_maintenance", "maintenance_type", "repair_count"],
            "quality": ["defect_count", "pass_fail", "tolerance"],
        },
        "feature_engineering": [
            {"step": "rolling_statistics",
             "description": "Compute rolling mean, std, min, max of sensor readings (windows: 5, 20, 60)",
             "columns_needed": ["temperature", "vibration", "pressure"],
             "output_features": ["sensor_rolling_mean", "sensor_rolling_std"]},
            {"step": "time_since_maintenance",
             "description": "Calculate days/hours since last scheduled maintenance",
             "columns_needed": ["last_maintenance_date"],
             "output_features": ["days_since_maintenance"]},
            {"step": "sensor_interactions",
             "description": "Create temp × vibration, pressure / rpm interaction features",
             "columns_needed": ["temperature", "vibration", "pressure", "rpm"],
             "output_features": ["temp_vibration", "pressure_per_rpm"]},
            {"step": "log_operational_metrics",
             "description": "Log-transform run_time and cycle_count",
             "columns_needed": ["run_time", "cycle_count"],
             "output_features": ["log_run_time", "log_cycle_count"]},
            {"step": "lag_features",
             "description": "Create 1, 6, 24-step lag features for sensor time series",
             "columns_needed": ["temperature", "vibration"],
             "output_features": ["temp_lag_1", "temp_lag_6", "vibration_lag_1"]},
        ],
        "recommended_models": {
            "failure_detection": ["isolation_forest", "xgboost", "random_forest"],
            "rul": ["gradient_boosting", "lstm", "random_forest"],
            "quality": ["random_forest", "gradient_boosting", "svm"],
        },
        "primary_metrics": {
            "binary": ["roc_auc", "recall", "f1_score"],
            "regression": ["mae", "rmse", "r2"],
            "anomaly": ["precision", "recall", "f1_score"],
        },
        "data_quality_checks": [
            "Check sensor readings for physically impossible values",
            "Detect sensor drift or stuck values (constant readings)",
            "Validate timestamps are monotonically increasing",
            "Check for class imbalance (failures are rare events)",
            "Assess data gaps during planned downtime periods",
        ],
        "column_name_patterns": [
            r"sensor", r"machine", r"temperature", r"vibration", r"pressure",
            r"failure", r"defect", r"maintenance", r"yield", r"rpm",
        ],
    },

    # ── Marketing ────────────────────────────────────────────────────
    "marketing": {
        "name": "Marketing & Campaign Analytics",
        "icon": "📢",
        "description": "Campaign ROI, lead scoring, attribution, and customer segmentation.",
        "use_cases": [
            "Lead conversion prediction",
            "Campaign response modelling",
            "Marketing mix modelling",
            "Customer segmentation (RFM)",
            "Attribution modelling",
            "Email open/click prediction",
        ],
        "target_column_hints": ["converted", "clicked", "opened", "responded", "subscribed",
                                  "unsubscribed", "roi", "ctr"],
        "key_features": {
            "campaign": ["email_opens", "clicks", "impressions", "ad_spend", "channel"],
            "lead": ["lead_source", "score", "industry", "company_size", "job_title"],
            "engagement": ["website_visits", "time_on_site", "pages_viewed", "demo_requested"],
        },
        "feature_engineering": [
            {"step": "engagement_score",
             "description": "Weighted sum: opens×1 + clicks×3 + page_views×2",
             "columns_needed": ["email_opens", "clicks", "page_views"],
             "output_features": ["engagement_score"]},
            {"step": "recency_decay",
             "description": "Days since last interaction with exponential decay weight",
             "columns_needed": ["last_interaction_date"],
             "output_features": ["recency_days", "recency_decay_weight"]},
            {"step": "channel_encoding",
             "description": "One-hot encode marketing channel (email, social, paid, organic)",
             "columns_needed": ["channel"],
             "output_features": ["channel_email", "channel_social", "channel_paid"]},
            {"step": "ctr_features",
             "description": "Compute click-through rate: clicks / impressions",
             "columns_needed": ["clicks", "impressions"],
             "output_features": ["ctr", "log_impressions"]},
            {"step": "lead_velocity",
             "description": "Days from first touch to current stage",
             "columns_needed": ["first_touch_date", "current_stage_date"],
             "output_features": ["lead_age_days", "velocity_score"]},
        ],
        "recommended_models": {
            "conversion": ["xgboost", "logistic_regression", "random_forest"],
            "response": ["gradient_boosting", "logistic_regression", "naive_bayes"],
            "segmentation": ["kmeans", "dbscan", "gaussian_mixture"],
        },
        "primary_metrics": {
            "conversion": ["roc_auc", "precision_at_k", "lift"],
            "segmentation": ["silhouette_score", "calinski_harabasz", "inertia"],
        },
        "data_quality_checks": [
            "Check for duplicate lead IDs",
            "Validate date fields for campaign attribution windows",
            "Detect suspiciously high CTR (bot traffic)",
            "Check for future-dated events",
            "Validate spend/revenue figures are positive",
        ],
        "column_name_patterns": [
            r"campaign", r"click", r"impression", r"lead", r"email",
            r"channel", r"conversion", r"utm", r"source", r"ctr",
        ],
    },

    # ── HR / People Analytics ────────────────────────────────────────
    "hr_people_analytics": {
        "name": "HR & People Analytics",
        "icon": "👥",
        "description": "Employee attrition, performance prediction, hiring analytics, and engagement.",
        "use_cases": [
            "Employee attrition prediction",
            "Performance rating prediction",
            "Hiring success prediction",
            "Time-to-fill forecasting",
            "Compensation benchmarking",
            "Engagement / satisfaction modelling",
        ],
        "target_column_hints": ["attrition", "left", "terminated", "performance", "promoted",
                                  "rating", "satisfaction", "tenure"],
        "key_features": {
            "employment": ["tenure", "salary", "job_level", "department", "job_role"],
            "performance": ["performance_rating", "overtime", "training_hours", "promotions"],
            "satisfaction": ["satisfaction_score", "work_life_balance", "manager_rating"],
            "demographic": ["age", "gender", "education", "marital_status"],
        },
        "feature_engineering": [
            {"step": "tenure_buckets",
             "description": "Segment tenure into cohorts: 0–1yr, 1–3yr, 3–7yr, 7+ yr",
             "columns_needed": ["tenure_years"],
             "output_features": ["tenure_bucket"]},
            {"step": "salary_band",
             "description": "Compute salary percentile within job_level / department",
             "columns_needed": ["salary", "job_level"],
             "output_features": ["salary_percentile", "salary_vs_band_mean"]},
            {"step": "engagement_composite",
             "description": "Average of satisfaction_score, work_life_balance, manager_rating",
             "columns_needed": ["satisfaction_score", "work_life_balance", "manager_rating"],
             "output_features": ["engagement_composite"]},
            {"step": "overtime_flag",
             "description": "Binary: did employee work overtime in last period?",
             "columns_needed": ["overtime_hours"],
             "output_features": ["overtime_flag"]},
            {"step": "promotion_lag",
             "description": "Years since last promotion",
             "columns_needed": ["last_promotion_date"],
             "output_features": ["years_since_promotion"]},
        ],
        "recommended_models": {
            "attrition": ["xgboost", "random_forest", "logistic_regression"],
            "performance": ["gradient_boosting", "random_forest", "ridge_regression"],
            "clustering": ["kmeans", "hierarchical"],
        },
        "primary_metrics": {
            "attrition": ["roc_auc", "recall", "f1_score"],
            "performance": ["mae", "r2", "rmse"],
        },
        "data_quality_checks": [
            "Ensure sensitive attributes (gender, age, race) are handled fairly",
            "Check for survivorship bias (only current employees in dataset)",
            "Validate salary ranges by job level / location",
            "Check date logic: hire_date < termination_date",
            "Assess class imbalance for attrition (typically 10–20%)",
        ],
        "column_name_patterns": [
            r"employee", r"attrition", r"tenure", r"salary", r"performance",
            r"department", r"manager", r"satisfaction", r"promotion", r"hire",
        ],
    },

    # ── Real Estate ──────────────────────────────────────────────────
    "real_estate": {
        "name": "Real Estate & Property",
        "icon": "🏠",
        "description": "Property price prediction, rent estimation, and investment analytics.",
        "use_cases": [
            "Property price prediction",
            "Rental price estimation",
            "Investment opportunity scoring",
            "Market trend forecasting",
            "Neighbourhood analysis",
        ],
        "target_column_hints": ["price", "sale_price", "rent", "value", "assessed_value",
                                  "list_price", "sold_price"],
        "key_features": {
            "property": ["sqft", "bedrooms", "bathrooms", "lot_size", "year_built"],
            "location": ["zip_code", "neighbourhood", "city", "latitude", "longitude"],
            "market": ["days_on_market", "price_per_sqft", "price_reduced"],
        },
        "feature_engineering": [
            {"step": "age_of_property",
             "description": "current_year - year_built = property age",
             "columns_needed": ["year_built"],
             "output_features": ["property_age"]},
            {"step": "price_per_sqft",
             "description": "price / sqft as a normalised price metric",
             "columns_needed": ["price", "sqft"],
             "output_features": ["price_per_sqft"]},
            {"step": "room_ratio",
             "description": "bedrooms / bathrooms ratio; total_rooms / sqft",
             "columns_needed": ["bedrooms", "bathrooms", "sqft"],
             "output_features": ["bed_bath_ratio", "rooms_per_sqft"]},
            {"step": "log_price",
             "description": "Log-transform price and sqft (right-skewed)",
             "columns_needed": ["price", "sqft"],
             "output_features": ["log_price", "log_sqft"]},
            {"step": "neighbourhood_stats",
             "description": "Mean/median price per neighbourhood (target encoding)",
             "columns_needed": ["neighbourhood"],
             "output_features": ["neighbourhood_median_price"]},
        ],
        "recommended_models": {
            "price": ["gradient_boosting", "xgboost", "random_forest", "ridge_regression"],
            "classification": ["random_forest", "xgboost", "logistic_regression"],
        },
        "primary_metrics": {
            "price": ["mae", "rmse", "r2", "mape"],
            "classification": ["roc_auc", "f1_score"],
        },
        "data_quality_checks": [
            "Check for duplicate property IDs",
            "Validate lat/lon coordinates are within realistic ranges",
            "Detect outlier prices (e.g. $1 sales)",
            "Validate year_built is reasonable (1800–current year)",
            "Check that sqft > 0",
        ],
        "column_name_patterns": [
            r"price", r"sqft", r"bedroom", r"bathroom", r"property",
            r"zip", r"neighbourhood", r"lot", r"garage", r"built",
        ],
    },

    # ── Energy / Utilities ───────────────────────────────────────────
    "energy_utilities": {
        "name": "Energy & Utilities",
        "icon": "⚡",
        "description": "Energy consumption forecasting, demand response, and grid analytics.",
        "use_cases": [
            "Energy consumption forecasting",
            "Demand response optimisation",
            "Equipment failure prediction",
            "Renewable energy output forecasting",
            "Meter anomaly detection",
        ],
        "target_column_hints": ["consumption", "demand", "load", "kwh", "output",
                                  "generation", "usage"],
        "key_features": {
            "temporal": ["timestamp", "hour", "day_of_week", "month", "is_holiday"],
            "weather": ["temperature", "humidity", "wind_speed", "solar_radiation"],
            "operational": ["load", "voltage", "frequency", "power_factor"],
        },
        "feature_engineering": [
            {"step": "temporal_features",
             "description": "Extract hour, day_of_week, month, quarter, is_peak_hour, is_holiday",
             "columns_needed": ["timestamp"],
             "output_features": ["hour", "dow", "month", "is_peak_hour", "is_holiday"]},
            {"step": "weather_interactions",
             "description": "temperature × humidity heat index; temp² for non-linear effect",
             "columns_needed": ["temperature", "humidity"],
             "output_features": ["heat_index", "temperature_squared"]},
            {"step": "lag_and_rolling",
             "description": "24h lag, 7-day lag, 24h rolling mean/std of consumption",
             "columns_needed": ["consumption"],
             "output_features": ["consumption_lag24", "consumption_lag168", "rolling_mean_24"]},
            {"step": "load_factor",
             "description": "Load factor = avg_demand / peak_demand (efficiency metric)",
             "columns_needed": ["avg_demand", "peak_demand"],
             "output_features": ["load_factor"]},
        ],
        "recommended_models": {
            "forecasting": ["gradient_boosting", "prophet", "lstm", "random_forest"],
            "anomaly": ["isolation_forest", "autoencoder", "one_class_svm"],
        },
        "primary_metrics": {
            "forecasting": ["mae", "rmse", "mape", "r2"],
            "anomaly": ["precision", "recall", "f1_score"],
        },
        "data_quality_checks": [
            "Check for zero or negative energy readings (meter errors)",
            "Validate timestamp continuity (no missing intervals)",
            "Detect stuck meter readings (constant values over long periods)",
            "Check for daylight saving time gaps",
            "Validate weather data against known climate ranges",
        ],
        "column_name_patterns": [
            r"consumption", r"demand", r"kwh", r"load", r"generation",
            r"solar", r"wind", r"temperature", r"meter", r"energy",
        ],
    },

    # ── Logistics / Supply Chain ─────────────────────────────────────
    "logistics_supply_chain": {
        "name": "Logistics & Supply Chain",
        "icon": "📦",
        "description": "Demand forecasting, route optimisation, and supply chain risk analytics.",
        "use_cases": [
            "Demand forecasting",
            "Delivery delay prediction",
            "Inventory optimisation",
            "Supplier risk scoring",
            "Route efficiency optimisation",
        ],
        "target_column_hints": ["delay", "late", "lead_time", "demand", "quantity",
                                  "on_time", "shipped", "returned"],
        "key_features": {
            "order": ["order_date", "ship_date", "delivery_date", "quantity", "weight"],
            "route": ["origin", "destination", "distance", "carrier", "mode"],
            "product": ["sku", "category", "perishable", "priority"],
        },
        "feature_engineering": [
            {"step": "lead_time",
             "description": "Days from order_date to delivery_date",
             "columns_needed": ["order_date", "delivery_date"],
             "output_features": ["lead_time_days"]},
            {"step": "day_of_week_order",
             "description": "Order day-of-week (orders on Fridays often delayed)",
             "columns_needed": ["order_date"],
             "output_features": ["order_dow", "order_is_weekend"]},
            {"step": "weight_volume_ratio",
             "description": "Dimensional weight vs actual weight",
             "columns_needed": ["weight", "volume"],
             "output_features": ["weight_volume_ratio"]},
            {"step": "carrier_performance",
             "description": "Historical on-time delivery rate per carrier",
             "columns_needed": ["carrier", "on_time"],
             "output_features": ["carrier_on_time_rate"]},
        ],
        "recommended_models": {
            "delay": ["gradient_boosting", "random_forest", "logistic_regression"],
            "demand": ["gradient_boosting", "prophet", "sarima"],
            "risk": ["isolation_forest", "xgboost"],
        },
        "primary_metrics": {
            "delay": ["roc_auc", "recall", "f1_score"],
            "demand": ["mae", "rmse", "mape"],
        },
        "data_quality_checks": [
            "Validate delivery_date >= ship_date >= order_date",
            "Check for negative quantities",
            "Detect duplicate order IDs",
            "Validate carrier codes against known list",
            "Check lead_time distribution for outliers",
        ],
        "column_name_patterns": [
            r"order", r"shipment", r"delivery", r"carrier", r"sku",
            r"supply", r"inventory", r"warehouse", r"lead_time", r"freight",
        ],
    },

    # ── Insurance ────────────────────────────────────────────────────
    "insurance": {
        "name": "Insurance & Risk",
        "icon": "🛡️",
        "description": "Claims prediction, fraud detection, pricing, and risk modelling.",
        "use_cases": [
            "Claims severity prediction",
            "Fraud detection",
            "Premium pricing optimisation",
            "Churn prediction",
            "Risk segmentation",
            "Loss ratio forecasting",
        ],
        "target_column_hints": ["claims", "fraud", "severity", "premium", "loss",
                                  "risk_score", "cancelled"],
        "key_features": {
            "policy": ["premium", "coverage_amount", "deductible", "policy_type"],
            "claims": ["claim_amount", "claim_type", "claim_date", "n_prior_claims"],
            "policyholder": ["age", "gender", "years_with_company", "credit_score"],
        },
        "feature_engineering": [
            {"step": "loss_ratio",
             "description": "claims / premium = loss ratio (key actuarial metric)",
             "columns_needed": ["claims_paid", "premium"],
             "output_features": ["loss_ratio"]},
            {"step": "claim_frequency",
             "description": "Number of claims per year of policy",
             "columns_needed": ["n_claims", "policy_years"],
             "output_features": ["claim_frequency"]},
            {"step": "log_claim_amount",
             "description": "Log-transform claim amounts (right-skewed)",
             "columns_needed": ["claim_amount"],
             "output_features": ["log_claim_amount"]},
            {"step": "time_since_claim",
             "description": "Days since most recent claim",
             "columns_needed": ["last_claim_date"],
             "output_features": ["days_since_last_claim"]},
        ],
        "recommended_models": {
            "fraud": ["isolation_forest", "xgboost", "random_forest"],
            "severity": ["gradient_boosting", "tweedie_regression", "random_forest"],
            "churn": ["xgboost", "logistic_regression", "random_forest"],
        },
        "primary_metrics": {
            "fraud": ["roc_auc", "precision", "recall"],
            "severity": ["mae", "gini_coefficient", "rmse"],
        },
        "data_quality_checks": [
            "Check for class imbalance in fraud labels",
            "Validate claim amounts are non-negative",
            "Check policy end_date > start_date",
            "Detect duplicate claim IDs",
            "Validate premium values against coverage amounts",
        ],
        "column_name_patterns": [
            r"claim", r"premium", r"policy", r"loss", r"coverage",
            r"insurance", r"deductible", r"risk", r"underwrite", r"actuarial",
        ],
    },
}


# ════════════════════════════════════════════════════════════════════
#  Industry Templates Engine
# ════════════════════════════════════════════════════════════════════

class IndustryTemplates:
    """
    Manages industry-specific ML pipeline templates.
    """

    # ----------------------------------------------------------------
    #  List / Get
    # ----------------------------------------------------------------
    @staticmethod
    def list_templates() -> List[Dict[str, Any]]:
        """Return a lightweight summary of all available templates."""
        return [
            {
                "id": tid,
                "name": t["name"],
                "icon": t["icon"],
                "description": t["description"],
                "n_use_cases": len(t["use_cases"]),
                "n_feature_steps": len(t["feature_engineering"]),
            }
            for tid, t in _TEMPLATES.items()
        ]

    @staticmethod
    def get_template(industry_id: str) -> Dict[str, Any]:
        """Return full template by ID."""
        if industry_id not in _TEMPLATES:
            return {"error": f"Unknown industry: {industry_id}. "
                             f"Available: {list(_TEMPLATES.keys())}"}
        return {"industry_id": industry_id, **_TEMPLATES[industry_id]}

    @staticmethod
    def list_industries() -> List[str]:
        """Return all industry IDs."""
        return list(_TEMPLATES.keys())

    # ----------------------------------------------------------------
    #  Apply template to a DataFrame
    # ----------------------------------------------------------------
    @staticmethod
    def apply_template(
        df: pd.DataFrame,
        industry_id: str,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Match a DataFrame against an industry template and return
        actionable recommendations tailored to the actual columns.
        """
        start_time = time.time()

        if industry_id not in _TEMPLATES:
            return {"error": f"Unknown industry: {industry_id}"}

        template = _TEMPLATES[industry_id]
        df_cols = [c.lower() for c in df.columns]
        actual_cols = list(df.columns)

        # ── Identify matching feature categories ────────────────────
        matched_features: Dict[str, List[str]] = {}
        for category, col_hints in template.get("key_features", {}).items():
            found = [
                actual_col
                for actual_col, lower_col in zip(actual_cols, df_cols)
                if any(hint.lower() in lower_col for hint in col_hints)
            ]
            if found:
                matched_features[category] = found

        # ── Identify applicable feature engineering steps ────────────
        applicable_steps: List[Dict[str, Any]] = []
        for step in template["feature_engineering"]:
            needed = [c.lower().rstrip("_*") for c in step["columns_needed"]]
            found_cols = [
                actual_col
                for actual_col, lower_col in zip(actual_cols, df_cols)
                if any(n in lower_col for n in needed)
            ]
            completeness = round(len(found_cols) / max(len(needed), 1), 2)
            applicable_steps.append({
                **step,
                "columns_found": found_cols,
                "completeness": completeness,
                "applicable": completeness >= 0.5,
            })

        # Sort by completeness descending
        applicable_steps.sort(key=lambda x: x["completeness"], reverse=True)

        # ── Detect target column ────────────────────────────────────
        auto_target: Optional[str] = target_col
        if not auto_target:
            for hint in template.get("target_column_hints", []):
                for actual_col, lower_col in zip(actual_cols, df_cols):
                    if hint in lower_col:
                        auto_target = actual_col
                        break
                if auto_target:
                    break

        # ── Infer task type from target ──────────────────────────────
        task_type = None
        if auto_target and auto_target in df.columns:
            target_series = df[auto_target].dropna()
            n_unique = target_series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(target_series)
            task_type = "regression" if (is_numeric and n_unique > 20) else "classification"

        # ── Model recommendations for inferred task ──────────────────
        rec_models: Optional[List[str]] = None
        all_model_recs = template.get("recommended_models", {})
        if task_type:
            # Best-effort match
            for key, models in all_model_recs.items():
                if task_type in key or key in task_type:
                    rec_models = models
                    break
            if rec_models is None and all_model_recs:
                rec_models = list(all_model_recs.values())[0]

        # ── Data quality checks ──────────────────────────────────────
        quality_issues = IndustryTemplates._run_quality_checks(df, template, auto_target)

        # ── Column coverage score ────────────────────────────────────
        total_hints = sum(len(v) for v in template.get("key_features", {}).values())
        total_matched = sum(len(v) for v in matched_features.values())
        coverage_score = round(total_matched / max(total_hints, 1) * 100, 1)

        duration = round(time.time() - start_time, 3)
        return {
            "industry_id": industry_id,
            "template_name": template["name"],
            "dataset_rows": len(df),
            "dataset_cols": len(actual_cols),
            "coverage_score_pct": coverage_score,
            "matched_features": matched_features,
            "detected_target": auto_target,
            "inferred_task": task_type,
            "recommended_models": rec_models,
            "primary_metrics": template.get("primary_metrics", {}),
            "feature_engineering_steps": applicable_steps,
            "data_quality_checks": quality_issues,
            "use_cases": template["use_cases"],
            "duration_sec": duration,
        }

    # ----------------------------------------------------------------
    #  Auto-recommend industry
    # ----------------------------------------------------------------
    @staticmethod
    def recommend_industry(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Heuristically recommend the best-fit industry template
        by matching column names against each template's patterns.
        """
        start_time = time.time()
        col_names_str = " ".join(df.columns.tolist()).lower()

        scores: List[Dict[str, Any]] = []
        for tid, template in _TEMPLATES.items():
            patterns = template.get("column_name_patterns", [])
            match_count = sum(
                1 for p in patterns
                if re.search(p, col_names_str)
            )
            score = round(match_count / max(len(patterns), 1) * 100, 1)
            scores.append({
                "industry_id": tid,
                "name": template["name"],
                "icon": template["icon"],
                "match_score_pct": score,
                "patterns_matched": match_count,
                "total_patterns": len(patterns),
            })

        scores.sort(key=lambda x: x["match_score_pct"], reverse=True)
        best = scores[0] if scores else None

        return {
            "recommendations": scores,
            "best_match": best,
            "n_columns": len(df.columns),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _run_quality_checks(
        df: pd.DataFrame,
        template: Dict[str, Any],
        target_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run automated data-quality checks relevant to the template."""
        issues: List[Dict[str, Any]] = []

        # Duplicate rows
        n_dupl = int(df.duplicated().sum())
        if n_dupl > 0:
            issues.append({
                "check": "duplicate_rows",
                "severity": "warning",
                "message": f"{n_dupl} duplicate rows detected ({round(n_dupl/len(df)*100, 1)}%)",
            })

        # High missing rate
        for col in df.columns:
            missing_pct = float(df[col].isna().mean() * 100)
            if missing_pct > 30:
                issues.append({
                    "check": "high_missing",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                })

        # Class imbalance (classification targets)
        if target_col and target_col in df.columns:
            target = df[target_col].dropna()
            if target.nunique() <= 20 and len(target) > 0:
                vc = target.value_counts(normalize=True)
                min_class_pct = float(vc.min() * 100)
                if min_class_pct < 5:
                    issues.append({
                        "check": "class_imbalance",
                        "severity": "warning",
                        "column": target_col,
                        "message": (
                            f"Severe class imbalance: smallest class = {min_class_pct:.1f}%. "
                            "Consider oversampling (SMOTE) or class_weight='balanced'."
                        ),
                    })

        # Negative values in positive-only columns
        positive_keywords = ["price", "quantity", "amount", "salary", "revenue",
                              "age", "sqft", "weight", "distance"]
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            if any(k in col_lower for k in positive_keywords):
                n_neg = int((df[col] < 0).sum())
                if n_neg > 0:
                    issues.append({
                        "check": "negative_values",
                        "severity": "error",
                        "column": col,
                        "message": f"Column '{col}' has {n_neg} negative values (expected positive)",
                    })

        # Future dates
        for col in df.select_dtypes(include=["datetime"]).columns:
            n_future = int((df[col] > pd.Timestamp.now()).sum())
            if n_future > 0:
                issues.append({
                    "check": "future_dates",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column '{col}' has {n_future} future dates",
                })

        # Constant columns
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                issues.append({
                    "check": "constant_column",
                    "severity": "info",
                    "column": col,
                    "message": f"Column '{col}' is constant — can be dropped",
                })

        return issues
