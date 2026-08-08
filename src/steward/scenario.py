"""The demo company's data estate, defined once.

This single definition feeds both the DataHub seeder and the in-memory fake, so
the live demo and the test suite are provably looking at the same graph.

The shape is a deliberately ordinary analytics stack — raw ingestion, dbt
models, a feature table, dashboards, and two ML models — because the point being
demonstrated is that ordinary stacks hide non-obvious blast radius. The
interesting property of this graph: `raw.stripe.charges.amount_cents` looks like
a leaf-level payments column, but four hops downstream it feeds the production
churn model. Nobody reading the raw table would know that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Column


def dataset_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{platform},{name},{env})"


def dashboard_urn(tool: str, dash_id: str) -> str:
    return f"urn:li:dashboard:({tool},{dash_id})"


def mlmodel_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:mlModel:(urn:li:dataPlatform:{platform},{name},{env})"


def datajob_urn(flow: str, job: str) -> str:
    return (
        f"urn:li:dataJob:(urn:li:dataFlow:(airflow,{flow},PROD),{job})"
    )


def user_urn(name: str) -> str:
    return f"urn:li:corpuser:{name}"


@dataclass
class SeedEntity:
    urn: str
    entity_type: str
    name: str
    platform: str | None = None
    description: str | None = None
    owners: list[str] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    #: URNs this entity reads from.
    upstreams: list[str] = field(default_factory=list)
    #: ML models reach their training data through a training job rather than a
    #: direct dataset edge — that is how DataHub models it, and how real ML
    #: lineage actually works.
    training_jobs: list[str] = field(default_factory=list)


# -- raw layer -------------------------------------------------------------

RAW_CHARGES = dataset_urn("snowflake", "raw.stripe.charges")
RAW_CHECKOUT = dataset_urn("snowflake", "raw.events.checkout_events")
RAW_CUSTOMERS = dataset_urn("snowflake", "raw.crm.customers")

# -- modelled layer --------------------------------------------------------

DIM_CUSTOMER = dataset_urn("snowflake", "analytics.core.dim_customer")
FCT_ORDERS = dataset_urn("snowflake", "analytics.core.fct_orders")
CUSTOMER_LTV = dataset_urn("snowflake", "analytics.marts.customer_ltv")
CUSTOMER_FEATURES = dataset_urn("snowflake", "analytics.features.customer_features")

# -- orchestration ---------------------------------------------------------

JOB_ORDERS = datajob_urn("daily_analytics", "build_fct_orders")
JOB_FEATURES = datajob_urn("ml_features", "build_customer_features")
JOB_TRAIN_CHURN = datajob_urn("ml_training", "train_churn_predictor")
JOB_TRAIN_LTV = datajob_urn("ml_training", "train_ltv_regressor")

# -- consumption -----------------------------------------------------------

DASH_REVENUE = dashboard_urn("looker", "executive_revenue_overview")
DASH_CHURN = dashboard_urn("looker", "churn_monitoring")

MODEL_CHURN = mlmodel_urn("mlflow", "churn_predictor")
MODEL_LTV = mlmodel_urn("mlflow", "ltv_regressor")


ENTITIES: list[SeedEntity] = [
    SeedEntity(
        urn=RAW_CHARGES,
        entity_type="DATASET",
        name="raw.stripe.charges",
        platform="snowflake",
        description="Raw Stripe charge events, landed by Fivetran every 15 minutes.",
        owners=[user_urn("payments_team")],
        columns=[
            Column("charge_id", "VARCHAR", "Stripe charge identifier.", nullable=False),
            Column("customer_id", "VARCHAR", "Stripe customer identifier.", nullable=False),
            # The column the whole demo turns on.
            Column(
                "amount_cents",
                "NUMBER(38,0)",
                "Charge amount in the smallest currency unit.",
                nullable=False,
            ),
            Column("currency", "VARCHAR", "ISO-4217 currency code.", nullable=False),
            Column("status", "VARCHAR", "succeeded | pending | failed", nullable=False),
            Column("created_at", "TIMESTAMP_NTZ", "Charge creation time, UTC.", nullable=False),
        ],
    ),
    SeedEntity(
        urn=RAW_CHECKOUT,
        entity_type="DATASET",
        name="raw.events.checkout_events",
        platform="snowflake",
        description="Client-side checkout funnel events.",
        owners=[user_urn("growth_eng")],
        columns=[
            Column("event_id", "VARCHAR", nullable=False),
            Column("session_id", "VARCHAR", nullable=False),
            Column("customer_id", "VARCHAR"),
            Column("step", "VARCHAR", "Funnel step name."),
            Column("occurred_at", "TIMESTAMP_NTZ", nullable=False),
        ],
    ),
    SeedEntity(
        urn=RAW_CUSTOMERS,
        entity_type="DATASET",
        name="raw.crm.customers",
        platform="snowflake",
        description="Customer records synced from the CRM.",
        owners=[user_urn("crm_ops")],
        columns=[
            Column("customer_id", "VARCHAR", nullable=False),
            Column("email", "VARCHAR", "Primary contact email. PII."),
            Column("signup_date", "DATE", nullable=False),
            Column("plan_tier", "VARCHAR", "free | pro | enterprise"),
        ],
    ),
    SeedEntity(
        urn=DIM_CUSTOMER,
        entity_type="DATASET",
        name="analytics.core.dim_customer",
        platform="snowflake",
        description="Conformed customer dimension.",
        owners=[user_urn("analytics_eng")],
        upstreams=[RAW_CUSTOMERS],
        columns=[
            Column("customer_key", "VARCHAR", nullable=False),
            Column("plan_tier", "VARCHAR"),
            Column("signup_date", "DATE"),
        ],
    ),
    SeedEntity(
        urn=FCT_ORDERS,
        entity_type="DATASET",
        name="analytics.core.fct_orders",
        platform="snowflake",
        description="One row per completed order, with revenue in USD.",
        owners=[user_urn("analytics_eng")],
        upstreams=[RAW_CHARGES, RAW_CHECKOUT],
        columns=[
            Column("order_id", "VARCHAR", nullable=False),
            Column("customer_key", "VARCHAR", nullable=False),
            # Derived directly from amount_cents.
            Column("revenue_usd", "NUMBER(38,2)", "Order revenue converted to USD."),
            Column("ordered_at", "TIMESTAMP_NTZ", nullable=False),
        ],
    ),
    SeedEntity(
        urn=CUSTOMER_LTV,
        entity_type="DATASET",
        name="analytics.marts.customer_ltv",
        platform="snowflake",
        description="Rolling 12-month lifetime value per customer.",
        owners=[user_urn("analytics_eng")],
        upstreams=[FCT_ORDERS, DIM_CUSTOMER],
        columns=[
            Column("customer_key", "VARCHAR", nullable=False),
            Column("ltv_usd", "NUMBER(38,2)", "Trailing 12-month revenue."),
            Column("orders_count", "NUMBER(38,0)"),
        ],
    ),
    SeedEntity(
        urn=CUSTOMER_FEATURES,
        entity_type="DATASET",
        name="analytics.features.customer_features",
        platform="snowflake",
        description="Feature table serving the churn and LTV models.",
        owners=[user_urn("ml_platform")],
        upstreams=[CUSTOMER_LTV, FCT_ORDERS],
        columns=[
            Column("customer_key", "VARCHAR", nullable=False),
            Column("ltv_usd", "NUMBER(38,2)", "Sourced from customer_ltv."),
            Column("orders_last_30d", "NUMBER(38,0)"),
            Column("avg_order_value_usd", "NUMBER(38,2)", "Derived from revenue_usd."),
        ],
    ),
    SeedEntity(
        urn=JOB_ORDERS,
        entity_type="DATA_JOB",
        name="build_fct_orders",
        description="Airflow task materialising fct_orders each morning.",
        owners=[user_urn("analytics_eng")],
        upstreams=[RAW_CHARGES, RAW_CHECKOUT],
    ),
    SeedEntity(
        urn=JOB_FEATURES,
        entity_type="DATA_JOB",
        name="build_customer_features",
        description="Airflow task refreshing the ML feature table hourly.",
        owners=[user_urn("ml_platform")],
        upstreams=[CUSTOMER_LTV],
    ),
    SeedEntity(
        urn=DASH_REVENUE,
        entity_type="DASHBOARD",
        name="Executive Revenue Overview",
        platform="looker",
        description="Board-level revenue reporting. Reviewed weekly by the exec team.",
        owners=[user_urn("bi_team")],
        upstreams=[FCT_ORDERS, CUSTOMER_LTV],
    ),
    SeedEntity(
        urn=DASH_CHURN,
        entity_type="DASHBOARD",
        name="Churn Monitoring",
        platform="looker",
        description="Tracks churn model scores against actual churn.",
        owners=[user_urn("bi_team")],
        upstreams=[CUSTOMER_FEATURES],
    ),
    SeedEntity(
        urn=JOB_TRAIN_CHURN,
        entity_type="DATA_JOB",
        name="train_churn_predictor",
        description="Weekly retraining run for the production churn model.",
        owners=[user_urn("ml_platform")],
        upstreams=[CUSTOMER_FEATURES],
    ),
    SeedEntity(
        urn=JOB_TRAIN_LTV,
        entity_type="DATA_JOB",
        name="train_ltv_regressor",
        description="Ad-hoc training run for the staging LTV regressor.",
        owners=[user_urn("ml_platform")],
        upstreams=[CUSTOMER_LTV],
    ),
    SeedEntity(
        urn=MODEL_CHURN,
        entity_type="MLMODEL",
        name="churn_predictor",
        platform="mlflow",
        description=(
            "Production churn model, v3. Scores every active customer nightly; "
            "output drives the retention discount campaign."
        ),
        owners=[user_urn("ml_platform")],
        training_jobs=[JOB_TRAIN_CHURN],
    ),
    SeedEntity(
        urn=MODEL_LTV,
        entity_type="MLMODEL",
        name="ltv_regressor",
        platform="mlflow",
        description="Staging LTV regressor, not yet serving traffic.",
        owners=[user_urn("ml_platform")],
        training_jobs=[JOB_TRAIN_LTV],
    ),
]


BY_URN = {entity.urn: entity for entity in ENTITIES}


#: The change the demo asks about. Realistic, small-looking, and four hops from
#: a production model.
DEMO_CHANGE = (
    "We want to rename raw.stripe.charges.amount_cents to amount_minor_units "
    "and change its type from NUMBER to VARCHAR to support zero-decimal "
    "currencies. What breaks?"
)
