"""
privacy.py — PII filtering module for NYC Taxi API
--------------------------------------------------

This module defines utilities to detect and remove Personally Identifiable
Information (PII) from incoming data before it reaches the model or logs.

We enforce the principle of:
- Data minimization
- No PII in logs
- No storage of raw inputs
"""

import re
import pandas as pd

# Define patterns for known PII fields
PII_FIELD_NAMES = {
    "name",
    "email",
    "phone",
    "phone_number",
    "address",
    "ssn",
    "passport",
    "license_number",
}

# Regex patterns for accidental PII leakage
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")


def scrub_pii_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes or masks known PII fields and PII-like patterns from the DataFrame.

    Strategy:
      - Drop common PII columns if present
      - Mask email/phone patterns inside string columns
    """
    df = df.copy()

    # Drop columns that are PII
    columns_to_drop = [col for col in df.columns if col.lower() in PII_FIELD_NAMES]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Mask email/phone from all string values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).apply(
                lambda x: EMAIL_REGEX.sub("[REDACTED_EMAIL]", x)
            )
            df[col] = df[col].apply(
                lambda x: PHONE_REGEX.sub("[REDACTED_PHONE]", x)
            )

    return df


def scrub_pii_from_record(record: dict) -> dict:
    """
    Scrubs PII from a single JSON-like dictionary (used in FastAPI /predict).
    """
    clean = {}
    for key, value in record.items():
        if key.lower() in PII_FIELD_NAMES:
            continue  # drop PII field completely

        if isinstance(value, str):
            # mask patterns
            value = EMAIL_REGEX.sub("[REDACTED_EMAIL]", value)
            value = PHONE_REGEX.sub("[REDACTED_PHONE]", value)

        clean[key] = value

    return clean
