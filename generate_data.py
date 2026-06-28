"""
Generate a synthetic Detroit housing dataset.

This dataset is constructed to mirror the structure, feature ranges, and
price dynamics described in the City of Detroit Open Data Portal's
"Property Sales" dataset (data.detroitmi.gov) and 2025-2026 Detroit
residential market reporting (median sale price ~$95,000-$105,000 citywide
in 2025, with strong neighborhood-level dispersion from ~$25,000 in
high-vacancy east-side tracts to $400,000+ in Palmer Woods, Indian Village,
Boston-Edison, and Corktown). It is SYNTHETIC data generated for this
academic assignment, not a direct download of city records -- this keeps
the notebook fully reproducible offline while preserving realistic,
explainable relationships between features and price for modeling purposes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

neighborhoods = {
    # name: (base_price, premium_std, vacancy_rate, school_score_mean)
    "Palmer Woods":        (385000, 70000, 0.03, 7.8),
    "Indian Village":      (340000, 65000, 0.04, 7.2),
    "Boston-Edison":       (260000, 55000, 0.05, 6.8),
    "Corktown":            (295000, 60000, 0.04, 6.5),
    "West Village":        (245000, 50000, 0.06, 6.6),
    "Midtown":             (210000, 45000, 0.05, 6.9),
    "Lafayette Park":      (190000, 40000, 0.07, 6.3),
    "Bagley":              (135000, 30000, 0.10, 5.8),
    "Grandmont-Rosedale":  (120000, 28000, 0.11, 5.6),
    "University District": (175000, 35000, 0.08, 6.4),
    "Sherwood Forest":     (165000, 32000, 0.09, 6.1),
    "Jefferson-Chalmers":  (95000, 25000, 0.15, 5.0),
    "Morningside":         (78000, 20000, 0.18, 4.6),
    "East English Village":(88000, 22000, 0.16, 4.8),
    "Brightmoor":          (42000, 14000, 0.32, 3.6),
    "Chadsey-Condon":      (38000, 12000, 0.35, 3.4),
    "Warrendale":          (60000, 16000, 0.24, 4.2),
    "Cody Rouge":          (55000, 15000, 0.26, 4.0),
    "Russell Woods":       (70000, 18000, 0.22, 4.4),
    "Greenfield":          (66000, 17000, 0.23, 4.3),
}

rows = []
for i in range(N):
    nb = np.random.choice(list(neighborhoods.keys()))
    base_price, premium_std, vacancy_rate, school_mean = neighborhoods[nb]

    sqft = np.clip(np.random.normal(1350, 480), 480, 4200)
    bedrooms = np.clip(np.round(np.random.normal(3, 0.9)), 1, 7)
    bathrooms = np.clip(np.round(np.random.normal(1.5, 0.6) * 2) / 2, 1, 5)
    lot_size = np.clip(np.random.normal(4800, 1800), 1200, 15000)
    year_built = int(np.clip(np.random.normal(1948, 22), 1890, 2024))
    age = 2026 - year_built

    stories = np.random.choice([1, 1.5, 2, 2.5], p=[0.45, 0.15, 0.35, 0.05])
    garage = np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
    has_basement = np.random.choice([0, 1], p=[0.25, 0.75])
    has_porch = np.random.choice([0, 1], p=[0.3, 0.7])
    renovated = np.random.choice([0, 1], p=[0.78, 0.22])
    fireplace = np.random.choice([0, 1], p=[0.7, 0.3])

    condition_score = np.clip(
        np.random.normal(6.5 - vacancy_rate * 8, 1.4), 1, 10
    )
    distance_to_downtown_km = np.clip(np.random.normal(9, 4.5), 0.5, 28)
    crime_index = np.clip(
        np.random.normal(45 + vacancy_rate * 120, 15), 5, 100
    )
    school_rating = np.clip(np.random.normal(school_mean, 1.1), 1, 10)
    walk_score = np.clip(np.random.normal(55 - distance_to_downtown_km * 0.8, 12), 5, 98)
    property_tax_annual = np.round(np.clip(np.random.normal(1800, 700), 300, 6000), 0)

    vacant_lot_nearby = np.random.choice([0, 1], p=[1 - vacancy_rate, vacancy_rate])

    # Price model: base neighborhood price + feature effects + noise
    price = base_price
    price += (sqft - 1350) * 95
    price += (bedrooms - 3) * 6500
    price += (bathrooms - 1.5) * 9000
    price += (lot_size - 4800) * 3.2
    price -= age * 280
    price += (stories - 1) * 8000
    price += garage * 7000
    price += has_basement * 9000
    price += has_porch * 3500
    price += renovated * 28000
    price += fireplace * 6000
    price += (condition_score - 6.5) * 9500
    price -= distance_to_downtown_km * 1800
    price -= crime_index * 450
    price += (school_rating - 6) * 7000
    price += (walk_score - 55) * 600
    price -= vacant_lot_nearby * 12000
    price += np.random.normal(0, premium_std * 0.6)
    price = max(price, 12000)  # Detroit has a real floor near land-value-only sales

    rows.append({
        "neighborhood": nb,
        "sqft": round(sqft, 0),
        "bedrooms": int(bedrooms),
        "bathrooms": bathrooms,
        "lot_size_sqft": round(lot_size, 0),
        "year_built": year_built,
        "stories": stories,
        "garage_spaces": garage,
        "has_basement": has_basement,
        "has_porch": has_porch,
        "renovated_last_10yrs": renovated,
        "fireplace": fireplace,
        "condition_score": round(condition_score, 1),
        "distance_to_downtown_km": round(distance_to_downtown_km, 2),
        "crime_index": round(crime_index, 1),
        "school_rating": round(school_rating, 1),
        "walk_score": round(walk_score, 1),
        "property_tax_annual": property_tax_annual,
        "vacant_lot_nearby": vacant_lot_nearby,
        "sale_price": round(price, 0),
    })

df = pd.DataFrame(rows)

# Inject some realistic missingness (common in real property records)
for col, frac in [("school_rating", 0.04), ("walk_score", 0.03),
                   ("condition_score", 0.05), ("garage_spaces", 0.02)]:
    idx = df.sample(frac=frac, random_state=1).index
    df.loc[idx, col] = np.nan

# Inject a few outliers (data entry errors, common in county/city records)
outlier_idx = df.sample(n=8, random_state=2).index
df.loc[outlier_idx, "sqft"] = df.loc[outlier_idx, "sqft"] * 0.05  # bad entries

df.to_csv("/home/claude/detroit_house_project/detroit_housing.csv", index=False)
print(df.shape)
print(df.head())
print(df.isna().sum())
