"""
Generates a synthetic rainfall/weather dataset for the Rising Waters
flood-prediction model.

This is sample data meant to demonstrate the pipeline end-to-end.
Replace with real historical rainfall/weather records for production use
(e.g. IMD / data.gov.in rainfall datasets).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000


def generate_row():
    rainfall_mm = np.random.gamma(shape=2.0, scale=25)          # 0 - 300ish mm
    temperature_c = np.random.normal(27, 4)                     # deg C
    humidity_pct = np.clip(np.random.normal(70, 15), 20, 100)   # %
    wind_speed_kmph = np.clip(np.random.normal(15, 8), 0, 80)
    river_level_m = np.clip(np.random.normal(3, 1.2), 0, 12)
    soil_saturation_pct = np.clip(np.random.normal(50, 20), 0, 100)

    # simple synthetic risk rule with noise, so the label correlates with
    # the features but isn't a trivial linear function of any single one
    risk_score = (
        0.035 * rainfall_mm
        + 0.02 * humidity_pct
        + 0.4 * river_level_m
        + 0.015 * soil_saturation_pct
        - 0.05 * temperature_c
        + np.random.normal(0, 1.5)
    )
    flood_risk = int(risk_score > 5.5)

    return {
        "rainfall_mm": round(rainfall_mm, 2),
        "temperature_c": round(temperature_c, 2),
        "humidity_pct": round(humidity_pct, 2),
        "wind_speed_kmph": round(wind_speed_kmph, 2),
        "river_level_m": round(river_level_m, 2),
        "soil_saturation_pct": round(soil_saturation_pct, 2),
        "flood_risk": flood_risk,
    }


def main():
    rows = [generate_row() for _ in range(N)]
    df = pd.DataFrame(rows)
    df.to_csv("data/weather_data.csv", index=False)
    print(f"Wrote data/weather_data.csv with {len(df)} rows")
    print(df["flood_risk"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
