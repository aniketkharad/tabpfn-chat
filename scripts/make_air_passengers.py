"""Generate the canonical Air Passengers time-series dataset (144 rows, 1949-1960)."""

from pathlib import Path
import pandas as pd

PASSENGERS = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]

def generate():
    records = []
    idx = 0
    for year in range(1949, 1961):
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}-01"
            records.append({
                "date": date_str,
                "year": year,
                "month_num": month,
                "passengers": PASSENGERS[idx],
            })
            idx += 1

    df = pd.DataFrame(records)
    
    # Save to demos directory
    out_dir = Path("demos/03_time_series_forecasting")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "air_passengers.csv", index=False)
    print(f"Saved {len(df)} rows to {out_dir / 'air_passengers.csv'}")

    # Also save to static samples for browser UI
    static_samples = Path("src/tabchat/frontend/static/samples")
    static_samples.mkdir(parents=True, exist_ok=True)
    df.to_csv(static_samples / "air_passengers.csv", index=False)

if __name__ == "__main__":
    generate()
