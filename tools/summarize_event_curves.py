"""Read-only summary of every event-study curve in reports/event_study/.

For each (cause -> target, direction) it prints:
  * events            number of events
  * sig_fdr_hours     horizons (hours) where the event CAR differs from random
                      times after FDR correction -- pure statistical evidence,
                      WITHOUT the 1% economic-size filter used for 'onset'
  * first_sig         first such horizon
  * CAR@1h..48h       mean cumulative log return of the target after events

Run from the project root:  python -m tools.summarize_event_curves
"""
import glob
import os

import pandas as pd

HORIZONS = [1, 4, 12, 24, 48]

rows = []
for path in sorted(glob.glob(os.path.join("reports", "event_study", "*_curve.csv"))):
    name = os.path.basename(path)[: -len("_curve.csv")]
    prefix, direction = name.rsplit("_", 1)
    df = pd.read_csv(path)
    sig = df[df["significant_fdr"].astype(bool)]["horizon_h"].tolist()
    car = df.set_index("horizon_h")["mean_car"]
    rows.append({
        "pair": prefix, "dir": direction[:3], "events": int(df["n_events"].max()),
        "sig_fdr_hours": len(sig), "first_sig": sig[0] if sig else "-",
        **{f"CAR@{h}h": f"{100 * car.get(h, float('nan')):+.2f}%" for h in HORIZONS},
    })

if not rows:
    print("No *_curve.csv files found in reports/event_study/")
else:
    print(pd.DataFrame(rows).to_string(index=False))
