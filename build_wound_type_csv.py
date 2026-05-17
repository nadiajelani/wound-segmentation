import os, csv

TEST_IMG_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images/"
OUTPUT_CSV   = "data/test_metadata.csv"

RULES = [
    (["dfuc", "dfu", "diabetic", "foot_ulcer", "footulcer"], "diabetic_foot_ulcer"),
    (["pressure", "pu_", "_pu", "sacral", "heel", "decub"], "pressure_injury"),
    (["surgical", "surgery", "post_op", "postop", "incision"], "surgical"),
]

def infer_type(filename):
    fn = filename.lower()
    for keywords, wtype in RULES:
        if any(kw in fn for kw in keywords):
            return wtype
    return "unknown"

exts = (".jpg", ".jpeg", ".png", ".bmp")
files = sorted(f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(exts))
os.makedirs("data", exist_ok=True)
rows = [{"file": f, "wound_type": infer_type(f)} for f in files]
counts = {}
for r in rows:
    wt = r["wound_type"]
    counts[wt] = counts.get(wt, 0) + 1

with open(OUTPUT_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["file", "wound_type"])
    writer.writeheader()
    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr    wr  ems():
    print(f"  {k}: {v}")
if counts.get("unknowif counts.get("unknowif counts.get("unknowif counts.get("unknowif counts.ges")
    pr    pr    pr    pr    pr    pr    pr    pr    pr    pr    pr    cal")
