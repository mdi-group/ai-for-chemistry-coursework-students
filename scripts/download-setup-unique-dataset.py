import pandas as pd
import numpy as np
import random
from uuid import uuid4
import os
import urllib.request
import sys


# === CONFIGURATION ===
base_url = "https://raw.githubusercontent.com/keeeto/reading-ml-chemistry/refs/heads/master/"  # replace with actual URL
files = {
    "coursework": "dielectric_coursework_dataset.pkl",
    "heldout": "dielectric_heldout_dataset.pkl"
}
save_dir = "data"  # local directory to store files

# === CHECK IF FILES ALREADY EXIST ===
all_exist = all(os.path.exists(os.path.join(save_dir, fname)) for fname in files.values())

if all_exist:
    print("✅ Files already exist in the local directory. No download needed.")
    sys.exit(0)

# === CREATE SAVE DIRECTORY IF NEEDED ===
os.makedirs(save_dir, exist_ok=True)

# === DOWNLOAD MISSING FILES ===
for label, filename in files.items():
    local_path = os.path.join(save_dir, filename)
    if os.path.exists(local_path):
        print(f"✔️  {filename} already exists — skipping.")
        continue

    url = base_url + filename
    print(f"⬇️  Downloading {filename} from {url} ...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"    Saved to: {local_path}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

        
# === LOAD INTO PANDAS ===

# Load base dataset
df = pd.read_pickle("data/dielectric_coursework_dataset.pkl")

# Convert 'slme' to numeric and coerce errors (e.g. "na" → NaN)
df["n"] = pd.to_numeric(df["n"], errors="coerce")

# Keep only rows where SLME is a real number (drop NaN)
df = df[df["n"].notnull()].reset_index(drop=True)

# Step 1: Sample 900–2000 rows
n_samples = random.randint(900, 2000)
df_sample = df.sample(n=n_samples, random_state=random.randint(1, 10000)).reset_index(drop=True)

# Step 2: Add 10–20 outliers (with slme > 100)
n_outliers = random.randint(10, 20)
outlier_indices = random.sample(range(len(df_sample)), n_outliers)
df_sample.loc[outlier_indices, "n"] = np.random.uniform(101, 150, size=n_outliers)

# Step 3: Add 10–20 duplicated rows
n_duplicates = random.randint(10, 20)
duplicate_rows = df_sample.sample(n=n_duplicates, random_state=random.randint(1, 10000))
df_augmented = pd.concat([df_sample, duplicate_rows], ignore_index=True)

# Optional: Shuffle the final dataset
df_augmented = df_augmented.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)

# Step 4: Save to file with a unique name
filename = f"student_dataset_{uuid4().hex[:8]}.pkl"
df_augmented.to_pickle(filename)

print(f"Custom student dataset saved to: {filename}")
print(f"Rows sampled: {n_samples}")
print(f"Final dataset shape: {df_augmented.shape}")
