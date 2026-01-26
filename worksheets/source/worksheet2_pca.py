import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1) Load the CSV
# ---------------------------
df = pd.read_csv("../ws2_india.csv")

scheduled_cols = [
    "Assamese","Bengali","Bodo","Dogri","Gujarati","Hindi","Kannada","Kashmiri","Konkani",
    "Maithili","Malayalam","Manipuri","Marathi","Nepali","Odia","Punjabi","Sanskrit",
    "Santali","Sindhi","Tamil","Telugu","Urdu"
]

missing = [c for c in scheduled_cols if c not in df.columns]
if missing:
    raise ValueError(f"f/pcolumns: {missing}")

X = df[scheduled_cols].to_numpy(dtype=float)


mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)
sd[sd == 0] = 1.0
#Z = (X - mu) / sd #corr
Z= (X - mu)  #cov
R = np.cov(Z, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(R)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

scores = Z @ eigvecs  # rows = regions, cols = PCs

out = pd.DataFrame({
    "Region": df["Region"].astype(str),
    "PC1": scores[:, 0],
    "PC2": scores[:, 1],
    "PC3": scores[:, 2],
})


pd.set_option("display.max_rows", None)
pd.set_option("display.width", 140)
print(out.to_string(index=False))

x="PC1"
#y="PC2"
y="PC3"

plt.figure()
plt.scatter(out[x], out[y])
for i, r in enumerate(out["Region"]):
    plt.text(out[x].iat[i], out[y].iat[i], r, fontsize=8)
plt.xlabel(x)
plt.ylabel(y)

plt.tight_layout()
plt.show()
