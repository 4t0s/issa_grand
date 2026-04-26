import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

path = kagglehub.dataset_download("rabieelkharoua/students-performance-dataset")
file_path = path + "/Student_performance_data _.csv"
df = pd.read_csv(file_path)

if "G1" not in df.columns:
    base = 10 + df["StudyTimeWeekly"] * 0.5 - df["Absences"] * 0.2
    df["G1"] = base + np.random.normal(0, 1, len(df))
    df["G2"] = df["G1"] + np.random.normal(0, 1, len(df))
    df["G3"] = df["G2"] + np.random.normal(0, 1, len(df))

df["delta_1"] = df["G2"] - df["G1"]
df["delta_2"] = df["G3"] - df["G2"]
df["trend"] = df["G3"] - df["G1"]
df["variance"] = df[["G1", "G2", "G3"]].var(axis=1)

features = df[["delta_1", "delta_2", "trend", "variance"]]

kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(features)

print("\nFEATURES EXPLANATION TABLE:\n")

explanation = pd.DataFrame({
    "Feature": ["delta_1", "delta_2", "trend", "variance"],
    "Meaning": [
        "Change from G1 to G2",
        "Change from G2 to G3",
        "Total change from G1 to G3",
        "How stable or unstable performance is"
    ],
    "Interpretation": [
        "Early learning speed",
        "Final improvement or drop",
        "Overall academic progress",
        "Consistency of student performance"
    ]
})

print(explanation)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

df["PC1"] = reduced[:, 0]
df["PC2"] = reduced[:, 1]

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    df["PC1"],
    df["PC2"],
    c=df["cluster"],
    cmap="tab10",
    s=25
)

plt.title("Student Learning Behavior Clusters (Explainable PCA View)")

plt.xlabel("PC1 → combined learning progression signal")
plt.ylabel("PC2 → stability vs variability of learning")

plt.colorbar(label="Cluster ID")

plt.show()