import pandas as pd
import numpy as np
import kagglehub
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
df["variance"] = df[["G1","G2","G3"]].var(axis=1)

features = df[["delta_1", "delta_2", "trend", "variance"]]

kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(features)

def map_cluster(row):
    if row["trend"] > 3:
        return "rising"
    elif row["trend"] < -3:
        return "declining"
    elif row["variance"] < 1:
        return "consistent"
    elif row["variance"] > 5:
        return "volatile"
    else:
        return "moderate"

df["learner_type"] = df.apply(map_cluster, axis=1)

X = features
y = df["cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_student(g1, g2, g3):
    delta_1 = g2 - g1
    delta_2 = g3 - g2
    trend = g3 - g1
    variance = np.var([g1, g2, g3])
    x = [[delta_1, delta_2, trend, variance]]
    cluster = model.predict(x)[0]
    temp = {
        "delta_1": delta_1,
        "delta_2": delta_2,
        "trend": trend,
        "variance": variance
    }
    learner_type = map_cluster(temp)
    return cluster, learner_type

example = predict_student(10, 12, 16)
print(example)