import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student-mat.csv", sep=";")

df["avg_grade"] = df[["G1","G2","G3"]].mean(axis=1)
df["grade_trend"] = df["G3"] - df["G1"]
df["variance"] = df[["G1","G2","G3"]].var(axis=1)
df["consistency"] = 1 / (1 + df["variance"])
df["efficiency"] = df["avg_grade"] / (df["studytime"] + 1)

def label_student(row):
    if row["variance"] > 10:
        return "inconsistent"
    elif row["avg_grade"] < 10 and row["studytime"] > 5:
        return "struggler"
    elif row["avg_grade"] > 15 and row["studytime"] < 3:
        return "efficient"
    elif row["grade_trend"] > 5:
        return "deep_learner"
    else:
        return "explorer"

df["learner_type"] = df.apply(label_student, axis=1)

features = ["avg_grade", "grade_trend", "variance", "consistency", "efficiency", "studytime", "failures"]
X = df[features]

le = LabelEncoder()
y = le.fit_transform(df["learner_type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained")

def predict_student(data):
    df_input = pd.DataFrame([data])
    pred = model.predict(df_input)
    return le.inverse_transform(pred)[0]