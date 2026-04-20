import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("mobile_price_preprocessing.csv")

X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
input_example = X_train[0:5]

mlflow.set_experiment("mobile_price_classification")

with mlflow.start_run(run_name="RandomForest_autolog"):
    n_estimators = 1000
    max_depth = 25
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    mlflow.sklearn.log_model(
        sk_model=model, artifact_path="model", input_example=input_example
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
