import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("ecommerce_sales_data.csv")

data["Order Date"] = pd.to_datetime(data["Order Date"])
data["Year"] = data["Order Date"].dt.year
data["Month"] = data["Order Date"].dt.month
data = data.drop(columns=["Order Date"])

data = data.fillna(method="ffill")

target_column = "Sales"
X = data.drop(columns=[target_column])
y = data[target_column]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(24, 12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    max_depth=3
)
plt.title("Decision Tree Regression (Top Levels)")
plt.show()

