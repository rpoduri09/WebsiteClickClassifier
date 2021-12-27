# project: p7
# submitter: rpoduri
# partner: none
# hours: 1.5
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
import pandas as pd

class UserPredictor:
    def __init__(self):
        self.pipeline = None
        self.xcols = ["past_purchase_amt", "seconds", "age", "lap_visits", "key_visits", "mon_visits", "tab_visits"]
    
    def fit(self, train_users, train_logs, train_y):
        sec_df = train_logs.groupby("user_id").sum()
        train_users = train_users.merge(sec_df, on="user_id", how="left").fillna(0)
        lap_df = train_logs.loc[(train_logs["url"] == "/laptop.html")]
        lap_df = pd.DataFrame(lap_df.groupby("user_id").size(), columns=["lap_visits"])
        train_users = train_users.join(lap_df["lap_visits"], on="user_id", how="left").fillna(0)
        key_df = train_logs.loc[(train_logs["url"] == "/keyboard.html")]
        key_df = pd.DataFrame(key_df.groupby("user_id").size(), columns=["key_visits"])
        train_users = train_users.join(key_df["key_visits"], on="user_id", how="left").fillna(0)
        monitor_df = train_logs.loc[(train_logs["url"] == "/monitor.html")]
        monitor_df = pd.DataFrame(monitor_df.groupby("user_id").size(), columns=["mon_visits"])
        train_users = train_users.join(monitor_df["mon_visits"], on="user_id", how="left").fillna(0)
        tab_df = train_logs.loc[(train_logs["url"] == "/tablet.html")]
        tab_df = pd.DataFrame(tab_df.groupby("user_id").size(), columns=["tab_visits"])
        train_users = train_users.join(tab_df["tab_visits"], on="user_id", how="left").fillna(0)
        self.pipeline = Pipeline([
            ("pf", PolynomialFeatures(degree=3, include_bias=False)),
            ("std", StandardScaler()),
            ("lr", LogisticRegression(max_iter = 400))
        ])
        self.pipeline.fit(train_users[self.xcols], train_y["y"])


    def predict(self, test_users, test_logs):
        sec_df = test_logs.groupby("user_id").sum()
        test_users = test_users.merge(sec_df, on="user_id", how="left").fillna(0)
        lap_df = test_logs.loc[(test_logs["url"] == "/laptop.html")]
        lap_df = pd.DataFrame(lap_df.groupby("user_id").size(), columns=["lap_visits"])
        test_users = test_users.join(lap_df["lap_visits"], on="user_id", how="left").fillna(0)
        key_df = test_logs.loc[(test_logs["url"] == "/keyboard.html")]
        key_df = pd.DataFrame(key_df.groupby("user_id").size(), columns=["key_visits"])
        test_users = test_users.join(key_df["key_visits"], on="user_id", how="left").fillna(0)
        monitor_df = test_logs.loc[(test_logs["url"] == "/monitor.html")]
        monitor_df = pd.DataFrame(monitor_df.groupby("user_id").size(), columns=["mon_visits"])
        test_users = test_users.join(monitor_df["mon_visits"], on="user_id", how="left").fillna(0)
        tab_df = test_logs.loc[(test_logs["url"] == "/tablet.html")]
        tab_df = pd.DataFrame(tab_df.groupby("user_id").size(), columns=["tab_visits"])
        test_users = test_users.join(tab_df["tab_visits"], on="user_id", how="left").fillna(0)
        return self.pipeline.predict(test_users[self.xcols])   
