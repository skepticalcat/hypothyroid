import pandas as pd
import numpy as np

class DataPreProcess:

    def __init__(self):
        self.raw_df = pd.read_csv("dataset_57_hypothyroid.csv")
        pd.set_option("display.precision", 2)
        pd.set_option("display.max.columns", None)

    def explore(self, df):
        print(df.describe())
        #for column in df:
        #    print("missing in",column,len(df[df[column] == "?"]))
        #for column in df:
        #    print(column, max(df[column]))
        print(len(df[df["Class"] == 3]))

    def remove_unneeded_columns(self):
        for elem in ["TBG_measured", "TBG", "referral_source", "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured"]:
            del self.raw_df[elem]
        return self.raw_df

    def remove_rows_missing_sex(self, df):
        return df[df["sex"] != "?"]

    def remove_third_class_only_present_twice(self, df):
        return df[df["Class"] != 3]

    def cast_float_columns(self, df):
        df = df.replace(to_replace="?", value=np.nan)
        for elem in ["age", "TSH", "T3", "TT4", "T4U", "FTI"]:
            df[elem] = df[elem].astype(float)
        return df

    def fill_nan_with_mean(self, df):
        df = df.fillna(df.mean())
        return df

    def true_false_to_zero_one(self, df):
        boolmap = {'t': True, 'f': False}
        for elem in ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
                     "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
                     "goitre", "tumor", "hypopituitary", "psych"]:
            df[elem] = df[elem].map(boolmap)
            df[elem] = df[elem].astype(int)
        return df

    def sex_to_zero_one(self, df):
        boolmap = {'F': True, 'M': False}
        df["sex"] = df["sex"].map(boolmap)
        df["sex"] = df["sex"].astype(int)
        return df

    def classes_to_float(self, df):
        boolmap = {"negative": 0, "primary_hypothyroid": 1, "compensated_hypothyroid":2, "secondary_hypothyroid":3}
        df["Class"] = df["Class"].map(boolmap)
        df["Class"] = df["Class"].astype(int)
        return df