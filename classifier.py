from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from data_preprocessor import DataPreProcess

dpp = DataPreProcess()

df = dpp.remove_unneeded_columns()
df = dpp.remove_rows_missing_sex(df)
df = dpp.cast_float_columns(df)
df = dpp.fill_nan_with_mean(df)
df = dpp.true_false_to_zero_one(df)
df = dpp.sex_to_zero_one(df)
df = dpp.classes_to_float(df)
df = dpp.remove_third_class_only_present_twice(df)

labels = df["Class"]
del df["Class"]

df_train, df_test, train_labels, test_labels = train_test_split(df, labels, test_size=0.5, random_state=0)

classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0).fit(df_train, train_labels)
pred_res = classifier.predict(df_test)

precision, recall, fbeta, _ = precision_recall_fscore_support(test_labels, pred_res)

print("Accuracy:", classifier.score(df_test, test_labels))
for i in range(3):
    print("Class {}: Precision: {:.3f}, Recall: {:.3f}, F-beta: {:.3f}".format(i, precision[i],recall[i],fbeta[i]))