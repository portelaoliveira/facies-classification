import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import set_option
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from functions import (
    accuracy,
    accuracy_adjacent,
    compare_facies_plot,
    label_facies,
    make_facies_log_plot,
    plot_confusion_matrix,
)

set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

filename = "data/training_data.csv"
training_data = pd.read_csv(filename)
# print(training_data)


training_data["Well Name"] = training_data["Well Name"].astype("category")
training_data["Formation"] = training_data["Formation"].astype("category")
training_data["Well Name"].unique()
# print(training_data.describe())

blind = training_data[training_data["Well Name"] == "SHANKLE"]
training_data = training_data[training_data["Well Name"] != "SHANKLE"]

# 1=sandstone  2=c_siltstone   3=f_siltstone
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = [
    "#F4D03F",
    "#F5B041",
    "#DC7633",
    "#6E2C00",
    "#1B4F72",
    "#2E86C1",
    "#AED6F1",
    "#A569BD",
    "#196F3D",
]

facies_labels = ["SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS"]
# facies_color_map is a dictionary that maps facies labels
# to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]


training_data.loc[:, "FaciesLabels"] = training_data.apply(
    lambda row: label_facies(row, facies_labels), axis=1
)

make_facies_log_plot(
    training_data[training_data["Well Name"] == "SHRIMPLIN"],
    facies_colors,
    "wells",
)

# count the number of unique entries for each facies, sort them by
# facies number (instead of by number of entries)
facies_counts = training_data["Facies"].value_counts().sort_index()
# use facies labels to index each count
facies_counts.index = facies_labels

facies_counts.plot(
    kind="bar",
    color=facies_colors,
    title="Distribution of Training Data by Facies",
)
plt.savefig(f"imgs/distribution_of_training_data_by_facies.png")
plt.close()
# print(facies_counts)

inline_rc = dict(mpl.rcParams)
sns.set()
sns.pairplot(
    training_data.drop(
        ["Well Name", "Facies", "Formation", "Depth", "NM_M", "RELPOS"], axis=1
    ),
    hue="FaciesLabels",
    palette=facies_color_map,
    hue_order=list(reversed(facies_labels)),
)

# switch back to default matplotlib plot style
mpl.rcParams.update(inline_rc)
plt.savefig(f"imgs/wellsFacies.png")
plt.close()

correct_facies_labels = training_data["Facies"].values

feature_vectors = training_data.drop(
    ["Formation", "Well Name", "Depth", "Facies", "FaciesLabels"], axis=1
)
# print(feature_vectors.describe())

scaler = preprocessing.StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, correct_facies_labels, test_size=0.25, random_state=42
)

clf = svm.SVC()
clf.fit(X_train, y_train)
predicted_labels = clf.predict(X_test)

classes = {
    0: "SS",
    1: "CSiS",
    2: "FSiS",
    3: "SiSh",
    4: "MS",
    5: "WS",
    6: "D",
    7: "PS",
    8: "BS",
}


# Build confusion matrix
cf_matrix = confusion_matrix(y_test, predicted_labels)

plot_confusion_matrix(cf_matrix, classes, "confusion_matrix")

adjacent_facies = [
    [1],
    [0, 2],
    [1],
    [4],
    [3, 5],
    [4, 6, 7],
    [5, 7],
    [5, 6, 8],
    [6, 7],
]

print("Facies classification accuracy = %f" % accuracy(cf_matrix))
print(
    "Adjacent facies classification accuracy = %f"
    % accuracy_adjacent(cf_matrix, adjacent_facies)
)

# model selection takes a few minutes, change this variable
# to true to run the parameter loop
do_model_selection = True

if do_model_selection:
    C_range = np.array([0.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])
    gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])

    fig, axes = plt.subplots(
        3, 2, sharex="col", sharey="row", figsize=(10, 10)
    )
    plot_number = 0
    for outer_ind, gamma_value in enumerate(gamma_range):
        row = int(plot_number / 2)
        column = int(plot_number % 2)
        cv_errors = np.zeros(C_range.shape)
        train_errors = np.zeros(C_range.shape)
        for index, c_value in enumerate(C_range):
            clf = svm.SVC(C=c_value, gamma=gamma_value)
            clf.fit(X_train, y_train)

            train_conf = confusion_matrix(y_train, clf.predict(X_train))
            cv_conf = confusion_matrix(y_test, clf.predict(X_test))

            cv_errors[index] = accuracy(cv_conf)
            train_errors[index] = accuracy(train_conf)

        ax = axes[row, column]
        ax.set_title("Gamma = %g" % gamma_value)
        ax.semilogx(C_range, cv_errors, label="CV error")
        ax.semilogx(C_range, train_errors, label="Train error")
        plot_number += 1
        ax.set_ylim([0.2, 1])

    ax.legend(bbox_to_anchor=(1.05, 0), loc="lower left", borderaxespad=0.0)
    fig.text(0.5, 0.03, "C value", ha="center", fontsize=14)

    fig.text(
        0.04,
        0.5,
        "Classification Accuracy",
        va="center",
        rotation="vertical",
        fontsize=14,
    )
    plt.savefig("imgs/classification_accuracy.png")

clf = svm.SVC(C=10, gamma=1)
clf.fit(X_train, y_train)

cv_conf = confusion_matrix(y_test, clf.predict(X_test))

print("Optimized facies classification accuracy = %.2f" % accuracy(cv_conf))
print(
    "Optimized adjacent facies classification accuracy = %.2f"
    % accuracy_adjacent(cv_conf, adjacent_facies)
)

plot_confusion_matrix(cv_conf, classes, "confusion_matrix_optimized")

y_blind = blind["Facies"].values
well_features = blind.drop(
    ["Facies", "Formation", "Well Name", "Depth"], axis=1
)
X_blind = scaler.transform(well_features)
y_pred = clf.predict(X_blind)
blind["Prediction"] = y_pred
cv_conf = confusion_matrix(y_blind, y_pred)

print("Optimized facies classification accuracy = %.2f" % accuracy(cv_conf))
print(
    "Optimized adjacent facies classification accuracy = %.2f"
    % accuracy_adjacent(cv_conf, adjacent_facies)
)

compare_facies_plot(blind, "Prediction", facies_colors, "compare_facies_plot")

well_data = pd.read_csv("data/validation_data_nofacies.csv")
well_data["Well Name"] = well_data["Well Name"].astype("category")
well_features = well_data.drop(["Formation", "Well Name", "Depth"], axis=1)

X_unknown = scaler.transform(well_features)

# predict facies of unclassified data
y_unknown = clf.predict(X_unknown)
well_data["Facies"] = y_unknown
# print(well_data)

well_data["Well Name"].unique()


make_facies_log_plot(
    well_data[well_data["Well Name"] == "STUART"], facies_colors, "STUART"
)

make_facies_log_plot(
    well_data[well_data["Well Name"] == "CRAWFORD"], facies_colors, "CRAWFORD"
)

well_data.to_csv("imgs/table/well_data_with_facies.csv")
