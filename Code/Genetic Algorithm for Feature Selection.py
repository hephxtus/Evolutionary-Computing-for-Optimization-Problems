import random
import time

import numpy as np
import pandas as pd
from genetic_selection import GeneticSelectionCV
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

N_FEATURES = 5


# Genetic Algorithm for Feature Selection [25 marks]
# In this question, two datasets are given as follows:
# • Winsconsin Breast Cancer (wbcd.csv and wbcd.names), with 30 features and 2 classes.
# • Sonar (sonar.csv and sonar.names), with 60 features and 2 classes.

# Develop a GA to do feature selection on these two datasets. You can use a library, or
# reuse/modify the code from the GA for 0-1 knapsack.


# You should
# 1. Determine the proper individual representation and explain the reasons.

# 2. Determine one filter-based fitness function (e.g., information gain), and one wrapper-based fitness function
# (e.g., KNN).

# 3. Design proper crossover and mutation operators.

# 4. Set the necessary GA parameter values,
# such as population size, termination criteria, crossover and mutation rates, selection scheme.
# 5. For each instance,
#   - for 5 times with different random seeds. Each run will obtain one selected subset.
#   -   run the GA with the filter-based fitness function (called FilterGA)
#   -   run the GA with the wrapper-based fitness function (called WrapperGA)

# 6. Compare the mean and standard deviation of the computational time of the FilterGA and WrapperGA and draw
# your conclusions.

# 7. For each selected feature subset (5 subsets selected by FilterGA and 5 subsets selected by
# WrapperGA), transform the dataset by removing the unselected features.
# Then, choose a classifier (e.g.,  Naive Bayes) to do the classification on the transformed dataset.

# 8. Compare the mean and standard deviation of the classification accuracy of the 5 subsets selected by FilterGA and
# the 5 subsets selected by WrapperGA, and draw your conclusions.

# https://www.datacamp.com/tutorial/feature-selection-python

# Then Transform the dataset by removing the unselected features.
# Then, choose a classifier (e.g.,  Naive Bayes) to do the classification on the transformed dataset.
# Compare the mean and standard deviation of the classification accuracy of the 5 subsets selected by FilterGA and
# the 5 subsets selected by WrapperGA, and draw your conclusions.
import os
import pandas as pd

Dataset_dir = "../Datasets/feature-selection"
Output_dir = "../Output/knapsack-data"


def get_files():
    sonar_data = os.path.join(Dataset_dir, 'sonar',"sonar.data")
    wcbd_data = os.path.join(Dataset_dir, 'wbcd',"wbcd.data")
    sonar_names = os.path.join(Dataset_dir, 'sonar',"sonar.names")
    wcbd_names = os.path.join(Dataset_dir, 'wbcd',"wbcd.names")
    return sonar_data, sonar_names, wcbd_data, wcbd_names


def load_data(data_path, names_path):
    # Load the data
    # Load .data file
    print("Loading data from: ", data_path)
    # Load .names file
    print("Loading names from: ", names_path)

    # read .names into list where line 0 is the target range and the rest are feature names for data
    with open(names_path) as f:
        names = [line for line in f.readlines()]
        #remove new line character
        names = [name.lstrip() for name in names]
    # Set the target column name
    target_range = names[0].rstrip('.\n').split(',')
    names = [name.split(':')[0] for name in names[1:]]
    names.append('target')
    target_range = [int(i) for i in target_range]    # Set the feature column names
    data = pd.read_csv(data_path, names=names, sep=',', index_col=False)
    return data

def naive_bayes_classifier(data, labels):
    """
    Choose a classifier (e.g.,  Naive Bayes) to do the classification on the transformed dataset.
    :param data:
    :param labels:
    :return:
    """
        # Create a Gaussian Classifier
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(data, labels)
    return model.score(X_test, y_test)


def remove_unselected_features(data: pd.DataFrame, selected_features: pd.Index):
    return data[selected_features]

def classify(methods:list, datasets: list):
    """
    For each selected feature subset (5 subsets selected by FilterGA and 5 subsets selected by
    WrapperGA), transform the dataset by removing the unselected features. Then, choose
    a classifier (e.g., Naive Bayes) to do the classification on the transformed dataset.
    :param methods:
    :param datasets:
    :return: return the classification accuracy of the 5 subsets selected by FilterGA and
    the 5 subsets selected by WrapperGA, and draw your conclusions.
    """
    classification_accuracies = {method[0]: {} for method in methods}
    for dataset in datasets:
        dataset_name = dataset[0]
        data = dataset[1]
        labels = data['target']

        for method in methods:

            method_name = method[0]
            algorithm = method[1][dataset_name]
            classification_accuracies[method_name][dataset_name] = []
            for i in range(5):
                print(algorithm)
                selected_features = algorithm[i]['features']
                data_copy = remove_unselected_features(data, selected_features)
                accuracy = naive_bayes_classifier(data_copy, labels)
                classification_accuracies[method_name][dataset_name].append(accuracy)
    return classification_accuracies

def mean_and_std(values):
    return np.mean(values), np.std(values)

def run_ga(datasets: list, methods: list):
    """
    For each instance, run the GA with the filter-based fitness function (called FilterGA)
    and the GA with the wrapper-based fitness function (called WrapperGA) for 5 times
    with different random seeds. Each run will obtain one selected subset.
    :param datasets: list of datasets in format: [(name1, data1), (name2, data2), ...])]
    :param methods: list of methods in format: [(name1, method1), (name2, method2), ...])]
    :return: dictionary of computational_time, selected_features of each filter and wrapper for each seed and dataset
                in format {method: {dataset: [{computational_time: , selected_features: }, ...]}}
    """
    benchmarks = {method[0]: {} for method in methods}
    for dataset in datasets:
        dataset_name = dataset[0]
        data = dataset[1]
        for method in methods:
            method_name = method[0]
            method = method[1]
            benchmarks[method_name][dataset_name] = []
            for i in range(5):
                start_time = time.time()
                selected_features = method(data, i)
                computational_time = time.time() - start_time
                benchmarks[method_name][dataset_name].append({'time': computational_time,
                                                              'features': selected_features})
    return benchmarks
def load_files():
    #sonar_data, sonar_names, wcbd_data, wcbd_names
    sonar_path, sonar_names_path, wbcd_path,  wbcd_names_path = get_files()
    # Load the data

    wcbd_data = load_data(wbcd_path, wbcd_names_path)
    sonar_data = load_data(sonar_path, sonar_names_path)
    return sonar_data, wcbd_data


def WrapperGA(data: pd.DataFrame, random_state: int):
    """
    Determine one one wrapper based fitness function (e.g., KNN).
    :param data:
    :param random_state:
    :return:
    """
    random.seed(random_state)
    from sklearn.feature_selection import SequentialFeatureSelector as SFS
    # Split the data into training and testing
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    # Create the classifier
    clf = KNeighborsClassifier(n_neighbors=3)
    # Create the wrapper
    sfs = SFS(clf, n_features_to_select=N_FEATURES, direction='forward', scoring='accuracy', cv=5)
    # Fit the wrapper
    fit = sfs.fit_transform(X, y)

    # Get the results
    sel_features = sfs.get_feature_names_out()
    return sel_features

def crossover(parent1, parent2):
    """
    Crossover function
    :param parent1:
    :param parent2:
    :return:
    """
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def mutation(child, MUTATION_RATE=None):
    """
    Mutation function
    :param child:
    :return:
    """
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = 1 - child[i]
    return child

def FilterGA(data, random_state):
    """
    Determine one filter-based fitness function (e.g., information gain),
    :param data:
    :param random_state:
    :return:
    """
    random.seed(random_state)
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    # Split the data into training and testing
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # Create the filter
    filter = SelectKBest(mutual_info_classif, k=N_FEATURES)
    # Fit the filter
    fit = filter.fit_transform(X, y)
    # Get the results
    sel_features = X.columns[filter.get_support()]
    return sel_features

def compare(ga:list, key: str):
    """
    Compare the mean and standard deviation of the computational time of the FilterGA
    and WrapperGA and draw your conclusions.
    :param values:
    :return:
    """
    for method_name, _ in ga:
        print(method_name)
        for dataset_name, values in _.items():
            print(dataset_name)
            x = []
            for i in range(5):
                try:
                    x.append(values[i][key])
                except:
                    x.append(values[i])

            mean, std = mean_and_std(x)
            print("Mean: ", mean)
            print("Standard Deviation: ", std)
            print("")

if __name__ == "__main__":
    sonar_data, wbcd_data = load_files()
    feautures = run_ga([('sonar', sonar_data), ('wbcd', wbcd_data)], [('filter', FilterGA), ('wrapper', WrapperGA)])
    wrapper_out, filter_out = feautures['wrapper'], feautures['filter']
    print("GA TIMES:")
    compare([('filter,', filter_out), ('wrapper', wrapper_out)], 'time')
    classification = classify([('wrapper', wrapper_out), ('filter', filter_out)], [('sonar', sonar_data), ('wbcd', wbcd_data)])
    wrapper_out, filter_out = classification['wrapper'], classification['filter']
    print("CLASSIFICATION ACCURACY:")
    compare([('filter,', filter_out), ('wrapper', wrapper_out)], 'accuracy')

