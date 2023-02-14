"""Your task is to use Non-dominated Sorting Genetic Algorithm-II (NSGA-II) to minimise following two objectives in
feature selection:
    1) Minimise the classification error rate,
    2) Minimise the ratio of selected features, i.e., number of selected features divided by the total number of
        features.

In this question, two datasets are given as follows:
• Vehicle dataset (vehicle.dat and vehicle.doc), with 846 instances, 18 features and 4
classes.
• Musk “Clean1” dataset (clean1.data, clean1.info and clean1.names), with 476 instances, 168 features and 2 classes.

NOTE: the datasets are not split. You are allowed to use the entire dataset to do the feature
selection without considering the test and feature selection bias. In other words, this question
is an optimisation problem, and cares ONLY about the training performance.

You can use a library for NSGA-II. You should:
    • Determine the proper individual representation and explain the reasons.
    • Determine a wrapper-based fitness function (e.g., KNN).
    • Design proper crossover and mutation operators.
    • Set the necessary algorithm parameter values, such as population size, termination criteria, crossover and
        mutation rates, selection scheme.
    • For each dataset, run the NSGA-II for 3 times with different random seeds. Each run will obtain a set of
        non-dominated solutions (each solution is a feature subset).
    • Compute the hyper-volume of each of the 3 solution sets obtained by NSGA-II.
    • Draw a figure for each of the 3 solution set in the objective space (x-axis is classification
    error rate, y-axis is the ratio of selected features).
    • Compare the error rates of the obtained solutions with that of using the entire feature
        set. Make discussions on the error rates and number of selected features of the obtained
        solutions, and draw you conclusions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymoo.core.problem
from pandas import DataFrame
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
# pymoo expects a class that inherits from pymoo.core.problem.Problem
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from sklearn.neighbors import KNeighborsClassifier


class FeatureSelectionProblem(pymoo.core.problem.Problem):
    """A class to represent the feature selection problem"""

    def __init__(self, data: np.array, target, n_features, n_classes):
        super().__init__(n_var=n_features, n_obj=2, n_constr=0, xl=0, xu=1)
        self.data = data
        self.n_features = n_features
        self.n_classes = n_classes
        self.target = target
        self.knn = KNeighborsClassifier(n_neighbors=n_classes)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the fitness of the given feature subset to
            1) Minimise the classification error rate,
            2) Minimise the ratio of selected features, i.e., number of selected features divided by the total number of
                features.
        """
        # x is a matrix of shape (n_samples, n_features)
        # out is a dictionary with the following keys:
        #   F: the fitness of each solution, a matrix of shape (n_samples, n_objectives)
        #   CV: the constraint violation of each solution, a matrix of shape (n_samples, n_constraints)
        #   G: the constraint evaluation of each solution, a matrix of shape (n_samples, n_constraints)

        # Get the feature subset
        classification_error_rate = []
        ratio_of_selected_features = []
        for subset in range(x.shape[0]):
            feature_subset = x[0, :] == 1
            # Get the data subset
            data_subset = self.data[:, feature_subset]

            # Get the labels
            labels = self.target

            # Get the classifier
            classifier = self.knn
            # Train the classifier
            classifier.fit(data_subset, labels)
            # Get the predictions
            predictions = classifier.predict(data_subset)

            # # Get the number of correct predictions
            correct_predictions = (predictions == labels).sum()
            # Get the classification error rate
            classification_error_rate.append(1 - correct_predictions / len(labels))
            # Get the ratio of selected features
            ratio_of_selected_features.append(feature_subset.sum() / self.n_features)
            # Set the fitness
        out["F"] = np.column_stack([classification_error_rate, ratio_of_selected_features])


def load_data(path, datasource):
    """Load the data from the datasource"""
    import csv
    datapath = path / datasource
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(open(datapath).readline())
    data = pd.read_csv(datapath, dialect=dialect, header=None)

    print(f"Loaded {len(data)} rows of length {len(data.columns)} from {datasource}")
    return clean_data(data)


def clean_data(data: DataFrame):
    """Clean the data"""
    data = data.dropna(axis=1, how="all")

    # separate the data from the target values
    target = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    # label encode all non-numeric columns
    original_vals = [[]] * len(data.columns)
    for column in data.columns:
        if data[column].dtype == np.object:
            original_vals[data.columns.get_loc(column)] = data[column].unique()
            data[column] = pd.factorize(data[column])[0]

    # replace all NaN values with 0
    data = data.fillna(0)

    # normalise all columns to be between 0 and 1
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    return original_vals, data, target


from pymoo.indicators.hv import HV

if __name__ == '__main__':
    import os

    problem = get_problem("zdt5")
    # get the path of the current datasets directory
    path = Path(os.getcwd()).parent
    path = path / 'Datasets' / 'NSGA-II'
    # load the data
    _vehicle, vehicle_X, vehicle_y = load_data(path, 'vehicle/vehicle.dat')
    _musk, musk_X, musk_y = load_data(path, 'musk/clean1.data')
    # remove the class column from the data

    # remove the first column from the vehicle data
    # vehicle_X.pop(vehicle_X.columns[0])
    # remove the first 2 columns from the musk data
    # musk_X.pop(musk_X.columns[0])
    # musk_X.pop(musk_X.columns[0])

    # create the feature selection problem
    vehicle_problem = FeatureSelectionProblem(vehicle_X.to_numpy(), vehicle_y, len(vehicle_X.columns),
                                              len(vehicle_y.unique()))
    musk_problem = FeatureSelectionProblem(musk_X.to_numpy(), musk_y, len(musk_X.columns), len(musk_y.unique()))

    knn_vehicle = KNeighborsClassifier(n_neighbors=len(vehicle_y.unique()))
    knn_musk = KNeighborsClassifier(n_neighbors=len(musk_y.unique()))

    knn_vehicle.fit(vehicle_X, vehicle_y)
    knn_musk.fit(musk_X, musk_y)

    vehicle_predictions = knn_vehicle.predict(vehicle_X)
    musk_predictions = knn_musk.predict(musk_X)

    vehicle_correct_predictions = (vehicle_predictions == vehicle_y).sum()
    musk_correct_predictions = (musk_predictions == musk_y).sum()

    vehicle_error_rate = 1 - vehicle_correct_predictions / len(vehicle_y)
    musk_error_rate = 1 - musk_correct_predictions / len(musk_y)

    ref_point = np.array([1, 1])

    ind = HV(ref_point=ref_point)

    vehicle_hv = ind(np.array([vehicle_error_rate,  1]))
    musk_hv = ind(np.array([musk_error_rate, 1]))

    # create the algorithm
    from pymoo.termination.default import DefaultMultiObjectiveTermination

    # define termination criteria
    termination = DefaultMultiObjectiveTermination(
        xtol=0.01,
        cvtol=1e-6,
        ftol=0.005,
        period=20,
        n_max_gen=100,
        n_max_evals=10000
    )
    algorithm = NSGA2(pop_size=15,
                      sampling=BinaryRandomSampling(),
                      selection=TournamentSelection(func_comp=binary_tournament),
                      crossover=TwoPointCrossover(prob=0.9),
                      mutation=BitflipMutation(prob=0.1),
                      eliminate_duplicates=True)

    # run the algorithm


    vehicle_res, musk_res = [], []
    for i in range(3):
        vehicle_res.append(minimize(vehicle_problem, algorithm, seed=i, verbose=True, termination=termination))
        musk_res.append(minimize(musk_problem, algorithm, seed=i, verbose=True, termination=termination))
    # print the results
    print("Vehicle")
    print("===========")
    plot = Scatter(title="Vehicle", labels=["Error Rate", "Selected Features"])
    plot.add(problem.pareto_front(use_cache=False, ), plot_type="line", color="black")
    plot.add(np.array([vehicle_error_rate, 1]), color="red", marker="x", s=100)
    print(f"Vehicle error rate with all features: {vehicle_error_rate * 100}% \n")
    print(f"Vehicle HV: {vehicle_hv} \n")
    for res in vehicle_res:
        print(f"Vehicle Error rate with {float(res.F[0][1]) * 100}% of features: {res.F[0][0] * 100}%")

        plot.add(problem.pareto_front(use_cache=False, ), plot_type="line", color="black")
        plot.add(res.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
        print("Calculated Hyper-Volume:", ind(res.F), "\n")
        plot.show()
    # compute hyper-volume (the higher the better)

    # print(vehicle_X[vehicle_res.X].head())
    print("Musk")
    print("===========")
    plot = Scatter(title="Musk", labels=["Error Rate", "Selected Features"])
    plot.add(problem.pareto_front(use_cache=False, ), plot_type="line", color="black")
    plot.add(np.array([musk_error_rate, 1]), color="red", marker="x", s=100)
    print(f"Musk error rate with all features: {(musk_error_rate * 100)}\n")
    print(f"Musk HV: {musk_hv} \n")
    for res in musk_res:
        print(f"Musk Error rate with {(float(res.F[0][1]) * 100)}% of features: {res.F[0][0] * 100}%")

        plot.add(res.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)

        print("Calculated Hyper-Volume:", ind(res.F), "\n")
        plot.show()
    # res = minimize(vehicle,
    #                algorithm,
    #                ('n_gen', 500),
    #         AttributeError: 'Result' object has no attribute 'ndim'       seed=1,
    #                verbose=False)

    print(path)
