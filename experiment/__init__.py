import itertools
from typing import Callable
import pandas as pd
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from psyki.logic import Formula
from psyki.ski import Injector
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from datasets import SpliceJunction, BreastCancer, CensusIncome
from psyki.logic.prolog import TuProlog
from knowledge import PATH as KNOWLEDGE_PATH
from datasets import PATH as DATA_PATH


"""
1 - grid search to find a "good" classic NN (B) for the dataset, with a threshold (t, t could be computed as the accuracy of a DT)
2 - injectors
  2.1 - kbann, only gamma should be fine-tuned (optionally consider also the constraint variant)
  2.2 - kins, grid search with B as upperbound
  2.3 - kill, we could keep B (or apply grid search), keep attention to batch size and the accuracy of the KB
3 - for each run we SAVE the results in a csv
4 - post analysis of the results

Grid search
Epochs = 100
Batch size = [[64], 32]
Early stop = t is reached and no accuracy growth in [[5], 10] epochs
Input layer    -> the number of neurons is equal to the number of features (9, 240, 85)
Hidden layer 1 -> grid search [10, 50, 100, 500, 1000]
Hidden layer 2 -> grid search [10, 50, 100, 500, 1000]
Hidden layer 3 -> grid search [10, 50, 100, 500, 1000]
Output layer   -> number of classes (2, 3, 2)

Dataset
Breast Cancer: 699 -> 466 | 233
Splice Junction: 3190 -> 2127 | 163
Census Income: 32561 | 16281
"""


# TODO: to be split into 2 different searches
def grid_search(dataset_name: str, injector: Injector, training_params: dict, knowledge: list[Formula]):

    def create_nn(input_layer: int, output_layer: int, hidden_layers: int, neurons: list[int]) -> Model:
        input_layer = Input((input_layer,))
        x = input_layer
        for h in range(hidden_layers):
            x = Dense(neurons[h], activation="relu")(x)
        x = Dense(output_layer, activation="softmax")(x)
        model = Model(input_layer, x)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
        return model

    def create_educated_nn(input_layer: int, output_layer: int, hidden_layers: int, neurons: list[int], injector: Injector, formulae: list[Formula]) -> Model:
        model = create_nn(input_layer, output_layer, hidden_layers, neurons)
        injector._predictor = model.copy()
        educated_model = injector.inject(formulae)
        educated_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
        return educated_model

    def split_dataset(train, test):
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        dataset_split = dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=len(np.unique(train_y)))
        return dataset_split

    def splice_data():
        train = pd.read_csv(DATA_PATH / "splice-junction-data.csv")
        test = pd.read_csv(DATA_PATH / "splice-junction-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "splice-junction.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, SpliceJunction.class_mapping_short, knowledge

    def census_data():
        train = pd.read_csv(DATA_PATH / "census-income-data.csv")
        test = pd.read_csv(DATA_PATH / "census-income-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "census-income.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, CensusIncome.class_mapping, knowledge

    def breast_data():
        train = pd.read_csv(DATA_PATH / "breast-cancer-data.csv")
        test = pd.read_csv(DATA_PATH / "breast-cancer-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "breast-cancer.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, BreastCancer.class_mapping_short, knowledge

    if dataset_name == 'splice':
        y = splice_data()
    elif dataset_name == 'census':
        y = census_data()
    else:  # dataset_name == 'breast':
        y = breast_data()
    dataset, mapping, class_map, knowledge = y

    # 1° GRID SEARCH
    neurons_per_layer = [10, 50, 100]
    filter_neurons: Callable = lambda m: list(v for v in neurons_per_layer if v <= m)
    predictor = KerasClassifier(create_nn)
    gs_params = {
        'neurons': list(list(x) for x in itertools.product(neurons_per_layer, neurons_per_layer)),
        'hidden_layers': [1, 2],
        'input_layer': [240],
        'output_layer': [3]
    }
    gs = GridSearchCV(predictor, gs_params, cv=2)
    gs.fit(dataset['train_x'], dataset['train_y'], epochs=100, batch_size=64)
    best_params = gs.best_params_
    # TODO: save best parameters into a file

    # 2° GRID SEARCH (depending on the injector)
    predictor = KerasClassifier(create_educated_nn)
    gs_params = {
        'neurons': list(list(x) for x in itertools.product(filter_neurons(best_params['neurons'][0]), filter_neurons(best_params['neurons'][1]))),
        'hidden_layers': [range(best_params['hidden_layers'])],
        'input_layer': [240],
        'output_layer': [3]
    }
    gs = GridSearchCV(predictor, gs_params, cv=2)
    gs.fit(dataset['train_x'], dataset['train_y'], epochs=100, batch_size=64)
    best_params = gs.best_params_
    # TODO: save best parameters into a file


if __name__ == '__main__':
    dataset = 'splice'
    training_params = None
    formulae = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    injector = None
    grid_search(dataset, injector, training_params, formulae)
