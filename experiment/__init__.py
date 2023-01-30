import itertools
import sys
from typing import Callable
import pandas as pd
import numpy as np
from psyki.logic import Formula
from psyki.qos.energy import Energy
from psyki.qos.latency import Latency
from psyki.qos.memory import Memory
from psyki.ski import Injector
from psyki.ski.kill import LambdaLayer
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.random_seed import set_seed
from datasets import SpliceJunction, BreastCancer, CensusIncome
from psyki.logic.prolog import TuProlog
from knowledge import PATH as KNOWLEDGE_PATH
from datasets import PATH as DATA_PATH
from results import PATH as RESULT_PATH

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

BATCH_SIZE = 32
EPOCHS = 100
VERBOSE = 0
SEED = 0
NEURONS_PER_LAYERS = [10, 50]
LAYERS = [1, 2]

sys.setrecursionlimit(2000)


def filter_neurons(maximum_value: int):
    return list(v for v in NEURONS_PER_LAYERS if v <= maximum_value)


def create_nn(input_layer: int, output_layer: int, hidden_layers: int, neurons: list[int]) -> Model:
    input_layer = Input((input_layer,))
    x = input_layer
    for h in list(range(hidden_layers)):
        x = Dense(neurons[h], activation="relu")(x)
    x = Dense(output_layer, activation="softmax")(x)
    model = Model(input_layer, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    return model


def create_educated_nn(input_layer: int, output_layer: int, hidden_layers: int, neurons: list[int], injector,
                       formulae: list[Formula], injector_params: dict) -> Model:
    model = create_nn(input_layer, output_layer, hidden_layers, neurons)
    injector = injector(model, **injector_params)
    educated_model = injector.inject(formulae)
    educated_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    return educated_model


def get_dataset_and_knowledge(dataset_name: str):
    def split_dataset(train, test):
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        dataset_split = dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=len(np.unique(train_y)))
        return dataset_split

    def splice_data_and_knowledge():
        train = pd.read_csv(DATA_PATH / "splice-junction-data.csv")
        test = pd.read_csv(DATA_PATH / "splice-junction-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "splice-junction.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, SpliceJunction.class_mapping_short, knowledge

    def census_data_and_knowledge():
        train = pd.read_csv(DATA_PATH / "census-income-data.csv")
        test = pd.read_csv(DATA_PATH / "census-income-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "census-income.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, CensusIncome.class_mapping, knowledge

    def breast_data_and_knowledge():
        train = pd.read_csv(DATA_PATH / "breast-cancer-data.csv")
        test = pd.read_csv(DATA_PATH / "breast-cancer-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "breast-cancer.pl")).formulae
        dataset_split = split_dataset(train, test)
        return dataset_split, feature_mapping, BreastCancer.class_mapping_short, knowledge

    if dataset_name == SpliceJunction.name:
        return splice_data_and_knowledge()
    elif dataset_name == CensusIncome.name:
        return census_data_and_knowledge()
    else:  # dataset_name == 'breast':
        return breast_data_and_knowledge()


def grid_search(dataset_name: str, grid_search_params: dict, creator: Callable):
    set_seed(SEED)
    dataset, mapping, class_map, knowledge = get_dataset_and_knowledge(dataset_name)
    grid_search_params['input_layer'] = [len(dataset['train_x'].columns)]
    grid_search_params['output_layer'] = [len(np.unique(dataset['train_y']))]
    if 'injector' in grid_search_params.keys():
        if grid_search_params['injector'][0] == Injector.kill:
            if dataset_name == SpliceJunction.name:
                class_mapping = SpliceJunction.class_mapping_short
            elif dataset_name == CensusIncome.name:
                class_mapping = CensusIncome.class_mapping
            else:
                class_mapping = BreastCancer.class_mapping_short
            grid_search_params['injector_params'] = [{
                'feature_mapping': {k: v for v, k in enumerate(dataset['train_x'].columns)},
                'class_mapping': class_mapping
            }]
        else:
            grid_search_params['injector_params'] = [{
                'feature_mapping': {k: v for v, k in enumerate(dataset['train_x'].columns)}
            }]
    predictor = KerasClassifier(creator, **grid_search_params, random_state=SEED)
    gs = GridSearchCV(predictor, grid_search_params, cv=2, n_jobs=1, )
    gs.fit(dataset['train_x'], dataset['train_y'], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    return gs.best_params_


def compute_metrics(predictor1: Model, predictor2: Model, training_params: dict, result_file_name: str):
    # Energy
    energy = Energy.compute_during_training(predictor1, predictor2, training_params)
    # Memory
    memory = Memory.compute_during_training(predictor1, predictor2, training_params)
    # Latency
    latency = Latency.compute_during_training(predictor1, predictor2, training_params)
    file_name = result_file_name[:-4] + '-train' + result_file_name[-4:]
    pd.DataFrame({'energy': energy, 'memory': memory, 'latency': latency}).to_csv(RESULT_PATH / file_name)

    # Energy
    energy = Energy.compute_during_inference(predictor1, predictor2, training_params)
    # Memory
    memory = Memory.compute_during_inference(predictor1, predictor2, training_params)
    # Latency
    latency = Latency.compute_during_inference(predictor1, predictor2, training_params)
    file_name = result_file_name[:-4] + '-inference' + result_file_name[-4:]
    pd.DataFrame({'energy': energy, 'memory': memory, 'latency': latency}).to_csv(RESULT_PATH / file_name)


if __name__ == '__main__':
    # First grid search
    dataset = 'splice'
    grid_search_params = {
        'neurons': list(list(x) for x in itertools.product(NEURONS_PER_LAYERS, NEURONS_PER_LAYERS)),
        'hidden_layers': [1, 2],
        'input_layer': [240],
        'output_layer': [3]
    }
    best_params = grid_search(dataset, grid_search_params, create_nn)
    # Second grid search
    formulae = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    grid_search_params = {
        'injector': [Injector.kins],
        'formulae': [formulae],
        'neurons': list(list(x) for x in itertools.product(filter_neurons(best_params['neurons'][0]),
                                                           filter_neurons(best_params['neurons'][1]))),
        'hidden_layers': list(range(best_params['hidden_layers'])),
        'input_layer': [240],
        'output_layer': [3]
    }
    print("\n\n\n\n")
    best_params = grid_search(dataset, grid_search_params, create_educated_nn)
