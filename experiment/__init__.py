import itertools
import sys
from typing import Callable
import pandas as pd
import numpy as np
from psyki.logic import Formula
from psyki.qos.data import DataEfficiency
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
from results import PATH as RESULTS_PATH
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
Hidden layer 1 -> grid search [10, 50, 100]
Hidden layer 2 -> grid search [10, 50, 100]
Hidden layer 3 -> grid search [10, 50, 100]
Output layer   -> number of classes (2, 3, 2)

Dataset
Breast Cancer: 699 -> 466 | 233
Splice Junction: 3190 -> 2127 | 1063
Census Income: 32561 | 16281
"""

BATCH_SIZE = 64
EPOCHS = 100
VERBOSE = 0
SEED = 0
POPULATION_SIZE = 30
LOSS = "sparse_categorical_crossentropy"
NEURONS_PER_LAYERS = [10, 50, 100]
LAYERS = [1, 2, 3]
ACCEPTABLE_ACCURACY_DROP = 0.95
ACCEPTABLE_ACCURACY = 0.8
CALLBACK = []

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
    model.compile(optimizer="adam", loss=LOSS, metrics="accuracy")
    return model


def create_educated_nn(input_layer: int, output_layer: int, hidden_layers: int, neurons: list[int], injector,
                       formulae: list[Formula], injector_params: dict) -> Model:
    model = create_nn(input_layer, output_layer, hidden_layers, neurons)
    injector = injector(model, **injector_params)
    defensive_copy = [f.copy() for f in formulae]
    for formula in defensive_copy:
        formula.optimize()
        formula.trainable = True
    educated_model = injector.inject(defensive_copy)
    educated_model.compile(optimizer="adam", loss=LOSS, metrics="accuracy")
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
    if 'accuracy' in grid_search_params.keys():
        grid_search_params.pop('accuracy')
    predictor = KerasClassifier(creator, **grid_search_params, random_state=SEED)
    gs = GridSearchCV(predictor, grid_search_params, cv=2, n_jobs=1, )
    gs.fit(dataset['train_x'], dataset['train_y'], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=CALLBACK)
    gs.best_params_['accuracy'] = gs.best_score_
    return gs.best_params_


def compute_metrics_training(predictor1: Model, predictor2: Model, params: dict):
    metric = params['metric']
    params.pop('metric')
    if metric == 'energy':
        return Energy.compute_during_training(predictor1, predictor2, params)
    elif metric == 'memory':
        return Memory.compute_during_training(predictor1, predictor2, params)
    elif metric == 'latency':
        return Latency.compute_during_training(predictor1, predictor2, params)
    elif metric == 'data efficiency':
        return DataEfficiency.compute_during_training(predictor1, predictor2, params)


def compute_metrics_inference(predictor1: Model, predictor2: Model, params: dict):
    metric = params['metric']
    params.pop('metric')
    if metric == 'energy':
        return Energy.compute_during_inference(predictor1, predictor2, params)
    elif metric == 'memory':
        return Memory.compute_during_inference(predictor1, predictor2, params)
    elif metric == 'latency':
        return Latency.compute_during_inference(predictor1, predictor2, params)
    elif metric == 'data efficiency':
        return DataEfficiency.compute_during_inference(predictor1, predictor2, params)
    else:
        raise Exception('Unknown metric')


def run_experiments(datasets: list[SpliceJunction or BreastCancer or CensusIncome], injectors: list[Callable],
                    injector_names: list[str], columns: list[str], metrics_names: list[str], update: bool = False):
    # Iterate over datasets
    for dataset in datasets:
        print('\n\n' + dataset.name)
        results_training = pd.DataFrame(columns=columns)
        results_inference = pd.DataFrame(columns=columns)
        train_data = pd.read_csv(dataset.file_name)
        test_data = pd.read_csv(dataset.file_name_test)
        best_params = pd.read_csv(RESULTS_PATH / (dataset.name + '.csv'), index_col=0)
        uneducated_neurons = eval(best_params.loc['uneducated']['neurons'])
        layers = len(uneducated_neurons)
        # Iterate over injectors
        for injector, injector_name in zip(injectors, injector_names):
            print(injector_name)
            if injector_name != 'kbann':
                educated_neurons = eval(best_params.loc[injector_name]['neurons'])
            else:
                educated_neurons = uneducated_neurons
            formulae = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
            # KILL injector needs the class mapping
            injection_params = {'feature_mapping': {k: v for v, k in enumerate(train_data.columns)}}
            if injector_name == 'kill':
                injection_params['class_mapping'] = dataset.class_mapping_short
            # Create the educated predictor
            set_seed(SEED)
            educated = create_educated_nn(len(train_data.columns) - 1, len(dataset.class_mapping_short), layers,
                                          educated_neurons, injector, formulae, injection_params)
            for metric in metrics_names:
                params = {
                    'x': train_data.iloc[:, :-1],
                    'y': train_data.iloc[:, -1:],
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'verbose': VERBOSE,
                    'metric': metric
                }

                # Create the uneducated predictor
                set_seed(SEED)
                uneducated = create_nn(len(train_data.columns) - 1, len(dataset.class_mapping), layers,
                                       uneducated_neurons)
                if metric != 'data efficiency':
                    results_training.loc[injector_name, metric] = compute_metrics_training(uneducated, educated, params)
                else:
                    # Compute accuracy over POPULATION_SIZE (30) runs to get a statistically significant result
                    accuracies_educated = []
                    accuracies_uneducated = []
                    for idx in range(POPULATION_SIZE):
                        print("Training and computing accuracy for run " + str(idx + 1) + " of " + str(
                            POPULATION_SIZE) + "...")
                        set_seed(SEED + idx)
                        uneducated = create_nn(len(train_data.columns) - 1, len(dataset.class_mapping), layers,
                                               uneducated_neurons)
                        uneducated.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1:], epochs=EPOCHS,
                                       batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=CALLBACK)
                        accuracy_percent = 100 * uneducated.evaluate(test_data.iloc[:, :-1], test_data.iloc[:, -1:])[1]
                        accuracies_uneducated.append(accuracy_percent)
                        set_seed(SEED + idx)
                        for formula in formulae:
                            formula.trainable = True
                        educated = create_educated_nn(len(train_data.columns) - 1, len(dataset.class_mapping_short),
                                                      layers, educated_neurons, injector, formulae, injection_params)
                        educated.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1:], epochs=EPOCHS,
                                     batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=CALLBACK)
                        if injector_name == 'kill' and isinstance(educated, LambdaLayer.ConstrainedModel):
                            educated = educated.remove_constraints()
                            educated.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])
                        accuracy_percent = 100 * educated.evaluate(test_data.iloc[:, :-1], test_data.iloc[:, -1:])[1]
                        accuracies_educated.append(accuracy_percent)

                    accuracy_history = pd.DataFrame(
                        {'uneducated': accuracies_uneducated, 'educated': accuracies_educated})
                    accuracy_history.to_csv(RESULTS_PATH / (dataset.name + '_' + injector_name + '_accuracy.csv'))
                    params['train_x1'] = train_data.iloc[:, :-1]
                    params['train_y1'] = train_data.iloc[:, -1:]
                    params['train_x2'] = train_data.iloc[:, :-1]
                    params['train_y2'] = train_data.iloc[:, -1:]
                    params['epochs1'] = EPOCHS
                    params['epochs2'] = EPOCHS
                    params['metric1'] = np.mean(accuracies_uneducated)
                    params['metric2'] = np.mean(accuracies_educated)
                    params.pop('x')
                    params.pop('y')
                    params.pop('epochs')

                params['metric'] = metric
                # KILL injector needs the constraints to be removed
                if injector_name == 'kill' and isinstance(educated, LambdaLayer.ConstrainedModel):
                    educated = educated.remove_constraints()
                    educated.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                if metric == 'data efficiency':
                    params['metric'] = metric
                    data_efficiency = compute_metrics_inference(uneducated, educated, params)
                    results_training.loc[injector_name, metric] = data_efficiency
                    results_inference.loc[injector_name, metric] = results_training.loc[injector_name, metric]
                    results_training.loc[injector_name, 'accuracy'] = params['metric2']
                    results_inference.loc[injector_name, 'accuracy'] = params['metric2']
                    results_training.loc[injector_name, 'accuracy uneducated'] = params['metric1']
                    results_inference.loc[injector_name, 'accuracy uneducated'] = params['metric1']
                else:
                    params.pop('y')
                    params.pop('epochs')
                    params.pop('batch_size')
                    results_inference.loc[injector_name, metric] = compute_metrics_inference(uneducated, educated, params)

        if update:
            results_training.to_csv(RESULTS_PATH / (dataset.name + '_metrics_training.csv'), mode='+', float_format='%.2f')
            results_inference.to_csv(RESULTS_PATH / (dataset.name + '_metrics_inference.csv'), mode='+', float_format='%.2f')
        else:
            results_training.to_csv(RESULTS_PATH / (dataset.name + '_metrics_training.csv'), float_format='%.2f')
            results_inference.to_csv(RESULTS_PATH / (dataset.name + '_metrics_inference.csv'), float_format='%.3f')
