import pandas as pd
import numpy as np
from psyki.qos import QoS
from psyki.ski import Injector
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from datasets import SpliceJunction, BreastCancer, CensusIncome
from psyki.logic.prolog import TuProlog
from knowledge import PATH as KNOWLEDGE_PATH
from datasets import PATH as DATA_PATH


class ExperimentSKIQOS:
    def __init__(self, dataset_name: str, injector: str, hyper: dict, flags: dict):
        self.dataset_name = dataset_name
        self.injector = injector
        self.hyper = hyper
        self.flags = flags

        if self.dataset_name == 'splice':
            self.dataset, self.mapping, self.class_map, self.knowledge = self.splice_data()

        if self.dataset_name == 'census':
            self.dataset, self.mapping, self.class_map, self.knowledge = self.census_data()

        if self.dataset_name == 'breast':
            self.dataset, self.mapping, self.class_map, self.knowledge = self.breast_data()

        self.model = create_standard_fully_connected_nn(input_size=self.dataset['input_size'],
                                                        output_size=self.dataset['output_size'],
                                                        layers=4,
                                                        neurons=250,
                                                        activation='relu')

        if self.injector == 'kins':
            self.injector_arguments = {'feature_mapping': self.mapping,
                                       'injection_layer': len(self.model.layers) - 2}
        if self.injector == 'kill':
            self.injector_arguments = {'feature_mapping': self.mapping,
                                       'class_mapping': self.class_map}
        if self.injector == 'kbann':
            self.injector_arguments = {'feature_mapping': self.mapping}

    def test_qos(self):
        metric_arguments = dict(model=self.model,
                                dataset=self.dataset,
                                injection=self.injector,
                                injector_arguments=self.injector_arguments,
                                formulae=self.knowledge,
                                optim=self.hyper['optimizer'],
                                loss=self.hyper['loss'],
                                batch=self.hyper['batch'],
                                epochs=self.hyper['epochs'],
                                metrics=self.hyper['metric'],
                                max_neurons_width=self.hyper['max_neurons_width'],
                                max_neurons_depth=self.hyper['max_neurons_depth'],
                                max_layers=self.hyper['max_layers'],
                                grid_levels=self.hyper['grid_levels'],
                                threshold=self.hyper['threshold'],
                                alpha=0.8)

        qos = QoS(metric_arguments=metric_arguments, flags=self.flags)
        qos.compute(verbose=True)

    @staticmethod
    def split_dataset(train, test):
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        dataset_split = dict(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=len(np.unique(train_y)))
        return dataset_split

    @staticmethod
    def splice_data():

        train = pd.read_csv(DATA_PATH / "splice-junction-data.csv")
        test = pd.read_csv(DATA_PATH / "splice-junction-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "splice-junction.pl")).formulae
        dataset_split = ExperimentSKIQOS.split_dataset(train, test)
        return dataset_split, feature_mapping, SpliceJunction.class_mapping_short, knowledge

    @staticmethod
    def census_data():

        train = pd.read_csv(DATA_PATH / "census-income-data.csv")
        test = pd.read_csv(DATA_PATH / "census-income-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "census-income.pl")).formulae
        dataset_split = ExperimentSKIQOS.split_dataset(train, test)
        return dataset_split, feature_mapping, CensusIncome.class_mapping, knowledge

    @staticmethod
    def breast_data():

        train = pd.read_csv(DATA_PATH / "breast-cancer-data.csv")
        test = pd.read_csv(DATA_PATH / "breast-cancer-data-test.csv")
        feature_mapping = {k: v for k, v in zip(train.columns, list(range(len(train.columns))))}
        knowledge = TuProlog.from_file(str(KNOWLEDGE_PATH / "breast-cancer.pl")).formulae
        dataset_split = ExperimentSKIQOS.split_dataset(train, test)
        return dataset_split, feature_mapping, BreastCancer.class_mapping_short, knowledge


def create_standard_fully_connected_nn(input_size: int, output_size, layers: int, neurons: int, activation: str) -> Model:
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    return Model(inputs, x)


if __name__ == '__main__':
    # flags = dict(energy=True, latency=True, memory=True, grid_search=False)
    # arguments = dict(optimizer='sgd', loss='sparse_categorical_crossentropy', batch=32, epochs=100, metric='accuracy',
    #                  threshold=0.8, max_neurons_width=[500, 200, 100], max_neurons_depth=100, max_layers=8, grid_levels=4)
    # ExperimentSKIQOS(dataset_name='breast', injector='kins', hyper=arguments, flags=flags).test_qos()
    data = pd.read_csv(DATA_PATH / 'breast-cancer-data.csv')
    formulae = TuProlog.from_file(KNOWLEDGE_PATH / 'breast-cancer.pl').formulae
    model = create_standard_fully_connected_nn(data.shape[1]-1, 2, 3, 1000, 'relu')
    model.compile('adam', loss='categorical_crossentropy', metrics='accuracy')
    injector = Injector.kbann(model, {k: v for k, v in zip(data.columns[:-1], list(range(len(data.columns[:-1]))))})
    new_model: Model = injector.inject(formulae)
    new_model.compile('adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    new_model.fit(data.iloc[:, :-1], data.iloc[:, -1:], epochs=100, batch_size=32, verbose=1)
