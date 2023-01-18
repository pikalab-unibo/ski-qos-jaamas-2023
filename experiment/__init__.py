import unittest
import pandas as pd

from psyki.qos import QoS
import numpy as np
from psyki.qos.utils import split_dataset
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from datasets import load_splice_junction_dataset, load_breast_cancer_dataset, load_census_income_dataset, \
    SpliceJunction, BreastCancer, CensusIncome
from knowledge import get_census_income_knowledge, get_splice_junction_knowledge, get_breast_cancer_knowledge


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

        qos = QoS(metric_arguments=metric_arguments,
                  flags=self.flags)
        qos.compute(verbose=True)

    @staticmethod
    def splice_data():

        splice_junction_train = pd.read_csv('../datasets/splice-junction-data.csv')
        splice_junction_test = pd.read_csv('../datasets/splice-junction-data-test.csv')
        feature_mapping = {k: v for k, v in
                           zip(splice_junction_train.columns, list(range(len(splice_junction_train.columns))))}

        splice_junction_knowledge = get_splice_junction_knowledge()

        train_x, train_y = splice_junction_train.iloc[:, :-1], splice_junction_train.iloc[:, -1]
        test_x, test_y = splice_junction_test.iloc[:, :-1], splice_junction_test.iloc[:, -1]

        dataset_split = dict(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=np.max(train_y) + 1)

        return dataset_split, feature_mapping, SpliceJunction.class_mapping_short, splice_junction_knowledge

    @staticmethod
    def census_data():

        census_income_train = pd.read_csv('../datasets/census-income-data.csv')
        census_income_test = pd.read_csv('../datasets/census-income-data-test.csv')
        feature_mapping = {k: v for k, v in
                           zip(census_income_train.columns, list(range(len(census_income_train.columns))))}

        census_income_knowledge = get_census_income_knowledge()
        train_x, train_y = census_income_train.iloc[:, :-1], census_income_train.iloc[:, -1]
        test_x, test_y = census_income_test.iloc[:, :-1], census_income_test.iloc[:, -1]

        dataset_split = dict(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=np.max(train_y) + 1)

        return dataset_split, feature_mapping, CensusIncome.class_mapping, census_income_knowledge

    @staticmethod
    def breast_data():

        breast_cancer_train = pd.read_csv('../datasets/breast-cancer-data.csv')
        breast_cancer_test = pd.read_csv('../datasets/breast-cancer-data-test.csv')
        feature_mapping = {k: v for k, v in
                           zip(breast_cancer_train.columns, list(range(len(breast_cancer_train.columns))))}

        breast_cancer_knowledge = get_breast_cancer_knowledge()

        train_x, train_y = breast_cancer_train.iloc[:, :-1], breast_cancer_train.iloc[:, -1]
        test_x, test_y = breast_cancer_test.iloc[:, :-1], breast_cancer_test.iloc[:, -1]

        dataset_split = dict(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             input_size=train_x.shape[-1],
                             output_size=np.max(train_y) + 1)

        return dataset_split, feature_mapping, BreastCancer.class_mapping_short, breast_cancer_knowledge


def create_standard_fully_connected_nn(input_size: int, output_size, layers: int, neurons: int,
                                       activation: str) -> Model:
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    return Model(inputs, x)


if __name__ == '__main__':
    flags = dict(energy=True,
                 latency=True,
                 memory=True,
                 grid_search=False)

    arguments = dict(optimizer='sgd',
                     loss='sparse_categorical_crossentropy',
                     batch=16,
                     epochs=10,
                     metric='accuracy',
                     threshold=0.8,
                     max_neurons_width=[500, 200, 100],
                     max_neurons_depth=100,
                     max_layers=8,
                     grid_levels=4)

    ExperimentSKIQOS(dataset_name='breast',
                     injector='kins',
                     hyper=arguments,
                     flags=flags).test_qos()
