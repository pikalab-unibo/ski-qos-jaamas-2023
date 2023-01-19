import csv
import re
from pathlib import Path
import pandas as pd
from psyke.utils.logic import pretty_theory
from psyki.logic.datalog import get_formula_from_string
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog
from psyki.ski import Formula
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from psyke import Extractor
from datasets import PATH as DATA_PATH, SpliceJunction


PATH = Path(__file__).parents[0]


MAX_FEATURES_IN_RULE: int = 20
MAX_RULES: int = 20

# Ad-hoc symbols for the provided knowledge for splice junction
VARIABLE_BASE_NAME = 'X'
AND_SYMBOL = ' , '
OR_SYMBOL = ' ; '
NOT_SYMBOL = '¬'
LESS_EQUAL_SYMBOL = ' =< '
PLUS_SYMBOL = ' + '
STATIC_IMPLICATION_SYMBOL = ' <- '
MUTABLE_IMPLICATION_SYMBOL = ' <-- '
STATIC_RULE_SYMBOL = '::-'
MUTABLE_RULE_SYMBOL = ':-'
INDEX_IDENTIFIER = '@'
NOT_IDENTIFIER = 'not'
RULE_DEFINITION_SYMBOLS = (STATIC_RULE_SYMBOL, MUTABLE_RULE_SYMBOL)
RULE_DEFINITION_SYMBOLS_REGEX = '(' + '|'.join(RULE_DEFINITION_SYMBOLS) + ')'


def generate_missing_knowledge():

    def generate_knowledge(data_name):
        data = pd.read_csv(DATA_PATH / (data_name + '-data.csv'))
        predictor = DecisionTreeClassifier(max_depth=MAX_FEATURES_IN_RULE, max_leaf_nodes=MAX_RULES)
        train, _ = train_test_split(data, random_state=0, train_size=0.5)
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        predictor.fit(train_x, train_y)
        extractor = Extractor.cart(predictor, max_depth=MAX_FEATURES_IN_RULE, max_leaves=MAX_RULES)
        knowledge = extractor.extract(train)
        if data_name == 'breast-cancer':
            textual_knowledge = pretty_theory(knowledge)
            textual_knowledge = re.sub(r"([A-Z][a-zA-Z0-9]*)[ ]>[ ]([+-]?([0-9]*))[.]?[0-9]+", r"\g<1>",
                                       textual_knowledge)
            textual_knowledge = re.sub(r"([A-Z][a-zA-Z0-9]*)[ ]=<[ ]([+-]?([0-9]*))[.]?[0-9]+", r"not(\g<1>)",
                                       textual_knowledge)
            textual_knowledge = re.sub(r"(diagnosis)\((.*, )('1')\)", r"class(\g<2>malignant)", textual_knowledge)
            textual_knowledge = re.sub(r"(diagnosis)\((.*, )('0')\)", r"class(\g<2>benign)", textual_knowledge)
        else:
            textual_knowledge = pretty_theory(knowledge)
        with open(PATH / (data_name + ".txt"), "w") as text_file:
            text_file.write(textual_knowledge)

    generate_knowledge('census-income')
    generate_knowledge('breast-cancer')


def get_census_income_knowledge() -> list[Formula]:
    return prolog_to_datalog(file_to_prolog(str(PATH / 'census-income.txt')), trainable=True)


def get_breast_cancer_knowledge() -> list[Formula]:
    return prolog_to_datalog(file_to_prolog(str(PATH / 'breast-cancer.txt')), trainable=True)


def _get_knowledge(filename: str) -> list[str]:
    result = []
    with open(filename, mode="r", encoding="utf8") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return result


def get_splice_junction_knowledge() -> list[Formula]:
    return parse_splice_junction_knowledge()


# Ad-hoc parse function for the prior knowledge of the splice junction domain
def parse_splice_junction_clause(rest: str, rhs: str = '', aggregation: str = AND_SYMBOL) -> str:
    def next_index(i: str, indices: list[int], offset: int) -> int:
        new_index: int = int(i) + offset
        modified: bool = False
        while new_index not in indices:
            new_index += 1
            modified = True
        return new_index + previous_holes(indices, indices.index(new_index)) if not modified else new_index

    def previous_holes(l: list[int], i: int) -> int:
        j = 0
        for k in list(range(0, i)):
            if l[k] + 1 != l[k + 1]:
                j += 1
        return j

    def explicit_variables(e: str) -> str:
        result = ''
        for key in SpliceJunction.aggregate_features.keys():
            if key.lower() in e:
                values = [v for v in SpliceJunction.aggregate_features[key]]
                if len(values) > 1:
                    result += AND_SYMBOL.join(
                        NOT_SYMBOL + '(' + re.sub(key.lower(), value.lower(), e) + ')' for value in values)
        return NOT_SYMBOL + '(' + result + ')' if result != '' else e

    for j, clause in enumerate(rest.split(',')):
        index = re.match(INDEX_IDENTIFIER + '[-]?[0-9]*', clause)
        negation = re.match(NOT_IDENTIFIER, clause)
        n = re.match('[0-9]*of', clause)
        if index is not None:
            index = clause[index.regs[0][0]:index.regs[0][1]]
            clause = clause[len(index):]
            clause = re.sub('\'', '', clause)
            index = index[1:]
            rhs += aggregation.join(explicit_variables(
                VARIABLE_BASE_NAME + ('_' if next_index(index, list(range(-30, 0)) + list(range(1, 31)), i) < 0 else '') +
                str(abs(next_index(index, list(range(-30, 0)) + list(range(1, 31)), i))) +
                ' = ' + value.lower()) for i, value in enumerate(clause))
        elif negation is not None:
            new_clause = re.sub(NOT_IDENTIFIER, NOT_SYMBOL, clause)
            new_clause = re.sub('-', '_', new_clause.lower())
            new_clause = re.sub('\)', '())', new_clause)
            rhs += new_clause
        elif n is not None:
            new_clause = clause[n.regs[0][1]:]
            new_clause = re.sub('\(|\)', '', new_clause)
            inner_clause = parse_splice_junction_clause(new_clause, rhs, PLUS_SYMBOL)
            inner_clause = '(' + ('), (').join(e for e in inner_clause.split(PLUS_SYMBOL)) + ')'
            n = clause[n.regs[0][0]:n.regs[0][1] - 2]
            rhs += 'm_of_n(' + n + ', ' + inner_clause + ')'
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs


def parse_splice_junction_knowledge() -> list[Formula]:
    rules = []
    file = PATH / 'splice-junction.txt'
    with open(file) as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            if len(raw) > 0:
                rules.append(raw)
    new_rules = []
    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, op, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        rhs = parse_splice_junction_clause(rest)
        if name in SpliceJunction.class_mapping_short.keys():
            new_rules.append('class(' + name + ')' + (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
        new_rules.append(name + '(' + ')' + (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
    results = []
    term_regex = '[a-z]+'
    variable_regex = VARIABLE_BASE_NAME + '[_]?[0-9]+'
    regex = variable_regex + '[ ]?=[ ]?' + term_regex
    for rule in new_rules:
        tmp_rule = rule
        partial_result = ''
        while re.search(regex, tmp_rule) is not None:
            match = re.search(regex, tmp_rule)
            start, end = match.regs[0]
            matched_string = tmp_rule[start:end]
            ante = tmp_rule[:start]
            medio = matched_string[:re.search(variable_regex, matched_string).regs[0][1]] + \
                    matched_string[re.search(term_regex, matched_string).regs[0][0]:]
            partial_result += ante + medio
            tmp_rule = tmp_rule[end:]
        partial_result += tmp_rule
        results.append(partial_result)
    return [get_formula_from_string(rule) for rule in results]
