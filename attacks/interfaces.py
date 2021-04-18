from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import List, Tuple, Dict

import numpy as np

SampleResult = namedtuple('SampleResult', 'attack loss success_flag')  # attack is a List[str], loss a float in [0,1] and success_flag: bool


class AttackStrategy(metaclass=ABCMeta):

    def __init__(self):
        self.orig_sentence = None
        self.orig_label = None
        self.exploration_budget = None

    @abstractmethod
    def _reset(self):
        raise NotImplementedError

    def init_attack(self, orig_sentence: List[str], label: int, exploration_budget: int, replace_orig_if_needed=False):
        self.orig_sentence = orig_sentence
        self.orig_label = label
        self.exploration_budget = exploration_budget
        self._reset()

    def attack_space_size(self) -> Tuple[int, int]:
        if hasattr(self, '_attack_space_size') and self._attack_space_size is not None and self._attack_space_size > 0:
            return int(self._attack_space_size), int(self._attack_positions_size)
        raise NotImplementedError

    @abstractmethod
    def explore(self, batch_size: int) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, samples: List[SampleResult]):
        raise NotImplementedError

    @abstractmethod
    def exploit(self, batch_size: int) -> List[List[str]]:
        raise NotImplementedError


class Model(metaclass=ABCMeta):

    @abstractmethod
    def predict_proba(self, samples: List[List[str]]) -> np.array:
        raise NotImplementedError


class Dataset(metaclass=ABCMeta):

    @abstractmethod
    def shuffle(self, seed: int):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def get_sentence_to_perturb(self, i: int) -> Tuple[List[str], int]:
        raise NotImplementedError

    @abstractmethod
    def get_unsplitted_sentence_to_perturb(self, i: int) -> str:
        raise NotImplementedError

    def is_two_inputs_dataset(self):
        return False

    def get_second_input(self, i: int) -> str:
        if not self.is_two_inputs_dataset():
            raise TypeError('This dataset does not have multiple inputs')
        raise NotImplementedError

    @abstractmethod
    def generate_perturbed_sample(self, i: int, perturbed_sentence: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_labels(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_orig_row(self, i: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def create_augmented_dataset(self, attacks: Dict[int, str], out_path: str):
        raise NotImplementedError
