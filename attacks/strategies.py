import os

import numpy as np
import warnings
from abc import abstractmethod
from copy import deepcopy
from itertools import chain
from typing import List
import operator as op

from attacks.interfaces import AttackStrategy, SampleResult
from attacks.mcts_strategy import WideSynonymSwapMCTSStrategy, DeepSynonymSwapMCTSStrategy
from attacks.synonyms_utils import load_synonym_for_attacks, synonym_file_short_name, _get_synonyms_attack_space_size
from attacks.text_fooler_filters import get_filter
from common.utils import pertrub_sentence, get_possible_perturbations, Heap


class RandomAttackStrategy(AttackStrategy):

    def __init__(self):
        super().__init__()
        self._results = None

    @abstractmethod
    def get_perturbed_sentences(self, batch_size: int) -> List[List[str]]:
        pass

    def _reset(self):
        self._results = {}

    def explore(self, batch_size: int) -> List[List[str]]:
        # TODO - need to make sure doesn't repeat the same explorations as before
        return self.get_perturbed_sentences(batch_size)

    def update(self, samples: List[SampleResult]):
        self._results.update({loss: key for key, loss, sf in samples})

    def exploit(self, batch_size: int) -> List[List[str]]:
        # TODO - what happens if there are no results at all? shuold at least fall back to give the original value?
        if batch_size <= len(self._results):
            return [v for loss, v in sorted(self._results.items(), key=op.itemgetter(0))][:batch_size]
        else:
            return list(chain(self._results.values(), self.explore(batch_size - len(self._results))))


class RandomSynonymSwapAttackStrategy(RandomAttackStrategy):
    def __init__(self, synonyms_path: str, edit_distance: int, allow_smaller_eds=True, verbose=False, dataset=None, train=None, *,
                 force_budget=None, **_):
        super().__init__()
        self._synonyms, self._requires_pos = load_synonym_for_attacks(synonyms_path, dataset, train)
        self._syn_name = synonym_file_short_name(synonyms_path)
        self.edit_distance = edit_distance
        self.allow_smaller_eds = allow_smaller_eds
        self._verbose = verbose
        self._force_budget = force_budget

    def init_attack(self, orig_sentence: List[str], label: int, exploration_budget: int, replace_orig_if_needed=False):
        if self._force_budget is not None:
            exploration_budget = min(self._force_budget, exploration_budget)
        super().init_attack(orig_sentence, label, exploration_budget, replace_orig_if_needed)

        # this way we will not explore the same attack multiple times
        self._attacks_list = []
        orig_ed = self.edit_distance
        if 0 < orig_ed < 1:  # dynamic edit distance
            orig_ed = max(1, int(len(orig_sentence) * orig_ed))
        ed = orig_ed
        perturbed_sentences = self._attacks_list
        if ed == -1:
            perturbed_sentences.extend((s.split(' ') for s in
                                        chain.from_iterable(pertrub_sentence(self.orig_sentence, 0,
                                                                             exploration_budget - len(perturbed_sentences),
                                                                             self._synonyms, ed, False, True, False,
                                                                             self._requires_pos,
                                                                             replace_orig_if_needed=replace_orig_if_needed)[0].values())))
        while ed > 0 and len(perturbed_sentences) < exploration_budget:
            perturbed_sentences.extend((s.split(' ') for s in
                                        chain.from_iterable(pertrub_sentence(self.orig_sentence, 0,
                                                                             exploration_budget - len(perturbed_sentences),
                                                                             self._synonyms, ed, False, True, False,
                                                                             self._requires_pos,
                                                                             replace_orig_if_needed=replace_orig_if_needed)[0].values())))
            ed -= 1
            if not self.allow_smaller_eds:
                ed = 0
        if self._verbose:
            if len(perturbed_sentences) == 0:
                # raise RuntimeError("Couldn't perturb the sentence even once!")
                warnings.warn('Could not perturb given sample even once')
            elif len(perturbed_sentences) < exploration_budget:
                print(f'Failed to get enough perturbation for the sample (requested {exploration_budget}, got {len(perturbed_sentences)})')
        self._attack_space_size, self._attack_positions_size = _get_synonyms_attack_space_size(self.orig_sentence, orig_ed,
                                                                                              self._synonyms, self._requires_pos, True,
                                                                                               syn_name=self._syn_name)

    def get_perturbed_sentences(self, batch_size: int) -> List[List[str]]:
        n_attacks = min(batch_size, len(self._attacks_list))
        if n_attacks == 0:
            return []
        attacks, self._attacks_list = self._attacks_list[:n_attacks], self._attacks_list[n_attacks:]
        return attacks

    def __str__(self):
        return f'Random Synonym Swap, ED={self.edit_distance}, Syn={self._syn_name}' + \
               (f', Budget={self._force_budget}' if self._force_budget is not None else '')


class RandomSynonymSwapAttackStrategyED10(RandomSynonymSwapAttackStrategy):
    def __init__(self, dataset=None, train=None, *, force_budget=None, **_):
        super().__init__(synonyms_path=None, edit_distance=0.1, allow_smaller_eds=True, verbose=False, dataset=dataset, train=train,
                         force_budget=force_budget, **_)

    def __str__(self):
        return f'RandomSynonymSwapAttackStrategyED10'


class GreedySynonymSwapAttackStrategy(AttackStrategy):
    """
    Greedy attack with early-stopping (i.e. stop looking as soon as it finds a valid attack, regardless the leftover budget or the
    reached edit-distance)
    """

    def __init__(self, synonyms_path: str, edit_distance: int, early_stop: bool = False, dataset=None, train=None, **_):
        super().__init__()
        self._reset()
        self._synonyms, self._requires_pos = load_synonym_for_attacks(synonyms_path, dataset, train)
        self._syn_name = synonym_file_short_name(synonyms_path)
        self.edit_distance = edit_distance
        self._early_stop = early_stop

    def _reset(self):
        # self._possible_modifications = None  # since we are now computing it before _reset is called
        self._current_node = None
        self._attacks_planner = None
        self._best_loss = None
        self._best_attack = None
        self._current_ed = 0
        self._attack_specific_ed = 0

    def _get_planner(self):
        if len(self._possible_modifications) == 0:
            return []
        return [(k, v[i]) for i in range(max(len(v_) for v_ in self._possible_modifications.values()))
                                  for k, v in self._possible_modifications.items() if i < len(v)]  # TODO - can stop at exploration_budget..

    def init_attack(self, orig_sentence: List[str], label: int, exploration_budget: int, replace_orig_if_needed=False):
        self._possible_modifications = None
        possible_modifications = get_possible_perturbations(orig_sentence, self._synonyms, self._requires_pos,
                                                            append_self=replace_orig_if_needed, raise_on_failure=not replace_orig_if_needed)
        if replace_orig_if_needed:
            orig_sentence = [p[-1] for p in possible_modifications]  # override the sentence
            possible_modifications = [p[:-1] for p in possible_modifications]  # remove the last word
        elif len(possible_modifications) != len(orig_sentence):
            # This happens 8 times in sst2 train data and occur in offline augmentation if replace_orig_if_needed is False
            print(f'Warning Found a matching signature but the perturbations do not seem to match the length of the sentence so '
                  f'returning no perturbations: {" ".join(orig_sentence)}')
            possible_modifications = [[] for _ in range(len(orig_sentence))]
        self._possible_modifications = {i: ps for i, ps in enumerate(possible_modifications) if len(ps) > 0}
        super().init_attack(orig_sentence, label, exploration_budget, replace_orig_if_needed)

        self._current_node = deepcopy(orig_sentence)
        self._attacks_planner = self._get_planner()
        self._attack_specific_ed = self.edit_distance
        if self._attack_specific_ed == -1:
            self._attack_specific_ed = len(orig_sentence)  # we allow it to change everything
        if 0 < self._attack_specific_ed < 1:  # dynamic edit distance
            self._attack_specific_ed = max(1, int(len(orig_sentence) * self._attack_specific_ed))
        # this is actually not a lower bound since it assumes a specific ed, but it can do anything in between with allow_smaller_ed
        self._attack_space_size, self._attack_positions_size = _get_synonyms_attack_space_size(self.orig_sentence, self._attack_specific_ed,
                                                                                               self._synonyms, self._requires_pos, True,
                                                                                               syn_name=self._syn_name)

    def _perturb(self, index, syn):
        attack = deepcopy(self._current_node)
        attack[index] = syn
        return attack

    def explore(self, batch_size: int) -> List[List[str]]:
        attacks = []
        if self._best_loss is None:
            attacks.append(self._current_node)
        attacks.extend(self._perturb(*v) for _, v in zip(range(batch_size-len(attacks)), self._attacks_planner))
        # Note - there is an option here that the explore size is less than the batch size if we don't have enough candidates left
        # in this level
        self._last_attacks = attacks
        return attacks

    def _extend_attack_planner_on_update(self):
        # there is an option here that current_node will be the same as best_attack which means the greedy search is done
        # TODO - make sure this doesn't cause an early stopping before reaching maximal allowed ed
        if self._current_node != self._best_attack:
            # done!
            change_ind = [i for i, (w1, w2) in enumerate(zip(self._current_node, self._best_attack)) if w1 != w2][0]
            self._current_node = self._best_attack
            self._current_ed += 1
            if self._current_ed < self._attack_specific_ed:
                del self._possible_modifications[change_ind]  # this keeps us from modifying the same word multiple times
                self._attacks_planner = self._get_planner()

    def update(self, samples: List[SampleResult]):
        # we assume here that the update matches EXACTLY to the last exploration batch. if not, strange things will happen.. beware
        assert len(self._last_attacks) == len(samples), f'{len(self._last_attacks)}, {len(samples)}'
        if self._best_loss is None:
            self._best_loss = samples[0][1]
            self._best_attack = samples[0][0]  # this is the base sample
        best_attack, attack_loss, sf = min(samples, key=op.itemgetter(1))
        if attack_loss < self._best_loss:
            self._best_loss = attack_loss
            self._best_attack = best_attack
        if sf and self._early_stop:
            self._attacks_planner = []  # no need to explore further
        else:
            self._attacks_planner = self._attacks_planner[len(samples):]
            if len(self._attacks_planner) == 0:
                self._extend_attack_planner_on_update()

    def exploit(self, batch_size: int) -> List[List[str]]:
        attacks = [self._best_attack]
        # can hope that maybe one of the left candidates are good attacks, anything before would be worse than the best one by definition
        attacks.extend(self._perturb(*v) for _, v in zip(range(batch_size-len(attacks)), self._attacks_planner))
        return attacks

    def __str__(self):
        return f'Greedy Synonym Swap, ED={self.edit_distance}, Syn={self._syn_name}, ES={self._early_stop}'


class InformedGreedySynonymSwapAttackStrategy(AttackStrategy):
    """
    Informed Greedy attack with early-stopping (i.e. stop looking as soon as it finds a valid attack, regardless the leftover budget or the
    reached edit-distance)
    """

    # Note - this may explore the same things over and over (since this is actually a DAG, not a tree)
    REMOVE_WORD = -1

    @staticmethod
    def _get_node_ctor():
        class Node:
            @classmethod
            def init_nodes(cls, orig_sentence, perturbations, ed, method):
                cls.orig_sentence = orig_sentence
                cls.perturbations = perturbations
                cls.ed = ed
                cls.words_to_pertrub = set(cls.perturbations.keys())
                cls.method = method

            def __init__(self, attack_definition):
                self.attack_definition = attack_definition
                self.loss = np.inf
                self._sig = frozenset(attack_definition)
                # self._sig = '-'.join(map(str, chain.from_iterable(attack_definition)))

            def pertrub(self) -> List[str]:
                pert_sent = deepcopy(self.orig_sentence)
                for word_i, attack_i in self.attack_definition:
                    if attack_i != InformedGreedySynonymSwapAttackStrategy.REMOVE_WORD:
                        pert_sent[word_i] = self.perturbations[word_i][attack_i]
                    elif self.method == 'remove':
                        pert_sent[word_i] = " "
                    elif self.method == 'random':
                        # TODO - can use this to save some computations for later
                        pert_sent[word_i] = np.random.choice(self.perturbations[word_i], 1)[0]
                    elif self.method == 'mask':
                        pert_sent[word_i] = "MASK"
                    elif self.method == 'text-fooler':
                        pert_sent[word_i] = "<oov>"
                return pert_sent

            def expand(self, words_priority=None):
                # return its children
                if not self.is_expandable():
                    return []
                elif self.is_valid_node():
                    words_left_to_perturb = self.words_to_pertrub.difference({k for k, _ in self.attack_definition})
                    if words_priority is None:
                        # next nodes will try to remove words (to prioritize the next step)
                        return [type(self)(self.attack_definition + ((word_i, InformedGreedySynonymSwapAttackStrategy.REMOVE_WORD), ))
                                for word_i in words_left_to_perturb]
                    else:
                        # we are in the single priority mode so we can directly create valid nodes now
                        next_word_pertrub = None
                        for nwp in words_priority:
                            if nwp in words_left_to_perturb:
                                next_word_pertrub = nwp
                                break
                        if next_word_pertrub is None:
                            return []  # can't continue from this node (exhausted)
                        return [type(self)(self.attack_definition + ((next_word_pertrub, attack_i),))
                                for attack_i in range(len(self.perturbations[next_word_pertrub]))]

                else:
                    # next nodes will try to attack the chosen nodes
                    attacked_word = self.attack_definition[-1][0]
                    if attacked_word not in self.perturbations:
                        raise RuntimeError('WTF??')
                    return [type(self)(self.attack_definition[:-1] + ((attacked_word, attack_i),))
                            for attack_i in range(len(self.perturbations[attacked_word]))]

            def is_valid_node(self) -> bool:
                return self.attack_definition[-1][1] != InformedGreedySynonymSwapAttackStrategy.REMOVE_WORD

            def is_expandable(self) -> bool:
                return not self.is_valid_node() or len(self.attack_definition) < self.ed

            def signature(self):
                return self._sig
        return Node

    def __init__(self, synonyms_path: str, edit_distance: int, beam_size=1, method='random', early_stop: bool = False,
                 prioritize_once: bool = False, perform_semantic_filtering: bool = False, perform_pos_filtering: bool = False,
                 sim_score_window: int = 15, sim_score_threshold: float = 0.7, backtracking=True, dataset=None, train=None, name=None, **_):
        # possible methods: remove, mask, random, text-fooler
        # if prioritize_once is set, only performs the informing step at the start, and then re-use this information on subsequent steps
        assert method in ('remove', 'mask', 'random', 'text-fooler')
        super().__init__()
        self._reset()
        self.node_ctor = self._get_node_ctor()
        self._synonyms, self._requires_pos = load_synonym_for_attacks(synonyms_path, dataset, train)
        self._syn_name = synonym_file_short_name(synonyms_path)
        self.edit_distance = edit_distance
        assert beam_size >= 1
        self.beam_size = beam_size
        self.method = method
        self._early_stop = early_stop
        self._prioritize_once = prioritize_once
        self._perform_filtering = perform_semantic_filtering or perform_pos_filtering
        self._perform_semantic_filtering = perform_semantic_filtering
        self._perform_pos_filtering = perform_pos_filtering
        self._sim_score_window = sim_score_window
        self._sim_score_threshold = sim_score_threshold
        self._backtracking = backtracking
        self._name = name

    def _reset(self):
        self._is_drop_phase = True
        self._next_attacks = None
        self._frontier_heap = Heap(key=op.itemgetter(1), max_heap=False)  # min heap on the loss
        self._attacks_heap = Heap(key=op.itemgetter(1), max_heap=False)  # min heap on the loss
        self._explored_nodes = set()
        self._attack_specific_ed = 0
        self._should_early_stop = False
        self._words_priority = None
        self._attacks_filter = None

    def init_attack(self, orig_sentence: List[str], label: int, exploration_budget: int, replace_orig_if_needed=False):
        possible_modifications = get_possible_perturbations(orig_sentence, self._synonyms, self._requires_pos,
                                                            append_self=replace_orig_if_needed, raise_on_failure=not replace_orig_if_needed)
        if replace_orig_if_needed:
            orig_sentence = [p[-1] for p in possible_modifications]  # override the sentence
            possible_modifications = [p[:-1] for p in possible_modifications]  # remove the last word
        elif len(possible_modifications) != len(orig_sentence):
            # This happens 8 times in sst2 train data and occur in offline augmentation if replace_orig_if_needed is False
            print(f'Warning Found a matching signature but the perturbations do not seem to match the length of the sentence so '
                  f'returning no perturbations: {" ".join(orig_sentence)}')
            possible_modifications = [[] for _ in range(len(orig_sentence))]
        self._possible_modifications = {i: ps for i, ps in enumerate(possible_modifications) if len(ps) > 0}
        super().init_attack(orig_sentence, label, exploration_budget, replace_orig_if_needed)

        self._attack_specific_ed = self.edit_distance
        if self._attack_specific_ed == -1:
            self._attack_specific_ed = len(orig_sentence)  # we allow it to change everything
        if 0 < self._attack_specific_ed < 1:  # dynamic edit distance
            self._attack_specific_ed = max(1, int(len(orig_sentence) * self._attack_specific_ed))
        self.node_ctor.init_nodes(orig_sentence, self._possible_modifications, self._attack_specific_ed, self.method)
        self._next_attacks = [self.node_ctor(((word_i, self.REMOVE_WORD), )) for word_i in self._possible_modifications.keys()]
        self._last_attacks = None
        np.random.shuffle(self._next_attacks)
        self._attack_space_size, self._attack_positions_size = \
            _get_synonyms_attack_space_size(self.orig_sentence, self._attack_specific_ed, self._synonyms, self._requires_pos, True,
                                            syn_name=self._syn_name)
        if self._perform_filtering:
            self._attacks_filter = get_filter(filter_by_tf_pos=self._perform_pos_filtering, filter_by_sem=self._perform_semantic_filtering,
                                              orig_sent=orig_sentence, return_mask=True, tfhub_cache_path=None,
                                              sim_score_window=self._sim_score_window, sim_score_threshold=self._sim_score_threshold)
            # no need to filter attacks here as they are all one of the informed parts

    def explore(self, batch_size: int) -> List[List[str]]:
        if self._early_stop and self._should_early_stop:
            return []
        if len(self._next_attacks) == 0:
            if len(self._frontier_heap) == 0:
                return []  # nothing left to explore
            # plan next attacks
            if self._prioritize_once and self._words_priority is None:
                # we are right after the original expansion so the only explored nodes are those that test the saliency of each of the
                # replaceable words. each item in the heap is a tuple of an attack node and its loss. the heap will return with
                # ascending order (of loss). The attack node will have an attack_definition of length 1, with a tuple of type
                # (word_idx, -1). we want the word index
                self._words_priority = [node[0].attack_definition[0][0] for node in self._frontier_heap.nlargest(len(self._frontier_heap))]

            frontier_size = min(len(self._frontier_heap), self.beam_size)
            frontier = [self._frontier_heap.pop()[0] for _ in range(frontier_size)]
            if not self._backtracking:
                self._frontier_heap.clear()
            self._next_attacks = {attack.signature(): attack
                                  for frontier_node in frontier
                                  for attack in frontier_node.expand(self._words_priority)
                                  if attack.signature() not in self._explored_nodes}  # we may create duplicates here
            # filter attacks based on semantic similarity
            self._next_attacks = list(self._next_attacks.values())
            if self._perform_filtering:
                attacks_inds_to_filter = [i for i, attack in enumerate(self._next_attacks) if attack.is_valid_node()]
                attacks_to_filter = [self._next_attacks[i].pertrub() for i in attacks_inds_to_filter]
                attacks_last_word_ind = [self._next_attacks[i].attack_definition[-1][0] for i in attacks_inds_to_filter]
                mask_dict = dict(zip(attacks_inds_to_filter, self._attacks_filter(attacks_to_filter, attacks_last_word_ind)))
                self._next_attacks = [attack for i, attack in enumerate(self._next_attacks) if mask_dict.get(i, True)]

            np.random.shuffle(self._next_attacks)
        effective_bs = min(batch_size, len(self._next_attacks))
        self._last_attacks, self._next_attacks = self._next_attacks[:effective_bs], self._next_attacks[effective_bs:]
        return [attack.pertrub() for attack in self._last_attacks]

    def update(self, samples: List[SampleResult]):
        # we assume here that the update matches EXACTLY to the last exploration batch. if not, strange things will happen.. beware
        assert len(self._last_attacks) == len(samples)
        for attack_node, (attack, loss, sf) in zip(self._last_attacks, samples):
            self._explored_nodes.add(attack_node.signature())
            if attack_node.is_valid_node():
                self._attacks_heap.push((attack, loss))
                self._should_early_stop = self._should_early_stop or sf
            if attack_node.is_expandable():
                self._frontier_heap.push((attack_node, loss))
        self._last_attacks = None

    def exploit(self, batch_size: int) -> List[List[str]]:
        effective_bs = min(batch_size, len(self._attacks_heap))  # we may even be here before performing a single valid attack
        return [ps for ps, _ in self._attacks_heap.nlargest(effective_bs)] + \
               [deepcopy(self.orig_sentence) for _ in range(batch_size-effective_bs)]

    def __str__(self):
        if self._name is not None:
            return self._name
        sem_sim_str = 'NoFilter' if not self._perform_filtering else \
            f'Window={self._sim_score_window}, TH={self._sim_score_threshold}'
        return f'Informed Greedy Synonym Swap, ED={self.edit_distance}, BS={self.beam_size}, ES={self._early_stop}, Method={self.method}, ' \
            f'Syn={self._syn_name}, BT={self._backtracking}, PO={self._prioritize_once}, {sem_sim_str}'


class TextFoolerAttackStrategy(InformedGreedySynonymSwapAttackStrategy):
    _syn_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', 'text_fooler_synonyms.json')
    def __init__(self, edit_distance: int = -1, **_):
        super().__init__(synonyms_path=self._syn_path, edit_distance=edit_distance, beam_size=1, backtracking=False,
                         method='text-fooler', early_stop=True, prioritize_once=True, perform_semantic_filtering=True, sim_score_window=15,
                         sim_score_threshold=0.7, perform_pos_filtering=True)

    def __str__(self):
        return 'TextFoolerAttackStrategy'

class TextFoolerLikeAttackStrategyED10(InformedGreedySynonymSwapAttackStrategy):
    def __init__(self, dataset=None, train=None, **_):
        super().__init__(synonyms_path=None, edit_distance=0.1, beam_size=1, backtracking=False,
                         method='text-fooler', early_stop=True, prioritize_once=True, perform_semantic_filtering=False, sim_score_window=15,
                         sim_score_threshold=0.7, perform_pos_filtering=False, dataset=dataset, train=train)

    def __str__(self):
        return 'TextFoolerLikeAttackStrategyED10'


class InformedGreedyDefaultAttackStrategy(InformedGreedySynonymSwapAttackStrategy):
    _syn_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', 'text_fooler_synonyms.json')
    def __init__(self, edit_distance: int = -1, **_):
        super().__init__(synonyms_path=self._syn_path, edit_distance=edit_distance, beam_size=1, backtracking=True,
                         method='random', early_stop=True, prioritize_once=False, perform_semantic_filtering=True, sim_score_window=15,
                         sim_score_threshold=0.7, perform_pos_filtering=True)

    def __str__(self):
        return 'InformedGreedyAttackStrategy'


class NewInformedGreedyAttackStrategy(InformedGreedySynonymSwapAttackStrategy):
    def __init__(self, edit_distance: int = -1, dataset=None, train=None, **_):
        super().__init__(synonyms_path=None, edit_distance=edit_distance, beam_size=2, backtracking=True,
                         method='random', early_stop=True, prioritize_once=False, perform_semantic_filtering=False, sim_score_window=15,
                         sim_score_threshold=0.7, perform_pos_filtering=False, dataset=dataset, train=train)

    def __str__(self):
        return 'NewInformedGreedyAttackStrategy'


class IGBeam2ED10(InformedGreedySynonymSwapAttackStrategy):
    def __init__(self, edit_distance: int = 0.1, dataset=None, train=None, **_):
        super().__init__(synonyms_path=None, edit_distance=edit_distance, beam_size=2, backtracking=True,
                         method='random', early_stop=True, prioritize_once=False, perform_semantic_filtering=False,
                         perform_pos_filtering=False, dataset=dataset, train=train)

    def __str__(self):
        return 'IGBeam2ED10'


class IGBeam1ED10(InformedGreedySynonymSwapAttackStrategy):
    def __init__(self, edit_distance: int = 0.1, dataset=None, train=None, **_):
        super().__init__(synonyms_path=None, edit_distance=edit_distance, beam_size=1, backtracking=True,
                         method='random', early_stop=True, prioritize_once=False, perform_semantic_filtering=False,
                         perform_pos_filtering=False, dataset=dataset, train=train)

    def __str__(self):
        return 'IGBeam1ED10'


class InformedGreedyUnfilitredAttackStrategy(InformedGreedySynonymSwapAttackStrategy):
    _syn_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', 'text_fooler_synonyms.json')
    def __init__(self, edit_distance: int = -1, **_):
        super().__init__(synonyms_path=self._syn_path, edit_distance=edit_distance, beam_size=1, backtracking=True,
                         method='random', early_stop=True, prioritize_once=False, perform_semantic_filtering=False, sim_score_window=15,
                         sim_score_threshold=0.7, perform_pos_filtering=False)

    def __str__(self):
        return 'InformedGreedyUnfilteredAttackStrategy'


class OptimisticGreedySynonymSwapAttackStrategy(GreedySynonymSwapAttackStrategy):

    def init_attack(self, orig_sentence: List[str], label: int, exploration_budget: int, replace_orig_if_needed=False):
        super().init_attack(orig_sentence, label, exploration_budget, replace_orig_if_needed)
        self._words_dict = {i: ('', np.inf) for i in self._possible_modifications.keys()}
        self._best_loss = np.inf

    def update(self, samples: List[SampleResult]):
        assert len(self._last_attacks) == len(samples)
        explored_attacks, self._attacks_planner = self._attacks_planner[:len(samples)], self._attacks_planner[len(samples):]
        for (word, syn), (_, loss, sf) in zip(explored_attacks, samples):
             if loss < self._words_dict[word][1]:
                 self._words_dict[word] = (syn, loss)

    def exploit(self, batch_size: int) -> List[List[str]]:
        words = sorted(self._words_dict.items(), key=lambda item: item[1][1])[:self._attack_specific_ed]
        attack = deepcopy(self._current_node)
        for index, (syn, _) in words:
            attack[index] = syn
        return [attack]

    def __str__(self):
        return f'Optimistic Greedy Synonym Swap, ED={self.edit_distance}, Syn={self._syn_name}'


_strategies_names = {
    'rand_synonym_swap': RandomSynonymSwapAttackStrategy,
    'rand_ed10': RandomSynonymSwapAttackStrategyED10,
    'greedy_synonym_swap': GreedySynonymSwapAttackStrategy,
    'informed_greedy_synonym_swap': InformedGreedySynonymSwapAttackStrategy,
    'informed_greedy': InformedGreedyDefaultAttackStrategy,
    'new_informed_greedy': NewInformedGreedyAttackStrategy,
    'ig_beam2_ed10': IGBeam2ED10,
    'ig_beam1_ed10': IGBeam1ED10,
    'unfiltered_informed_greedy': InformedGreedyUnfilitredAttackStrategy,
    'text_fooler': TextFoolerAttackStrategy,
    'text_fooler_like_ed10': TextFoolerLikeAttackStrategyED10,
    'wide_mcts_synonym_swap': WideSynonymSwapMCTSStrategy,
    'deep_mcts_synonym_swap': DeepSynonymSwapMCTSStrategy,
    'optimistic_greedy_synonym_swap': OptimisticGreedySynonymSwapAttackStrategy,
}


def get_strategy(strategy_name: str, *args, **kwargs) -> AttackStrategy:
    mn = strategy_name.lower()
    if mn in _strategies_names:
        return _strategies_names[mn](*args, **kwargs)
    raise ValueError(f'Unknown strategy type: {strategy_name}')


def get_strategy_names():
    return list(_strategies_names.keys())
