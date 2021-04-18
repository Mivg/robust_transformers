# based on code from https://github.com/jind11/TextFooler
from typing import List, Union, Optional

import nltk
# requires download of nltk tagger -
from attacks.sem_sim_model import get_semantic_sim_predictor
from common.utils import get_spacy_pos, get_perplexity
import numpy as np

nltk.download('averaged_perceptron_tagger')


def _get_start_end(n_words, word_index, half_window_size, return_idx=False):
    """

    :param n_words:
    :param word_index:
    :param half_window_size:
    :return: start, end such that end-start == min(n_words, 2*half_window_size + 1)
    >>> _get_start_end(20, 0, 3)
    (0, 7)
    >>> _get_start_end(20, 2, 3)
    (0, 7)
    >>> _get_start_end(20, 8, 3)
    (5, 12)
    >>> _get_start_end(20, 14, 3)
    (11, 18)
    >>> _get_start_end(20, 18, 3)
    (13, 20)
    >>> _get_start_end(20, 19, 3)
    (13, 20)
    >>> _get_start_end(9, 1, 3)
    (0, 7)
    >>> _get_start_end(9, 1, 7)
    (0, 9)
    >>> _get_start_end(9, 5, 7)
    (0, 9)
    >>> _get_start_end(20, 0, 3, True)
    (0, 7, 0)
    >>> _get_start_end(20, 2, 3, True)
    (0, 7, 2)
    >>> _get_start_end(20, 8, 3, True)
    (5, 12, 3)
    >>> _get_start_end(20, 14, 3, True)
    (11, 18, 3)
    >>> _get_start_end(20, 18, 3, True)
    (13, 20, 5)
    >>> _get_start_end(20, 19, 3, True)
    (13, 20, 6)
    >>> _get_start_end(9, 1, 3, True)
    (0, 7, 1)
    >>> _get_start_end(9, 1, 7, True)
    (0, 9, 1)
    >>> _get_start_end(9, 5, 7, True)
    (0, 9, 5)
    """
    start = max(word_index - half_window_size, 0)
    end = start + 2*half_window_size+1
    if end > n_words:
        start -= (end-n_words)
        start = max(0, start)
        end = n_words
    if not return_idx:
        return start, end
    return start, end, word_index-start


def get_use_model_attacks_filter(orig_sent: List[str], tfhub_cache_path: str, sim_score_window: int = 15, sim_score_threshold: float = 0.7,
                                 return_mask: bool = True):
    assert sim_score_window % 2 == 1
    n_words = len(orig_sent)
    if sim_score_window > n_words:
        # no filtering will be done
        if return_mask:
            return lambda attacks, _: [True] * len(attacks)
        return lambda attacks, _: attacks

    sim_predictor = get_semantic_sim_predictor(tfhub_cache_path)
    half_sim_score_window = int((sim_score_window - 1) // 2)

    def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
        if isinstance(attacked_word_idx, int):
            start, end = _get_start_end(n_words, attacked_word_idx, half_sim_score_window)
            sents1 = [' '.join(orig_sent[start:end])]
            sents2 = [' '.join(sent[start:end]) for sent in attacks]
        else:
            assert len(attacks) == len(attacked_word_idx)
            sents1, sents2 = [], []
            for sent, idx in zip(attacks, attacked_word_idx):
                start, end = _get_start_end(n_words, idx, half_sim_score_window)
                sents1.append(' '.join(sent[start:end]))
                sents2.append(' '.join(orig_sent[start:end]))
        if return_mask:
            return sim_predictor(sents1, sents2) >= sim_score_threshold
        return [a for a, is_valid in zip(attacks, sim_predictor(sents1, sents2) >= sim_score_threshold) if is_valid]
    return filter_attacks


UniversalPos = ['NOUN', 'VERB', 'ADJ', 'ADV',
                'PRON', 'DET', 'ADP', 'NUM',
                'CONJ', 'PRT', '.', 'X']


# Function 1:
def get_pos(sent: List[str], tagset='universal'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    >>> get_pos('Are you going to the gym later today?'.split())
    ('NOUN', 'PRON', 'VERB', 'PRT', 'DET', 'NOUN', 'ADV', 'NOUN')
    '''
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        try:
            word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
        except IndexError as e:
            print(f'Error in nltk pos tagging on send="{sent}": {e}')
            raise
    _, pos_list = zip(*word_n_pos_list)
    return pos_list


ALLOWED_POS_SEY = {'NOUN', 'VERB'}
# Function 2: Pos Filter
def pos_word_filter(ori_pos: str, new_pos_list: List[str]):
    """

    :param ori_pos: the original pos tag of the word
    :param new_pos_list: the new pos tags of the word in each attack
    :return:
    >>> ori = get_pos('Are you going to the gym later today?'.split())
    >>> new = get_pos('Am I going to the gym after tomorrow?'.split())
    >>> pos_word_filter(ori[1], [new[1], new[2], ori[1]])
    [True, False, True]
    """

    return [ori_pos == new_pos or {ori_pos, new_pos} <= ALLOWED_POS_SEY for new_pos in new_pos_list]

"""
The way they used it (idx is the index of the last perturbed word):
            pos_ls = criteria.get_pos(original_text)
            synonyms_pos_ls = [criteria.get_pos(attack[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(attack) > 10 else criteria.get_pos(attack)[idx] for attack in attacks]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
"""


def _get_pos_of_window(text: List[str], idx: int) -> str:
    return get_pos(text[max(idx - 4, 0):idx + 5])[min(4, idx)]


def get_tf_pos_based_attacks_filter(orig_sent: List[str], return_mask: bool = True):
    orig_pos_tags = get_pos(orig_sent)

    def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
        if isinstance(attacked_word_idx, int):
            orig_pos_tag = orig_pos_tags[attacked_word_idx]
            idx = attacked_word_idx
            new_pos_tags = [_get_pos_of_window(attack, idx) if len(attack) > 10 else get_pos(attack)[idx] for attack in attacks]
            mask = pos_word_filter(orig_pos_tag, new_pos_tags)
        else:
            assert len(attacked_word_idx) == len(attacks)
            mask = [pos_word_filter(orig_pos_tags[idx], [_get_pos_of_window(attack, idx)])[0]
                    for attack, idx in zip(attacks, attacked_word_idx)]
        if return_mask:
            return mask
        return [attack for valid, attack in zip(mask, attacks) if valid]
    return filter_attacks


def get_spacy_pos_based_attacks_filter(orig_sent: List[str], window_size: int, return_mask: bool = True, compute_orig_once: bool = True):

    assert window_size % 2 == 1

    n_words = len(orig_sent)
    if n_words <= window_size:
        orig_pos_tags = get_spacy_pos(orig_sent)

        def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
            if isinstance(attacked_word_idx, int):
                orig_pos_tag = orig_pos_tags[attacked_word_idx]
                idx = attacked_word_idx
                new_pos_tags = [get_spacy_pos(attack)[idx] for attack in attacks]
                mask = [npt == orig_pos_tag for npt in new_pos_tags]
            else:
                assert len(attacked_word_idx) == len(attacks)
                mask = [orig_pos_tags[idx] == get_spacy_pos(attack)[idx] for attack, idx in zip(attacks, attacked_word_idx)]
            if return_mask:
                return mask
            return [attack for valid, attack in zip(mask, attacks) if valid]

        return filter_attacks

    half_score_window = int((window_size - 1) // 2)

    orig_pos_tags = get_spacy_pos(orig_sent) if compute_orig_once else None

    def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
        if isinstance(attacked_word_idx, int):
            start, end, new_attacked_word_idx = _get_start_end(n_words, attacked_word_idx, half_score_window, True)
            orig_pos_tag = get_spacy_pos(orig_sent[start:end])[new_attacked_word_idx] if not compute_orig_once else orig_pos_tags[attacked_word_idx]
            new_pos_tags = [get_spacy_pos(attack[start:end])[new_attacked_word_idx] for attack in attacks]
            mask = [npt == orig_pos_tag for npt in new_pos_tags]
        else:
            assert len(attacked_word_idx) == len(attacks)
            mask = []
            for attack, idx in zip(attacks, attacked_word_idx):
                start, end, new_idx = _get_start_end(n_words, idx, half_score_window, True)
                orig_pos_tag = get_spacy_pos(orig_sent[start:end])[new_idx] if not compute_orig_once else orig_pos_tags[idx]
                mask.append(orig_pos_tag == get_spacy_pos(attack[start:end])[new_idx])
        if return_mask:
            return mask
        return [attack for valid, attack in zip(mask, attacks) if valid]
    return filter_attacks


def get_gpt2_ppl_based_attacks_filter(orig_sent: List[str], window_size: int, ratio_threshold: float, cache_dir: str = None,
                                      return_mask: bool = True, device: str = 'cuda:0'):
    assert window_size % 2 == 1

    n_words = len(orig_sent)
    if n_words <= window_size:
        orig_ppl = get_perplexity(orig_sent, cache_dir, device)

        def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
            new_ppls = [get_perplexity(attack, cache_dir, device) for attack in attacks]
            mask = [orig_ppl / nppl >= ratio_threshold for nppl in new_ppls]
            if return_mask:
                return mask
            return [attack for valid, attack in zip(mask, attacks) if valid]

        return filter_attacks

    half_score_window = int((window_size - 1) // 2)

    def filter_attacks(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
        if isinstance(attacked_word_idx, int):
            start, end = _get_start_end(n_words, attacked_word_idx, half_score_window)
            orig_ppl = get_perplexity(orig_sent[start:end], cache_dir, device)
            new_ppls = [get_perplexity(attacks[start:end], cache_dir, device)for attack in attacks]
            mask = [orig_ppl / nppl >= ratio_threshold for nppl in new_ppls]
        else:
            assert len(attacked_word_idx) == len(attacks)
            mask = []
            for attack, idx in zip(attacks, attacked_word_idx):
                start, end = _get_start_end(n_words, idx, half_score_window)
                mask.append(get_perplexity(orig_sent[start:end], cache_dir, device) /
                            get_perplexity(attack[start:end], cache_dir, device) >= ratio_threshold)
        if return_mask:
            return mask
        return [attack for valid, attack in zip(mask, attacks) if valid]

    return filter_attacks



def get_filter(filter_by_tf_pos: bool, filter_by_sem: bool, orig_sent: List[str], return_mask: bool = True,
               tfhub_cache_path: Optional[str] = None, sim_score_window: int = 15, sim_score_threshold: float = 0.7,
               filter_spacy_pos: bool = False, spacy_pos_window: int = 100, filter_by_gpt2_ppl: bool = False,
               ppl_score_window: int = 100, ppl_score_ratio_threshold: float = 0.9, device: str = 'cuda:0'):
    assert not (filter_by_tf_pos and filter_by_gpt2_ppl)

    filters = []
    if filter_by_sem:
        filters.append(get_use_model_attacks_filter(orig_sent, tfhub_cache_path, sim_score_window, sim_score_threshold, True))
    if filter_spacy_pos:
        filters.append(get_spacy_pos_based_attacks_filter(orig_sent, window_size=spacy_pos_window, return_mask=True))
    if filter_by_tf_pos:
        filters.append(get_tf_pos_based_attacks_filter(orig_sent, return_mask=True))
    if filter_by_gpt2_ppl:
        filters.append(get_gpt2_ppl_based_attacks_filter(orig_sent, window_size=ppl_score_window, ratio_threshold=ppl_score_ratio_threshold,
                                                   cache_dir=tfhub_cache_path, return_mask=True, device=device))

    def _filter(attacks: List[List[str]], attacked_word_idx: Union[int, List[int]]) -> List[List[str]]:
        if len(attacks) == 0:
            return []
        mask = np.ones(len(attacks), dtype=bool)
        for filt in filters:
            inds = np.where(mask)[0]
            new_attacks = [attacks[i] for i in inds]
            new_indices = [attacked_word_idx[i] for i in inds] if not isinstance(attacked_word_idx, int) else attacked_word_idx
            new_mask = filt(new_attacks, new_indices)
            mask[inds] = new_mask
            if not any(mask):
                return [] if not return_mask else mask.tolist()
        if return_mask:
            return mask.tolist()
        return [attacks[i] for i in np.where(mask)[0]]
    return _filter


