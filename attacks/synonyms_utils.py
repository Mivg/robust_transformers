import abc
import itertools
import os
import pickle
import re
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Union, List

import nltk
import json
from tqdm import tqdm
import numpy as np

from attacks.glue_datasets import get_dataset_dir_name
from common.utils import get_possible_perturbations, clean_sent_for_syns
import operator as op

from data.constants import stop_words

_loaded_nltk = False
_wordtags = None
def _load_nltk_once():
    global _loaded_nltk, _wordtags
    if not _loaded_nltk:
        nltk.download('brown')
        nltk.download('universal_tagset')
        _wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in nltk.corpus.brown.tagged_words(tagset='universal'))


def _get_poses(w, min_occurences=10):
    _load_nltk_once()
    # res = vb.part_of_speech(w)  # apparently this can fail because of the api request
    # return set() if not res else {v['text'] for v in json.loads(res)}
    # possible nltk universal pos tags: {'NOUN', 'CONJ', 'NUM', 'PRT', 'ADP', 'ADV', 'PRON', 'DET', 'ADJ', 'VERB'}
    # which almost matches spacy tags: https://spacy.io/api/annotation#pos-universal
    return {k for k, v in _wordtags[w].items() if v >= min_occurences}


def _create_english_only_pos_neighbors_json(neighbors_dict):

    # nlp = spacy.load("en_core_web_sm") # python3 -m spacy download en_core_web_sm
    filtered_neigbors = {}
    # def _is_english(w):
    #     return bool(vb.part_of_speech(w))
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(12) as executor:
        all_words = list(set(synonyms.keys()).union(chain.from_iterable(synonyms.values())))
        word_to_pos = {w: v for w, v in tqdm(zip(all_words, executor.map(_get_poses, all_words)), total=len(all_words)) if len(v) > 0}

    for k, v in tqdm(neighbors_dict.items()):
        if len(v) == 0:
            continue
        key_poses = word_to_pos.get(k, set())
        if len(key_poses) == 0:
            continue
        value_poses = defaultdict(list)
        for w in v:
            for pos in word_to_pos.get(w, set()):
                value_poses[pos].append(w)
        value_poses = {k: v for k, v in value_poses.items() if k in key_poses}
        if len(value_poses) > 0:
            filtered_neigbors[k] = value_poses
    return filtered_neigbors


class SynonymsCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_possible_perturbations(self, sent: Union[str, List[str]], append_self=True, raise_on_failure=True) -> List[List[str]]:
        raise NotImplementedError


class EncodedSynonyms(SynonymsCreator):
    def __init__(self, synonyms_path):
        # Note - hashes are lower cases and pre-procssed, but not cleaned. vocabulary is only words in lower case. Punctuation and
        # capitalization must come in eval time form the given sentence and is not pre-computed
        data_path = os.path.dirname((os.path.dirname(synonyms_path)))
        # vocab_path = os.path.join(data_path, 'text_fooler_synonyms_vocab.json')
        vocab_path = os.path.join(data_path, 'counterfitted_neighbors_vocab.json')
        hashes_path = synonyms_path.replace('synonyms', 'hashes')
        with open(synonyms_path) as f:
            self.synonyms = {int(k): {int(k_):v_ for k_, v_ in v.items()} for k,v in json.load(f).items()}
        with open(vocab_path) as f:
            self.rev_vocab = json.load(f)
            self.vocab = np.array(list(zip(*sorted(self.rev_vocab.items(), key=op.itemgetter(1))))[0])
        with open(hashes_path) as f:
            self.hashes = json.load(f)

    def get_possible_perturbations(self, sent: Union[str, List[str]], append_self=True, raise_on_failure=True):
        if not isinstance(sent, str):
            split_sent = sent
            sent = ' '.join(sent)
        else:
            split_sent = sent.split(' ')
        sent_id = self.hashes.get(sent.lower(), None)
        split_sent, punct, capitalized = clean_sent_for_syns(split_sent)  # remove punctuations to find synonyms
        if sent_id is None:
            # raise ValueError(f'Given sentence is not in the expected attack space encoding: {sent}')
            print(f'Given sentence does not have encoded synonyms: {sent}')  # hopefully that won't happen often
            codes = {}
        else:
            codes = self.synonyms.get(sent_id, None)
            if codes is None:
                raise ValueError(f'Given sentence does not have encoded synonyms: {sent}')
        if not append_self:
            perts = [self.vocab[codes.get(i, [])].tolist() for i in range(len(split_sent))]
        else:
            perts = [self.vocab[codes.get(i, [])].tolist() + [w] for i, w in enumerate(split_sent)]
        for k in capitalized:
            perts[k] = [w.capitalize() for w in perts[k]]
        for k, v in punct.items():
            perts[k] = [v[0] + w + v[1] for w in perts[k]]  # add the punctuations back
        return perts


class CachedSynonyms(SynonymsCreator):
    def __init__(self):
        self._sig_to_idx_dict = {}
        self._idx_to_sent_dict = {}
        self._idx_to_syns_dict = {}
        self._idx_to_cached_idx_dict = {}
        self._idx_to_sent_len = {}
        self._stripped_sig_to_idx_dict = None  # computed in runtime only if needed
        self._black_list = set()
        self._known_sents = {}  # computed in runtime only if needed

    @staticmethod
    def from_pkl(syn_pkl_path):
        with open(syn_pkl_path, 'rb') as f:
            return pickle.load(f)

    def get_possible_perturbations(self, sent: Union[str, List[str]], append_self=True, raise_on_failure=True) -> List[List[str]]:
        # the following should have been a didacted _get_sent_idx but as we already pickled the synonyms, they would need a convertor to
        # include new functions
        sig = CachedSynonyms.get_sent_signature(sent)
        try:
            idx = self._sig_to_idx_dict[sig]
        except KeyError:
            sent_ = ' '.join(sent) if isinstance(sent, list) else sent
            if not hasattr(self, '_black_list'):
                self._black_list = set()
            if not hasattr(self, '_known_sents'):
                self._known_sents = {}

            if sent_ in self._known_sents:
                idx = self._known_sents[sent_]
            elif sent_ in self._black_list:
                # no need to check it again, we already know t's not here
                if raise_on_failure:
                    raise KeyError(f'Signature for the sentence was not found at all and is blacklisted: {sent}')
                else:
                    split_sent = sent.split(' ') if isinstance(sent, str) else sent
                    syns = [[] for _ in range(len(split_sent))] if not append_self else [[w] for w in split_sent]
                    return syns
            else:
                # there is an issue with BERT tokenizer transforming "do not" into "don't" so we will give it one last go before we give up here
                new_sig = sig.replace("don't", "donot")
                if new_sig in self._sig_to_idx_dict:
                    idx = self._sig_to_idx_dict[new_sig]
                else:
                    # there is still the option of the input being too long and clipped by the tokenizer. the following will take care of it,
                    # though very wasteful in cpu time.. (we need to both look for the sig as a start of one of the existing signatures,
                    # and also take care of both "do not" and clipping issue appearing together

                    # There is also a potential problem that happen in BoolQ for example with special characters being messed up causing a
                    # failure to find the correct signature, in which case we can ignore
                    potential_indices = []
                    success = False
                    for long_sig, idx in self._sig_to_idx_dict.items():
                        if len(long_sig) > len(sig) and (long_sig.startswith(sig) or long_sig.startswith(new_sig)):
                            potential_indices.append(idx)
                    if len(potential_indices) == 1:
                        idx = potential_indices[0]
                        success = True
                    elif append_self:
                        # let's make a last attempt since some non-ascii characters may interfere with the signatures. note that it might
                        # replace the sentence in terms of punctuations and thus can only be used if append self is on (to allow for replacement)
                        if not hasattr(self, '_stripped_sig_to_idx_dict') or self._stripped_sig_to_idx_dict is None:
                            self._stripped_sig_to_idx_dict = {re.sub('[^a-z0-9]', '', sig): idx for sig, idx in self._sig_to_idx_dict.items()}
                        stripped_sig = re.sub('[^a-z0-9]', '', sig)
                        stripped_new_sig = re.sub('[^a-z0-9]', '', new_sig)
                        if stripped_sig in self._stripped_sig_to_idx_dict:
                            idx = self._stripped_sig_to_idx_dict[stripped_sig]
                            success = True
                        elif stripped_new_sig in self._stripped_sig_to_idx_dict:
                            idx = self._stripped_sig_to_idx_dict[stripped_new_sig]
                            success = True
                    if success:
                        pass  # otherwise, we failed
                    elif not raise_on_failure:
                        split_sent = sent.split(' ') if isinstance(sent, str) else sent
                        syns = [[] for _ in range(len(split_sent))] if not append_self else [[w] for w in split_sent]
                        # print(f'Warning - got a sample with no matching signature: {sent}')
                        print(f'Warning - got a sample with no matching signature..')
                        self._black_list.add(sent_)
                        return syns
                    elif len(potential_indices) == 0:
                        raise KeyError(f'Signature for the sentence was not found at all: {sent}')
                    elif len(potential_indices) > 1:
                        raise KeyError(f'Signature for the sentence was found at multiple ({len(potential_indices)}) places: {sent}')
                self._known_sents[sent_] = idx

        syns = deepcopy(self.get_possible_perturbations_by_sent_idx(idx))
        if append_self:
            for s, w in zip(syns, self._idx_to_sent_dict[idx].split()):
                s.append(w)
        return syns

    def get_possible_perturbations_by_sent_idx(self, idx: int):
        if idx not in self._idx_to_syns_dict:
            idx = self._idx_to_cached_idx_dict[idx]
        syns = self._idx_to_syns_dict[idx]
        return [syns.get(j, []) for j in range(self._idx_to_sent_len[idx])]

    def get_sent_by_sent_idx(self, idx: int):
        # Note that this is the index in the cache, and may not correspond with the current shuffle of the dataset, so it shouldn't be used directly
        if idx not in self._idx_to_syns_dict:
            idx = self._idx_to_cached_idx_dict[idx]
        return self._idx_to_sent_dict[idx]

    @staticmethod
    def get_sent_signature(sent: Union[str, List[str]]):
        if not isinstance(sent, str):
            sent = ' '.join(sent)
        return re.sub('\s', '', sent.lower())

    def is_sig_already_exists(self, sent, idx=None, sig=None):
        sig = CachedSynonyms.get_sent_signature(sent) if sig is None else sig
        if idx is not None and idx in self._idx_to_sent_dict:
            assert CachedSynonyms.get_sent_signature(self.get_sent_by_sent_idx(idx)) == sig
        if sig not in self._sig_to_idx_dict:
            return False
        if idx is not None and idx not in self._idx_to_syns_dict and idx not in self._idx_to_cached_idx_dict:
            # link this idx to the already existing one
            cached_idx = self._sig_to_idx_dict[sig]
            self._idx_to_cached_idx_dict[idx] = cached_idx

        return True

    def add_new_sent(self, sent, syns, idx, sig=None):
        sig = CachedSynonyms.get_sent_signature(sent) if sig is None else sig
        if sig in self._sig_to_idx_dict:
            raise ValueError('Adding a new sent but signature already exists')
        self._sig_to_idx_dict[sig] = idx
        self._idx_to_sent_dict[idx] = sent
        self._idx_to_syns_dict[idx] = syns
        self._idx_to_sent_len[idx] = len(sent.split(' '))

    def _run_sanity_check(self):
        for sig, idx in self._sig_to_idx_dict.items():
            assert sig == CachedSynonyms.get_sent_signature(self._idx_to_sent_dict[idx])

    def _increase_indices_by(self, addition):
        self._sig_to_idx_dict = {sig: idx+addition for sig, idx in self._sig_to_idx_dict.items()}
        self._idx_to_sent_dict = {idx+addition: sent for idx, sent in self._idx_to_sent_dict.items()}
        self._idx_to_syns_dict = {idx+addition: syns for idx, syns in self._idx_to_syns_dict.items()}
        self._idx_to_cached_idx_dict = {idx+addition: cidx+addition for idx, cidx in self._idx_to_cached_idx_dict.items()}
        self._idx_to_sent_len = {idx+addition: l for idx, l in self._idx_to_sent_len.items()}


    @staticmethod
    def merge_multiple_cached_synonyms(*cached_synonyms) -> 'CachedSynonyms':
        # there might be indices clashes here, so we would make sure it won't happen, even though we may not use some indices. If we want,
        # we can correct it later but it's not crucial.
        # changes are made in place so be warned!
        largest_index = [max(cs._sig_to_idx_dict.values())+1 for cs in cached_synonyms[:-1]]
        additions = np.cumsum(largest_index).tolist()

        orig_cached_synonyms = cached_synonyms[0]
        orig_cached_synonyms._run_sanity_check()
        for cs, addition in zip(cached_synonyms[1:], additions):
            cs._run_sanity_check()
            cs._increase_indices_by(addition)
            # copy/link the cached sentences
            for sig, idx in cs._sig_to_idx_dict.items():
                if orig_cached_synonyms.is_sig_already_exists('', idx, sig):
                    continue
                orig_cached_synonyms.add_new_sent(sent=cs._idx_to_sent_dict[idx], syns=cs._idx_to_syns_dict[idx], idx=idx, sig=sig)
            # now link the sentences which were linked before
            for idx, cached_idx in cs._idx_to_cached_idx_dict.items():
                assert orig_cached_synonyms.is_sig_already_exists(cs.get_sent_by_sent_idx(idx), idx)
        orig_cached_synonyms._run_sanity_check()
        return orig_cached_synonyms

    def __len__(self):
        return len(self._sig_to_idx_dict) + len(self._idx_to_cached_idx_dict)



class RandomEncodedSynonyms(SynonymsCreator):
    def __init__(self, vocabulary_path, n_syns=50):  # TODO XXX
        raise NotImplementedError('This probably wont work as is after the punctuations and capitalization fixes')
        with open(vocabulary_path) as f:
            rev_vocab = set(json.load(f).keys()).difference(stop_words)
        self.rev_vocab_list = np.array(list(rev_vocab))
        self.rev_vocab = {w: i for i, w in enumerate(self.rev_vocab_list)}
        self.n_syns = n_syns

    def get_possible_perturbations(self, sent: Union[str, List[str]], append_self=True, raise_on_failure=True) -> List[List[str]]:
        split_sent = sent.split(' ') if isinstance(sent, str) else sent
        rep_words = [i for i, w in enumerate(split_sent) if w in self.rev_vocab]
        res = [[w] if append_self else [] for w in split_sent]
        inds = np.random.randint(0, len(self.rev_vocab), (len(rep_words), self.n_syns + int(append_self)))
        if append_self:
            inds[:, -1] = [self.rev_vocab[split_sent[i]] for i in rep_words]
        for i, w_is in zip(rep_words, inds):
            res[i] = self.rev_vocab_list[w_is].tolist()
        return res



_encoded_synonyms = None
_random_synonyms = None


def load_synonym_for_attacks(synonyms_path, dataset=None, train=None):
    if synonyms_path is None:
        assert dataset is not None and train is not None
        syn_suffix = 'train' if train else 'dev'
        synonyms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                                     get_dataset_dir_name(dataset), f'synonyms_{syn_suffix}.pkl')
        global _encoded_synonyms
        if _encoded_synonyms is None:
            _encoded_synonyms = CachedSynonyms.from_pkl(synonyms_path)
        return _encoded_synonyms, None
    elif 'vocab' in synonyms_path:
        global _random_synonyms
        if _random_synonyms is None:
            _random_synonyms = RandomEncodedSynonyms(synonyms_path)
        return _random_synonyms, None
    with open(synonyms_path) as f:
        synonyms = defaultdict(list)
        synonyms.update(json.load(f))
    for k, v in synonyms.items():
        if len(v) > 0:
            return synonyms, isinstance(v, dict)

MOCK_SIZES = False
CAP_AT = None
attack_size_cache = {}
def set_mock_sizes(mock=True):
    global MOCK_SIZES
    MOCK_SIZES = mock

def set_cap_at(cap_at=None):
    global CAP_AT
    CAP_AT = cap_at
def _get_synonyms_attack_space_size(orig_sentence, edit_distance, synonyms, requires_pos, allow_smaller_ed, syn_name):
    global MOCK_SIZES, CAP_AT
    if MOCK_SIZES:
        return 1, 1

    def _inner():
        possible_modifications = sorted([len(ps) for i, ps in
                                         enumerate(get_possible_perturbations(orig_sentence, synonyms, requires_pos, False))
                                         if len(ps) > 0])[::-1]
        pos = len(possible_modifications)
        if not allow_smaller_ed and len(possible_modifications) < edit_distance:
            return 1, pos
        if not allow_smaller_ed:
            if CAP_AT is None:  # compute exact result which may take a very very long time for large edit distance and many attack positions
                return np.product(np.asarray(list(itertools.combinations(possible_modifications, edit_distance))), axis=1).sum() + 1, pos
            total = 0
            for c in itertools.combinations(possible_modifications, edit_distance):
                total += np.product(c)
                if total >= CAP_AT:
                    break
            return total+1, pos
        if len(possible_modifications) == 1:
            return possible_modifications[0] + 1, pos
        # TODO - prepare estimates here as it takes way too long
        if CAP_AT is None:  # compute exact result which may take a very very long time for large edit distance and many attack positions
            sz = sum(np.product(np.asarray(list(itertools.combinations(possible_modifications, eds))), axis=1).sum()
                       for eds in list(range(1, min(len(possible_modifications), edit_distance) + 1))) + 1
            # there may be an overflow
            if sz < 0:
                sz = 1e18
            return sz, pos
        total = 0
        for eds in range(1, min(len(possible_modifications), edit_distance) + 1):
            for c in itertools.combinations(possible_modifications, edit_distance):
                total += np.product(c)
                if total >= CAP_AT:
                    break
        return total+1, pos
    key = ' '.join(orig_sentence) + str(edit_distance) + syn_name + str(requires_pos) + str(allow_smaller_ed)
    if key not in attack_size_cache:
        attack_size_cache[key] = _inner()
    return attack_size_cache[key]


SYN_NAMES = {
    'counterfitted_neighbors': 'Full',
    'english_pos_counterfitted_neighbors': 'POS',
    'filtered_counterfitted_neighbors': 'Token',
}


def synonym_file_short_name(syn_path):
    if syn_path is None:
        return "PhiPrime"
    syn_name = os.path.splitext(os.path.basename(syn_path))[0]
    return SYN_NAMES.get(syn_name, syn_name)



if __name__ == '__main__':
    with open('./../data/counterfitted_neighbors.json') as f:
        synonyms = json.load(f)
    filtered_results = _create_english_only_pos_neighbors_json(synonyms)
    with open('./../data/english_pos_counterfitted_neighbors.json', 'w') as f:
        json.dump(filtered_results, f)