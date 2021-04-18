import heapq
import itertools
import json
import random
import re
from copy import deepcopy
from itertools import product as cartesian_product
from typing import List, Union

import numpy as np
import pandas as pd
import os

import torch
from netifaces import AF_INET
import netifaces as ni
from math import factorial as fact

import spacy

from hf_transformers.run_glue import _tsv_to_csv

nlp = None
def _load_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print('You need to download spacy models by using "python3 -m spacy download en_core_web_sm"')
            exit(100)

pos_cache = {}
def get_spacy_pos(sent: Union[str, List[str]], only_words=True) -> List[str]:
    """
    Assuming no multiple whitespaces!
    :param sent:
    :return:
    >>> get_spacy_pos('Hello-world how are things today? ,NYC,', only_words=False)
    ['INTJ', 'PUNCT', 'NOUN', 'ADV', 'AUX', 'NOUN', 'NOUN', 'PUNCT', 'PUNCT', 'PROPN', 'PUNCT']
    >>> get_spacy_pos('Hello-world how are things today?', only_words=True)
    ['INTJ', 'ADV', 'AUX', 'NOUN', 'NOUN']
    >>> get_spacy_pos("How about multiple. sentences that? are joined by punctuations")
    ['ADV', 'ADP', 'ADJ', 'NOUN', 'DET', 'AUX', 'VERB', 'ADP', 'NOUN']
    """
    global pos_cache
    _load_nlp()
    if not isinstance(sent, str):
        split_sent = sent
        sent = ' '. join(sent)
    else:
        split_sent = sent.split(' ')
    if not only_words:
        return [t.pos_ for t in nlp(sent)]
    if sent not in pos_cache:
        doc = nlp(sent)
        if len(doc) != len(split_sent):
            posses = []
            curr_word_ind = 0
            for t in doc:
                if split_sent[curr_word_ind].startswith(t.text):
                    posses.append(t.pos_)
                    curr_word_ind += 1
                    if curr_word_ind == len(split_sent):
                        break
            if curr_word_ind != len(split_sent):
                print(f'Spacy did not find pos tags for all words in the given sentence! failed to find: {repr(split_sent[curr_word_ind])}')
                raise RuntimeError('Spacy did not find pos tags for all words in the given sentence!')
        else:
            posses = [t.pos_ for t in doc]
        pos_cache[sent] = posses
    return pos_cache[sent]


gpt2_model, gpt2_tokenizer = None, None

def _load_gpt2(cache_dir, device):
    global gpt2_model, gpt2_tokenizer
    if gpt2_model is None:
        try:

            model_id = 'gpt2-large'
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            gpt2_model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
            gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_id, cache_dir=cache_dir)
        except OSError:
            print('Failed to load GPT2 model')
            exit(100)

ppl_cache = {}
def get_perplexity(sent, cache_dir, device):
    global gpt2_model, gpt2_tokenizer, ppl_cache
    if not isinstance(sent, str):
        sent = ' '.join(sent)
    if sent not in ppl_cache:
        _load_gpt2(cache_dir, device)
        # based on https://huggingface.co/transformers/perplexity.html
        max_length = gpt2_model.config.n_positions
        stride = 512
        encodings = gpt2_tokenizer(sent, return_tensors='pt')

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)

        ppl_cache[sent] = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl_cache[sent]


def create_perturbed_sentences_with_edit_distance(split_sentence, possible_perturbations, replacable_words_inds, n_pertrubs,
                                                  edit_distance, max_perturbations):
    # TODO - shoould I assume the possible perturbations always include the original word?
    if edit_distance > len(split_sentence):
        return []
    valid_pertrubs = n_pertrubs[n_pertrubs>1]

    def shuold_choose_from_exhaustive(total_pert):
        return total_pert <= 5 * max_perturbations or total_pert <= 50

    if len(valid_pertrubs) < edit_distance:
        return []
    elif len(valid_pertrubs) > edit_distance:
        # compute lower bound efficiently
        try:
            total_perturbations = ((valid_pertrubs.min()-1) ** edit_distance) * \
                                  (fact(len(valid_pertrubs)) / fact(len(valid_pertrubs)-edit_distance) / fact(edit_distance))
        except OverflowError:
            total_perturbations = int(1e14)
        if shuold_choose_from_exhaustive(total_perturbations):
            # if the current estimation lends us in a place where we are about to compute the entire attack space, need to make sure it
            # is not too big by computing the real value
            total_perturbations = \
                sum(np.product(pert) for pert in itertools.combinations(n_pertrubs[replacable_words_inds] - 1, edit_distance))
    else:  # elif len(valid_pertrubs) == edit_distance:
        total_perturbations = np.product(valid_pertrubs-1)  # don't take into account the original word

    if shuold_choose_from_exhaustive(total_perturbations):
        all_pertrubed_sentences = []
        # this makes sure we won't consider the original word as a possible perturbation
        perts = np.array([perts[:-1] for perts in possible_perturbations])
        for words_inds_to_replace in itertools.combinations(replacable_words_inds, edit_distance):
            new_sent_format = ' '.join(split_sentence[i] if i not in words_inds_to_replace else '{}' for i in range(len(split_sentence)))
            for rep_words in cartesian_product(*perts[list(words_inds_to_replace)]):
                all_pertrubed_sentences.append(new_sent_format.format(*rep_words))
                if len(all_pertrubed_sentences) > total_perturbations:
                    raise RuntimeError(f'Computation error! got already {total_perturbations+1} though expected only '
                                       f'{total_perturbations} to exist.'
                                       f'\nSent: {split_sentence}'
                                       f'\nPert: {possible_perturbations}'
                                       f'\nED: {edit_distance},\tMaxPert: {max_perturbations}'
                                       f'\nRepWordInd: {replacable_words_inds},'
                                       f'\nNPertrubs: {n_pertrubs}')
        if total_perturbations <= max_perturbations:
            return all_pertrubed_sentences
        else:
            return np.random.choice(all_pertrubed_sentences, replace=False, size=max_perturbations)
    else:
        all_pertrubed_sentences = set()
        total_attempts = 0
        while len(all_pertrubed_sentences) < max_perturbations:
            if total_attempts / max_perturbations >= 2:
                msg = f'Been trying to get {max_perturbations} perturbations where presumably had {total_perturbations} possibilities ' \
                    f'but already gave it {total_attempts} shots and not done'
                print(msg)
            if total_attempts / max_perturbations >= 5:
                msg += f'\nThe sentence: ' + ' '.join(split_sentence) + f'\nPerturbations: {possible_perturbations}'
                raise RuntimeError(msg)

            words_inds_to_replace = np.random.choice(replacable_words_inds, replace=False, size=edit_distance)
            new_sent = deepcopy(split_sentence)
            for i in words_inds_to_replace:
                # we force these words to be replaced to have the edit distance required
                try:
                    new_sent[i] = random.choice(possible_perturbations[i][:-1])  # faster than np.random for list of strings
                except IndexError as e:
                    msg = f'Issue in random choice at index {i}. Sentence: ' + (' '.join(split_sentence)) + (f'\nPertrubations: {possible_perturbations}')
                    raise IndexError(msg) from e
            all_pertrubed_sentences.add(' '.join(new_sent))
            total_attempts += 1
        return list(all_pertrubed_sentences)


spacy_to_nltk_pos_map = {
    'NOUN': 'NOUN', 'CONJ': 'CONJ', 'NUM': 'NUM', 'PART': 'PRT', 'ADP': 'ADP', 'ADV': 'ADV', 'PRON': 'PRON', 'PROPN': 'PRON', 'DET': 'DET',
    'ADJ': 'ADJ', 'VERB': 'VERB', 'CCONJ': 'CONJ'
}


# unclean_word_pattern = re.compile('^(.+)[\.,\?!]]$')
unclean_word_pattern = re.compile('^([^\w]*)(\w+)([^\w]*)$')
has_non_letters = re.compile('^.*[^\w]+.*')

def clean_sent_for_syns(split_sent):
    """
    Note that it will only clean words if they start/end with a string of non-letter character.
    It will also not separate word-word pairs
    :param split_sent:
    :return:
    >>> clean_sent_for_syns(['Hello', 'World.', 'How,', 'are!', 'you?', '?', '!', ',', '.', 'today.of', 'all,days?'])
    (['hello', 'world', 'how', 'are', 'you', '?', '!', ',', '.', 'today.of', 'all,days?'], {1: ('', '.'), 2: ('', ','), 3: ('', '!'), 4: ('', '?')}, {0, 1, 2})
    >>> clean_sent_for_syns(['.this', 'also:', '"gets', '"cleaned"', '"but,,,', 'this', 'is', 'not', 'because.it.has.inner.things!!!'])
    (['this', 'also', 'gets', 'cleaned', 'but', 'this', 'is', 'not', 'because.it.has.inner.things!!!'], {0: ('.', ''), 1: ('', ':'), 2: ('"', ''), 3: ('"', '"'), 4: ('"', ',,,')}, set())
    >>> clean_sent_for_syns(['...Hello!.'])
    (['hello'], {0: ('...', '!.')}, {0})
    """
    punct = {}
    capitalized = set()
    clean_sent = []
    for i, w in enumerate(split_sent):
        if w.istitle():
            capitalized.add(i)
            w = w.lower()
        m = unclean_word_pattern.match(w)
        if m is None or has_non_letters.match(w) is None:
            clean_sent.append(w)
            continue
        else:
            clean_sent.append(m.group(2))
            punct[i] = (m.group(1), m.group(3))
    return clean_sent, punct, capitalized


# TODO - remove punctuations (there are sentences with comma in the middle which confuses the attacker)
nlp_cache = {}
def get_possible_perturbations(split_sentence, neighbors_dict, requires_pos, append_self=True, raise_on_failure=True):
    if requires_pos is None:
        # assuming that neighbors dict is actually an instance of SynonymsCreator. This needs the original sentence as this is what is hashed
        perts = neighbors_dict.get_possible_perturbations(split_sentence, append_self, raise_on_failure=raise_on_failure)
    else:
        split_sentence, punct, capitalized = clean_sent_for_syns(split_sentence)  # remove punctuations to find synonyms
        if not requires_pos:
            perts = [neighbors_dict.get(w, []) + ([w] if append_self else []) for w in split_sentence]
        else:
            _load_nlp()
            doc = " ".join(split_sentence)
            if doc not in nlp_cache:
                nlp_cache[doc] = nlp(doc)
            doc = nlp_cache[doc]
            perts = [neighbors_dict.get(w, {}).get(spacy_to_nltk_pos_map.get(token.pos_, 'UNK'), []) + ([w] if append_self else [])
                    for w, token in zip(split_sentence, doc)]
        for k in capitalized:
            perts[k] = [w.capitalize() for w in perts[k]]
        for k, v in punct.items():
            perts[k] = [v[0] + w + v[1] for w in perts[k]]  # add the punctuations back
    return perts


def pertrub_sentence(sentence, sentence_index, max_perturbations, neighbors_dict, edit_distance, is_edit_dist_pert, is_dump_all,
                     add_orig=True, requires_pos=False, filter_func=None, replace_orig_if_needed=False):
    assert edit_distance != 0
    dist_to_sentences_dict = {}
    total_perturbations = 0

    # make it fool proof..
    split_sentence = sentence.split(' ') if isinstance(sentence, str) else sentence
    sentence = sentence if isinstance(sentence, str) else ' '.join(sentence)

    possible_perturbations = get_possible_perturbations(split_sentence, neighbors_dict, requires_pos, append_self=True,
                                                        raise_on_failure=not replace_orig_if_needed)  # append_self is True!
    if replace_orig_if_needed:
        split_sentence = [p[-1] for p in possible_perturbations]
        sentence = ' '.join(split_sentence)
    elif len(possible_perturbations) != len(split_sentence):
        # This happens 8 times in sst2 train data and occur in offline augmentation if replace_orig_if_needed is False
        print(f'Warning Found a matching signature but the perturbations do not seem to match the length of the sentence so '
              f'returning no perturbations: {sentence}')
        possible_perturbations = [[w] for w in split_sentence]

    # considering every edit_distance - upper bound of relevant attack space
    n_pertrubs = np.array([float(len(p)) for p in possible_perturbations])
    replacable_words_inds = np.where(n_pertrubs > 1)[0]
    if len(replacable_words_inds) > 0 and replacable_words_inds[-1] >= len(split_sentence):
        raise ValueError('It seems that the synonyms found cannot match the sentence given. '
                         'Consider using replace_orig_if_needed (but do it carefully and make sure that the difference are indeed minor). '
                         f'The problematic sentence was: {sentence}')
    if len(replacable_words_inds) > 0 and possible_perturbations[replacable_words_inds[0]][-2] == possible_perturbations[replacable_words_inds[0]][-1]:
        # this is important, but disabling for now. TODO - restore this assertion and make sure it never happens
        #     raise ValueError(f'There is an issue of double addition of perturbation. split_sentence: {sentence}, perturbations: {possible_perturbations}')
        print(f'Warning - There is an issue of double addition of perturbation. split_sentence: {sentence}, perturbations: {possible_perturbations}')
        possible_perturbations = [list(set(syns)) for syns in possible_perturbations]

    max_perturbs = np.product(n_pertrubs)

    if max_perturbs < max_perturbations and not is_dump_all:
        print(f'For sentence number {sentence_index} cannot create a full perturbation set since it has only '
              f'{max_perturbs} possible perturbations')

    if max_perturbs <= 1:
        # no perturbations for this sample
        pass
    elif filter_func is not None:
        if is_edit_dist_pert:
            raise NotImplementedError('This option is not implemented')
        new_sents = {sentence}
        total_attempts = 0
        while len(new_sents) < max_perturbations + 1:
            if total_attempts > 5 * max_perturbations:
                break
                # raise RuntimeError(f'Been trying to get {max_perturbations + 1} perturbations where presumably had '
                #                    f'{max_perturbs} possibilities but already gave it {total_attempts} shots and not done')
            total_attempts += 1
            changes = 0
            sent = split_sentence
            for wi in np.random.permutation(replacable_words_inds):
                sents = []
                for w in np.random.permutation(possible_perturbations[wi]):
                    sents.append(deepcopy(sent))
                    sents[-1][wi] = w
                mask = filter_func(sents, int(wi))
                if np.any(mask):
                    sent = sents[np.argmax(mask)]
                    changes += 1
                if changes == edit_distance:
                    break
            if changes == edit_distance or edit_distance == -1:
                new_sents.add(' '.join(sent))

        new_sents.remove(sentence)
        dist_to_sentences_dict[edit_distance] = list(new_sents)
        total_perturbations = len(dist_to_sentences_dict[edit_distance])
    elif not is_edit_dist_pert:  # only a single edit-distance perturbation
        if len(replacable_words_inds) < edit_distance:
            pass  # no possible perturbations with high enough edit distance

        elif edit_distance < 0:  # no limit on the edit distance
            if max_perturbs <= 5 * max_perturbations or max_perturbs < 200:  # In this case, just sampling a random permutation may take way longer since we will
                # sample existing sentences with high probability
                all_pertrubs = set([' '.join(words) for words in cartesian_product(*possible_perturbations)])
                all_pertrubs.remove(sentence)  # this one we want for sure
                all_pertrubs = list(all_pertrubs)
                if len(all_pertrubs) <= max_perturbations:
                    dist_to_sentences_dict[-1] = all_pertrubs
                else:
                    dist_to_sentences_dict[-1] = np.random.choice(all_pertrubs, size=max_perturbations, replace=False).tolist()
            else:
                new_sents = {sentence}
                total_attempts = 0
                while len(new_sents) < max_perturbations + 1:
                    if total_attempts > 5*max_perturbations:
                        print(possible_perturbations)
                        raise RuntimeError( f'Been trying to get {max_perturbations+1} perturbations where presumably had '
                                            f'{max_perturbs} possibilities but already gave it {total_attempts} shots and not done')
                    total_attempts += 1
                    new_sents.add(' '.join(np.random.choice(pp) for pp in possible_perturbations))
                new_sents.remove(sentence)
                dist_to_sentences_dict[-1] = list(new_sents)
            total_perturbations = len(dist_to_sentences_dict[-1])
        else:
            dist_to_sentences_dict[edit_distance] = create_perturbed_sentences_with_edit_distance(split_sentence, possible_perturbations,
                                                                                                  replacable_words_inds, n_pertrubs,
                                                                                                  edit_distance, max_perturbations)
            total_perturbations = len(dist_to_sentences_dict[edit_distance])
    else:
        for ed in range(1, edit_distance + 1):
            dist_to_sentences_dict[ed] = create_perturbed_sentences_with_edit_distance(split_sentence, possible_perturbations,
                                                                                       replacable_words_inds, n_pertrubs, ed,
                                                                                       max_perturbations)
            total_perturbations += len(dist_to_sentences_dict[ed])

    if add_orig:
        dist_to_sentences_dict[0] = [sentence] if isinstance(sentence, str) else sentence
        total_perturbations += 1

    return dist_to_sentences_dict, total_perturbations >= max_perturbations, total_perturbations


def read_glue_data(glue_data_dir, task_name, dataset):
    if task_name.lower() == 'boolq':
        file_path = os.path.join(glue_data_dir, task_name, dataset + '.jsonl')
        with open(file_path, encoding="utf-8") as lines:
            data = pd.DataFrame([json.loads(line) for line in lines])
        data['label'] = data['label'].apply(int)
    else:
        file_path = os.path.join(glue_data_dir, task_name, dataset + '.csv')
        if not os.path.exists(file_path):
            file_path = os.path.join(glue_data_dir, task_name, dataset + '.tsv')
            file_path = _tsv_to_csv(file_path, overwrite=False)[0]
        data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    return data


class Heap(object):
    """
    From https://stackoverflow.com/a/8875823
    """
    def __init__(self, initial=None, key=lambda x: x, max_heap: bool = False):
        self.sign = -1 if max_heap else 1
        self.key = lambda item: self.sign * key(item)
        self.index = 0
        if initial:
            self._data = [(self.key(item), i * self.sign, item) for i, item in enumerate(initial)]
            self.index = len(self._data) * self.sign
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += self.sign

    def pop(self):
        return heapq.heappop(self._data)[2]

    def nlargest(self, n):
        # really inefficient..
        res = [self.pop() for _ in range(min(len(self._data), n))]
        for r in res[::-1]:
            self.push(r)
        return res

    def clear(self):
        while len(self) > 0:
            self.pop()

    def __len__(self):
        return len(self._data)



# Function to display hostname and
# IP address
ip_regex = re.compile('(?:\d{1,3}\.){3}\d{1,3}')
def get_Host_name_IP():
    # this is the usual way to get it done but it returns local host which is not good enough
    # try:
    #     host_name = socket.gethostname()
    #     host_ip = socket.gethostbyname(host_name)
    #     print("Hostname :  ", host_name)
    #     print("IP : ", host_ip)
    #     return host_ip
    # except:
    #     print("Unable to get Hostname and IP")

    # https://stackoverflow.com/a/6250688
    ifs = ni.interfaces()
    for i in ifs[::-1]:
        if 'vpn' in i or 'docker' in i:
            continue
        add = ni.ifaddresses(i)
        if AF_INET not in add:
            continue
        add = add[AF_INET][0]
        if 'addr' not in add:
            continue
        ip = add['addr']
        if ip_regex.match(ip) is None:
            continue
        return ip
    raise RuntimeError('failed to find local ip address')


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    from transformers import is_torch_available, is_tf_available
    if is_torch_available():
        torch.manual_seed(seed)
        # if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        if is_tf_available():
            import tensorflow as tf
            tf.random.set_seed(seed)
            # ^^ safe to call this function even if cuda is not available


if __name__ == '__main__':
    sent="TANDEM is an odd slice in the Japanese pink genre-as it has the requisite sex-scenes and misogynistic tone that is all but required for these types of films-but also throws in a disjointed drama/dark-comedy storyline that seems like it'd have been better suited for a different type of film. <br /><br />The film starts with two lone guys at a restaurant-each daydreaming about a previous sexual encounter. One is a mutual subway groping, the other a pretty typical (for this type of film) semi-rape scenario. The two pervs meet and start talking after one lends the other a cigarette. They hang out for an evening and talk a bit about their respective sex- lives. The film is inter-cut with flashback scenes of both of the men's interactions with the women that are central in their lives. The two men have a falling out and the film ends on a weird but predictable note... <br /><br />I really don't know what to make of TANDEM. It sorta comes off as a soft-core, 'odd couple' type of anti-buddy-film, but doesn't really explore the subject-matter to any satisfying degree. There's also not much of the typical extreme sleaziness often so prevalent in these types of films-so I can't really figure out what the point was. I also cant quite tell if the film was supposed to be funny, depressing, or both.  I think that TANDEM could have had some potential as a more serious drama film with a dark-comedy edge- but as a soft-core sex film that tries to be too 'smart' for its own good-it just doesn't work.   Can't say I hated this one-but can't say there's anything notable about it either. 4/10"
    get_spacy_pos(sent)


    sent = ['onto', 'the', 'end', 'credits'] # 'flat as the scruffy sands of its titular communities'.split()
    perts = [['for', 'in', 'during', 'on', 'into', 'towards', 'toward', 'at', 'onto'],
             ['the'], ['terminates', 'ends', 'terminating', 'ending', 'terminate', 'ceases', 'termination', 'conclude', 'end'],
             ['credit', 'credence', 'appropriation', 'loans', 'credits']]
    # [['apartment', 'flat'], ['as'], ['the'], ['unkempt', 'disheveled', 'scruffy'], ['sable', 'sands'], ['of'], ['its'], ['titular'],
    # ['community', 'communities']]
    rep_word_inds = [0, 2, 3]  # [0, 3, 4, 8]
    valid_pertrubs = np.array([9,1,9,5])  #np.array([2,1,1,3,2,1,1,1,2])
    max_pertrubs = 3  # 10


    for ed in range(5, 1, -1):
        create_perturbed_sentences_with_edit_distance(sent, perts, rep_word_inds, valid_pertrubs, ed, max_pertrubs)