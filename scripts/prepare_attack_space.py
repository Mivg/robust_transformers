import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from tqdm import trange, tqdm
import numpy as np
from multiprocessing import get_context


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.synonyms_utils import CachedSynonyms
from common.utils import clean_sent_for_syns, get_spacy_pos
from attacks.glue_datasets import get_dataset
from attacks.text_fooler_filters import get_filter


NEIGHBORS_FILE = '../data/text_fooler_synonyms.json'
GLUE_DATA_DIR = os.environ.get('GLUE_DIR', '/media/disk2/maori/data')


def analyze_dataset_chunk(inputs):
    # init analysis configurations
    start, end, args, device = inputs
    dataset = _load_dataset(args)
    with open(args.neighbors_path) as f:
        neighbors = json.load(f)

    cached_synonyms = CachedSynonyms()
    is_two_inputs = dataset.is_two_inputs_dataset()

    if end is not None:
        updater = start == args.first_sample
        temp_name = args.out_path + f'_chunk_{start}_{end}.pkl'
        if os.path.isfile(temp_name):
            # continue from checkpoint
            cached_synonyms = CachedSynonyms.from_pkl(temp_name)
            print(f'Loading saved checkpoint to compute chunk {start + len(cached_synonyms)}-{end} instead of starting at {start}')
            start = start + len(cached_synonyms)
        range_gen = trange(start, end) if updater else range(start, end)
    else:
        updater = start[0] == -1
        if updater:
            start = start[1:]
        range_gen = tqdm(start) if updater else start  # start is an iterable with specific indices to re-compute
        temp_name = args.out_path + f'_fix_chunk_{start[0]}_{start[-1]}.pkl'
        if os.path.isfile(temp_name):
            # continue from checkpoint
            cached_synonyms = CachedSynonyms.from_pkl(temp_name)
            print(f'Loading saved checkpoint to compute fixed chunk of {len(start)-len(cached_synonyms)} instead of {len(start)}')
            start = start[len(cached_synonyms):]

    # init counters
    total_words_capitalized = total_cands_due_to_captialize = total_words_punctuation = total_cands_due_to_punctuation = 0
    total_cands = total_words = total_syns = total_syns_due_to_captialize = total_syns_due_to_punctuation = 0
    total_blacklisted_words = total_cands_lost = 0

    # build synonym cache for each sentence
    checkpoint_counter = args.checkpoint_every
    for i in range_gen:
        if checkpoint_counter == 0 and temp_name is not None:
            checkpoint_counter = args.checkpoint_every
            with open(temp_name, 'wb') as f:
                pickle.dump(cached_synonyms, f)

        checkpoint_counter -= 1
        sents = []
        inds = []
        sentence = dataset.get_unsplitted_sentence_to_perturb(i)

        curr_size = len(cached_synonyms)
        if cached_synonyms.is_sig_already_exists(sentence, i):
            assert len(cached_synonyms) == curr_size+1
            continue

        split_sentence = sentence.split(' ')
        perts_filter = get_filter(filter_by_tf_pos=args.perform_tf_pos_filtering, filter_by_sem=args.perform_use_semantic_filtering,
                                  orig_sent=split_sentence, return_mask=True, tfhub_cache_path=args.tfhub_cache_path,
                                  sim_score_window=args.sim_score_window, sim_score_threshold=args.sim_score_threshold,
                                  filter_spacy_pos=args.perform_spacy_pos_filtering, spacy_pos_window=args.spacy_pos_score_window,
                                  filter_by_gpt2_ppl=args.perform_gp2_ppl_filtering, ppl_score_window=args.ppl_score_window,
                                  ppl_score_ratio_threshold=args.ppl_score_ratio_threshold, device=f'cuda:{device}')

        orig_split_sent = deepcopy(split_sentence)
        clean_split_sentence, punct, capitalized = clean_sent_for_syns(split_sentence)
        total_words_punctuation += len(punct)
        total_words_capitalized += len(capitalized)
        # Preparing candidates thus need to capitalize and punctuate to match the original
        bbi = set()
        if is_two_inputs:
            bbi = get_blacklisted_perturbations_indices_in_two_inputs_tasks(clean_split_sentence, dataset.get_second_input(i))
            total_blacklisted_words += len(bbi)

        counts = prepare_pertrubed_sentences(
            bbi, capitalized, clean_split_sentence, inds, is_two_inputs, neighbors, orig_split_sent, punct, sents, total_cands,
            total_cands_due_to_captialize, total_cands_due_to_punctuation, total_cands_lost, total_words, args.clip_tokens)
        total_cands, total_cands_due_to_captialize, total_cands_due_to_punctuation, total_cands_lost, total_words = counts

        # Filtering invalid candidates
        mask = perts_filter(sents, inds)

        # add syns that were not filtered out
        syns_for_i = defaultdict(list)
        for m, j, sent in zip(mask, inds, sents):
            if m:
                total_syns += 1
                total_syns_due_to_captialize += int(j in capitalized)
                w = sent[j] # clean it back to have the correct code for it
                if j in punct:
                    total_syns_due_to_punctuation += 1
                    # w = w[:-1]
                syns_for_i[j].append(w)
        cached_synonyms.add_new_sent(sentence, syns_for_i, i)
        assert len(cached_synonyms) == curr_size + 1

    if temp_name is not None:
        with open(temp_name, 'wb') as f:
            pickle.dump(cached_synonyms, f)

    if updater:
        print(f'Analyzed {(end - start) if end is not None else len(start)} sentences with {total_words} words.\n'
              f'Found a total of {total_cands} candidates of which {total_syns} synonyms were verified')
        print(f'The number of capitalized words were {total_words_capitalized} which accounted for {total_cands_due_to_captialize} '
              f'candidates and {total_syns_due_to_captialize} synonyms')
        print(f'The number of punctuated words were {total_words_punctuation} which accounted for {total_cands_due_to_punctuation} '
              f'candidates and {total_syns_due_to_punctuation} synonyms')
        if is_two_inputs:
            print(f'The data contained two inputs which caused a total of {total_blacklisted_words} to be blacklisted which '
                  f'forfeited {total_cands_lost} candidates')
    return cached_synonyms


def prepare_pertrubed_sentences(bbi, capitalized, clean_split_sentence, inds, is_two_inputs, neighbors, orig_split_sent, punct, sents,
                                total_cands, total_cands_due_to_captialize, total_cands_due_to_punctuation, total_cands_lost, total_words,
                                clip_tokens):
    for j, w in enumerate(clean_split_sentence):
        if j >= clip_tokens:
            # no need to compute tokens for later words
            break
        total_words += 1
        syns = [syn for syn in neighbors.get(w, []) if syn != w]

        if is_two_inputs and j in bbi:
            # can count how many perturbations were skipped because of it, or at least how many words were skipped
            total_cands_lost += len(syns)
            continue
        total_cands += len(syns)
        if len(syns) == 0:
            continue
        if j in capitalized:
            total_cands_due_to_captialize += len(syns)
            syns = [syn.capitalize() for syn in syns]
        if j in punct:
            total_cands_due_to_punctuation += len(syns)
            syns = [punct[j][0] + syn + punct[j][1] for syn in syns]
        for syn in syns:
            sents.append(deepcopy(orig_split_sent))
            sents[-1][j] = syn
            inds.append(j)
    return total_cands, total_cands_due_to_captialize, total_cands_due_to_punctuation, total_cands_lost, total_words


def _load_dataset(args):
    return get_dataset(args.dataset.lower().replace('-', ''), args.data_dir, train=args.train)


BLACK_LIST_POSES = {'PROPN', 'NOUN', 'ADJ', 'VERB'}
def get_blacklisted_perturbations_indices_in_two_inputs_tasks(split_sentence1, sentence2):
    s2_poses = get_spacy_pos(sentence2)
    black_list_words = {w for w, p in zip(sentence2.split(), s2_poses) if p in BLACK_LIST_POSES}
    return {i for i, w in enumerate(split_sentence1) if w in black_list_words}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neighbors_path', default=NEIGHBORS_FILE)
    parser.add_argument('--data_dir', default=GLUE_DATA_DIR)
    parser.add_argument('--dataset', choices=('SST-2', 'BoolQ', 'IMDB'), default='SST-2')
    parser.add_argument('--tfhub_cache_path', type=str, default='/media/disk1/maori/cache')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_processes', type=int, default=8)

    parser.add_argument('--perform_tf_pos_filtering', action='store_true')

    parser.add_argument('--perform_use_semantic_filtering', action='store_true')
    parser.add_argument('--sim_score_window', type=int, default=15)
    parser.add_argument('--sim_score_threshold', type=float, default=0.7)

    parser.add_argument('--perform_spacy_pos_filtering', action='store_true')
    parser.add_argument('--spacy_pos_score_window', type=int, default=15)

    parser.add_argument('--perform_gp2_ppl_filtering', action='store_true')
    parser.add_argument('--ppl_score_window', type=int, default=101)
    parser.add_argument('--ppl_score_ratio_threshold', type=float, default=0.8)

    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_dev', action='store_true')
    parser.add_argument('--do_hashing', action='store_true')

    parser.add_argument('--first_sample', type=int, default=0)
    parser.add_argument('--debug_samples', type=int, default=-1)
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--clip_tokens', type=int, default=480, help='Will only compute synonyms to words numbered under that value')

    args = parser.parse_args()

    assert not (args.perform_spacy_pos_filtering and args.perform_tf_pos_filtering)
    # assert not (args.perform_use_semantic_filtering and args.perform_gp2_ppl_filtering)

    os.environ['TFHUB_CACHE_DIR'] = args.tfhub_cache_path
    np.random.seed(args.seed)
    ctx = get_context('spawn')

    # create a vocabulary of all words in the synonyms candidates
    args.vocab_path = os.path.splitext(args.neighbors_path)[0] + '_vocab.json'
    if not os.path.exists(args.vocab_path):
        print('Creating vocabulary...', end=' ')
        with open(args.neighbors_path) as f:
            neighbors = json.load(f)
        vocabulary = {w: i for i, w in enumerate(set(neighbors.keys()).union(chain.from_iterable(neighbors.values())))}
        with open(args.vocab_path, 'w') as f:
            json.dump(vocabulary, f)
        print('Done')

    for i, is_train in enumerate((False, True)):
        dataset = None
        args.train = is_train

        if args.do_hashing:
            dataset = _load_dataset(args)
            print(f'Creating hashes for train={is_train}')
            ordered_ds = {(' '.join(dataset.get_sentence_to_perturb(i)[0])).lower(): i for i in trange(len(dataset))}
            args.out_path = os.path.join(os.path.dirname(args.neighbors_path), args.dataset,
                                         'hashes_' + ('train' if is_train else 'dev') + '.json')
            dnmae = os.path.dirname(args.out_path)
            if not os.path.isdir(dnmae):
                os.makedirs(dnmae)
            with open(args.out_path, 'w') as f:
                json.dump(ordered_ds, f)

        if is_train and args.skip_train:
            print("skipping train set")
            continue
        if not is_train and args.skip_dev:
            print("skipping dev set")
            continue
        if dataset is None:
            dataset = _load_dataset(args)

        args.base_progress = 0.5 * i
        print(f'Starting on train={is_train}')
        args.out_path = os.path.join(os.path.dirname(args.neighbors_path), args.dataset,
                                     'synonyms_' + ('train' if is_train else 'dev'))

        dnmae = os.path.dirname(args.out_path + '.pkl')
        if not os.path.isdir(dnmae):
            os.makedirs(dnmae)

        n_samples = len(dataset)
        start = args.first_sample
        assert 0 <= start < n_samples
        n_samples -= start
        if args.debug_samples > 0:
            n_samples = min(n_samples, args.debug_samples)
        end = start + n_samples
        if args.first_sample > 0 or args.debug_samples > 0:
            args.out_path = args.out_path + f'_{start}_{end}'
        print('total samples: ', n_samples)

        if args.n_processes > 1:
            splits = np.linspace(start, end, num=args.n_processes + 1).astype(int)
            import torch
            devices = [i % torch.cuda.device_count() for i in range(args.n_processes)]
            with ctx.Pool(args.n_processes) as executor:
                # args, starting_sample_num=0, last_sample_num=np.inf, cuda_device=0, partial_proc_num=None, return_dataset=True
                outs = list(executor.map(analyze_dataset_chunk, zip(splits[:-1], splits[1:], [args]*args.n_processes, devices)))
            cached_synonyms = CachedSynonyms.merge_multiple_cached_synonyms(*outs)
        else:
            cached_synonyms = analyze_dataset_chunk((start, end, args, 0))

        with open(args.out_path + '.pkl', 'wb') as f:
            pickle.dump(cached_synonyms, f)
        # make sure we prepared all sentences
        assert len(cached_synonyms) == n_samples
