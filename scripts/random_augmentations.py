import os, sys
import argparse
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.synonyms_utils import load_synonym_for_attacks, synonym_file_short_name
from attacks.text_fooler_filters import get_filter
from common.utils import pertrub_sentence, read_glue_data

NEIGHBORS_FILE = '../data/counterfitted_neighbors.json'
VOCAB_FILE = '../data/text_fooler_synonyms_vocab.json'
GLUE_DATA_DIR = os.environ.get('GLUE_DIR', '/media/disk2/maori/data')
N_PERTURBATIONS = 10  # Including the original


# TODO - can add the grand total of possibilities for each edit distance and sentence index

def pertrub_sst2(data, args, neighbors, require_pos):
    # initializations
    aux_df_records = []
    records = []
    total_skipped = 0

    labels = np.array([], dtype=np.int32)
    for i, sentence in tqdm(enumerate(data['sentence']), total=len(data)):
        ed = int(args.edit_distance * len(sentence.split(' '))) if 0 < args.edit_distance < 1 else args.edit_distance
        ed = max(ed, 1)  # make sure it's not 0

        sentence = re.sub(' +', ' ', sentence).strip()  # remove multiple whitespace
        perts_filter = get_filter(filter_by_tf_pos=args.perform_pos_filtering, filter_by_sem=args.perform_semantic_filtering,
                                  orig_sent=sentence.split(' '), return_mask=True, tfhub_cache_path=None,
                                  sim_score_window=args.sim_score_window, sim_score_threshold=args.sim_score_threshold)

        dist_to_sentences_dict, success_flag, total_sentences = pertrub_sentence(sentence, i, args.pert, neighbors, ed,
                                                                                 args.edit_dist_pert, args.dump_all,
                                                                                 filter_func=perts_filter, requires_pos=require_pos,
                                                                                 replace_orig_if_needed=True)
        if not success_flag and not args.dump_all:
            total_skipped += 1
            continue

        labels = np.concatenate((labels, np.full(total_sentences, fill_value=data.label.values[i])))

        for ed, perts in dist_to_sentences_dict.items():
            records.extend(perts)
            aux_df_records.extend([(i, ed)] * len(perts))
    new_df = pd.DataFrame.from_dict({'sentence': records, 'label': labels})
    if 'idx' in data:
        new_df['idx'] = [f'{i}_' for i in range(len(new_df))]  # force it to be a string
        new_df = new_df[data.columns].astype(data.dtypes)
    aux_df = pd.DataFrame(columns=('SentnceIndex', 'EditDistance'), data=aux_df_records)
    return total_skipped, new_df, aux_df


def pertrub_snli(data, args, neighbors, requires_pos):
    if args.snli_pert == 'both':
        raise NotImplementedError
    # initializations
    aux_df_records = []
    records1 = []
    records2 = []
    pert_key, non_pert_key, pert_recs, non_pert_rec = ('sentence1', 'sentence2', records1, records2) if args.snli_pert == 'sent1' else \
        ('sentence2', 'sentence1', records2, records1)
    total_skipped = 0
    labels = []
    for i, (pert_sent, non_pert_sent) in tqdm(enumerate(zip(data[pert_key], data[non_pert_key])), total=len(data)):
        ed = int(args.edit_distance * len(pert_sent.split(' '))) if 0 < args.edit_distance < 1 else args.edit_distance
        if ed == 0:
            success_flag, total_sentences = False, 1
            dist_to_sentences_dict = {0: [pert_sent]}
        else:

            pert_sent = re.sub(' +', ' ', pert_sent).strip()  # remove multiple whitespace

            perts_filter = get_filter(filter_by_tf_pos=args.perform_pos_filtering, filter_by_sem=args.perform_semantic_filtering,
                                      orig_sent=pert_sent.split(' '), return_mask=True, tfhub_cache_path=None,
                                      sim_score_window=args.sim_score_window, sim_score_threshold=args.sim_score_threshold)

            dist_to_sentences_dict, success_flag, total_sentences = pertrub_sentence(pert_sent, i, args.pert, neighbors, ed,
                                                                                     args.edit_dist_pert, args.dump_all,
                                                                                     filter_func=perts_filter, requires_pos=require_pos,
                                                                                     replace_orig_if_needed=True)
        if not success_flag and not args.dump_all:
            total_skipped += 1
            continue

        labels.append(np.full(total_sentences, fill_value=data.label.values[i]))
        non_pert_rec.extend([non_pert_sent] * total_sentences)

        for ed, perts in dist_to_sentences_dict.items():
            pert_recs.extend(perts)
            aux_df_records.extend([(i, ed)] * len(perts))
    labels = np.concatenate(labels)
    new_df = pd.DataFrame.from_dict({'sentence1': records1, 'sentence2': records2, 'label': labels}).reindex(columns=data.columns)
    new_df[data.columns[0]] = np.arange(len(new_df))
    aux_df = pd.DataFrame(columns=('SentnceIndex', 'EditDistance'), data=aux_df_records)
    return total_skipped, new_df, aux_df


def pertrub_boolq(data, args, neighbors, requires_pos):
    # initializations
    aux_df_records = []
    passage_records = []
    question_records = []
    total_skipped = 0

    labels = np.array([], dtype=np.int32)
    for i, (passage, question) in tqdm(enumerate(zip(data['passage'], data['question'])), total=len(data)):
        ed = int(args.edit_distance * len(passage.split(' '))) if 0 < args.edit_distance < 1 else args.edit_distance
        ed = max(ed, 1)  # make sure it's not 0

        passage = re.sub(' +', ' ', passage).strip()  # remove multiple whitespace

        perts_filter = get_filter(filter_by_tf_pos=args.perform_pos_filtering, filter_by_sem=args.perform_semantic_filtering,
                                  orig_sent=passage.split(' '), return_mask=True, tfhub_cache_path=None,
                                  sim_score_window=args.sim_score_window, sim_score_threshold=args.sim_score_threshold)

        dist_to_sentences_dict, success_flag, total_sentences = pertrub_sentence(passage, i, args.pert, neighbors, ed,
                                                                                 args.edit_dist_pert, args.dump_all,
                                                                                 filter_func=perts_filter, requires_pos=require_pos,
                                                                                 replace_orig_if_needed=True)

        if not success_flag and not args.dump_all:
            total_skipped += 1
            continue

        labels = np.concatenate((labels, np.full(total_sentences, fill_value=data.label.values[i])))
        question_records.extend([question] * total_sentences)

        for ed, perts in dist_to_sentences_dict.items():
            passage_records.extend(perts)
            aux_df_records.extend([(i, ed)] * len(perts))
    new_df = pd.DataFrame.from_dict({'passage': passage_records, 'question': question_records, 'label': labels,
                                     'idx': np.arange(len(question_records))})
    aux_df = pd.DataFrame(columns=('SentnceIndex', 'EditDistance'), data=aux_df_records)
    return total_skipped, new_df, aux_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neighbors_path', default=None)
    parser.add_argument('--data_dir', default=GLUE_DATA_DIR)
    parser.add_argument('--out_dir', help='if given, will create a dir under the data_dir with his name, otherwise outputs will be saved '
                                          'in the task data dir', default=None)
    parser.add_argument('--task', choices=('SST-2', 'SNLI', 'BoolQ', 'MNLI', 'IMDB'), default='SST-2')
    parser.add_argument('--snli_pert', choices=('sent1', 'sent2', 'both'), default='sent1')

    parser.add_argument('--dataset', choices=('train', 'dev', 'val', 'test', 'dev_matched'), default='train')

    parser.add_argument('--pert', type=int, default=N_PERTURBATIONS)
    parser.add_argument('--dump_all', action='store_true', help='If true, also dumps the sentences that could not be perturbed enough '
                                                                'to meed requirements, along with all of their perturbations. '
                                                                'Otherwise, skips them. Irrelevant when <edit_dist_pert> is true '
                                                                '(defaults to true in this case)')
    parser.add_argument('--edit_distance', type=float, default=-1, help='if bigger than 0, this is the required edit distance between the '
                                                                      'original and the perturbed sentence')
    parser.add_argument('--edit_dist_pert', help='If true, performs at most <pert> perturbation for each edit distance up to '
                                                 '<edit_distance> which must be at least 1', action='store_true')

    parser.add_argument('--tfhub_cache_path', type=str, default='/media/disk2/maori/tfhub_cache')
    parser.add_argument('--perform_pos_filtering', action='store_true')
    parser.add_argument('--perform_semantic_filtering', action='store_true')
    parser.add_argument('--sim_score_window', type=int, default=15)
    parser.add_argument('--sim_score_threshold', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()

    os.environ['TFHUB_CACHE_DIR'] = args.tfhub_cache_path

    # args validation
    if args.edit_dist_pert:
        if args.edit_distance < 1:
            raise ValueError(f"Can't perform edit_distance perturbation analysis if the edit_distance is not strictly positive")
        args.dump_all = True

    np.random.seed(args.seed)

    # load data

    neighbors, require_pos = load_synonym_for_attacks(args.neighbors_path, args.task, args.dataset == 'train')
    assert not require_pos
    data = read_glue_data(args.data_dir, args.task, args.dataset)

    print(f'Perturbing {args.task} with edit distance {args.edit_distance}')
    added_details = ''
    if args.task == 'SST-2' or args.task == 'IMDB':
        total_skipped, new_df, aux_df = pertrub_sst2(data, args, neighbors, require_pos)
    elif args.task == 'BoolQ':
        total_skipped, new_df, aux_df = pertrub_boolq(data, args, neighbors, require_pos)
    elif args.task == 'SNLI' or args.task == 'MNLI':
        total_skipped, new_df, aux_df = pertrub_snli(data, args, neighbors, require_pos)
        added_details = f'{args.snli_pert}_'
    else:
        raise ValueError(f'Unknown task: {args.task}')

    print(f'Perturbed {len(data) - total_skipped}/{len(data)} {args.pert} times ({total_skipped} sentences were not perturbed '
          f'and {"were appended to the end" if args.dump_all else "skipped"})')
    print(f'Finished with {len(new_df)} samples, originally had {len(data)} samples')

    ed_str = f'{"full" if args.edit_dist_pert else "unconstrained" if args.edit_distance < 0 else str(args.edit_distance).replace(".", "_")}'
    filter_str = f'{"pos_" if args.perform_pos_filtering else ""}{"sem_" if args.perform_semantic_filtering else ""}' \
        f'{"filter_" if args.perform_semantic_filtering or args.perform_pos_filtering else ""}'
    base_out_dir = os.path.join(args.data_dir, args.task) if args.out_dir is None else os.path.join(args.data_dir, args.task, args.out_dir)
    if not os.path.isdir(base_out_dir):
        os.makedirs(base_out_dir)
    base_fname = os.path.join(base_out_dir,
                              f'{args.dataset}_{args.pert}_{added_details}{synonym_file_short_name(args.neighbors_path)}_'
                              f'Neighbors_{ed_str}_{filter_str}Analysis')
    if args.task == 'BoolQ':
        fname = base_fname + '_pert.jsonl'
        with open(fname, 'w', encoding="utf-8") as lines:
            new_df['label'] = new_df['label'].astype(bool)
            lines.write(new_df[data.columns].to_json(orient='records', lines=True))
    elif args.task == 'IMDB':
        fname = base_fname + '_pert.csv'
        new_df.to_csv(fname, index=False)
    else:
        fname = base_fname + '_pert.tsv'
        new_df.to_csv(fname, sep='\t', index=False)
    print(f'Dumped pertrub set to {fname}')

    fname = base_fname + '_aux.tsv'
    aux_df.to_csv(fname, sep='\t', index=False)
    print(f'Dumped pertrub auxiliary set to {fname}')

