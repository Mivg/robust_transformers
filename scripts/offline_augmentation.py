# this script takes in training data file and a strategy to attack with, and augment the data with attacks.
# Note that we only attack samples that was originally succeeding
import json
from argparse import ArgumentParser
import os

import sys
from collections import namedtuple

import numpy as np
import attr

from tqdm import tqdm
from time import time, sleep

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.glue_datasets import get_dataset, get_dataset_names
from attacks.interfaces import SampleResult
from attacks.models import get_model, get_model_names
from attacks.synonyms_utils import set_mock_sizes

from attacks.strategies import get_strategy_names, get_strategy

GLUE_DATA_DIR = os.environ.get('GLUE_DIR', '/media/disk2/maori/data')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--glue_data_dir', default=GLUE_DATA_DIR)
    parser.add_argument('--use_failed_attacks', action='store_true', help='if set, also uses attacks which failed to fool the model')
    parser.add_argument('--dataset', choices=get_dataset_names(), default='SST-2')
    parser.add_argument('--snli_pert_sent', choices=(1, 2), default=1, type=int)
    parser.add_argument('--strategy_name', choices=get_strategy_names(), default='rand_synonym_swap')
    parser.add_argument('--strategy_params', type=json.loads, default='{}')
    parser.add_argument('--model', help='which model to test', choices=get_model_names(), required=True)
    parser.add_argument("--eval_batch_size", default=None, type=int, required=False)
    parser.add_argument("--model_path", default=None, type=str, required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_attacks', help='MAXIMAL number of attacks to perform on each sample (budget)', type=int, required=True)
    parser.add_argument('--explore_batch_size', help='how many exploration samples to request at once.', type=int, required=False,
                        default=100)
    parser.add_argument('--n_simultaneous_samples', type=int, default=20, help='Number of samples to analyze together')

    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--skip_failures', action='store_true')

    parser.add_argument("--tfhub_cache_dir", type=str, help="path to which tensorflow hub should cache models",
                        default='/media/disk2/maori/tfhub_cache')
    parser.add_argument("--cache_dir", type=str, help="path to which hugging face should cache models",
                        default='/media/disk2/maori/data/hf_cache')

    parser.add_argument('--n_processes', help='if more than 1, split the computation into multiple processes, using all '
                                              'available gpus if needed', type=int, default=1)

    args = parser.parse_args()
    os.environ['TFHUB_CACHE_DIR'] = args.tfhub_cache_dir

    return args

@attr.s
class AnalyzedSample(object):
    sample_num = attr.ib()
    orig_sentence = attr.ib()
    label = attr.ib()
    orig_sample_prob = attr.ib()
    left_budget = attr.ib()


def run_analysis(args):
    start, end, cuda_device, args = args
    set_mock_sizes(True)
    print(f'Requested to analyze {start} to {end}')
    mp_mode = end is not None
    updater = start == 0
    kwargs = vars(args)
    dataset = get_dataset(args.dataset, train=True, **kwargs)
    if mp_mode:
        assert args.num_samples == len(dataset), 'In MP mode can only analyze the entire dataset'
        n_samples = end - start
    else:
        assert start == 0, 'Non mp mode can only start in the first sample'
        n_samples = args.num_samples
        end = len(dataset)

    if len(dataset) < n_samples:
        raise ValueError(f'Requested to test {n_samples} samples but {args.dataset} contains only {len(dataset)} samples')
    dataset.shuffle(args.seed)
    print(f'Loaded dataset in start=={start}')

    strategies = [get_strategy(args.strategy_name, **args.strategy_params, dataset=args.dataset, train=True)
                  for _ in range(args.n_simultaneous_samples)]
    print(f'Loaded strategies in start=={start}')

    model = get_model(args.model, **kwargs, n_labels=len(dataset.get_labels()), cuda_device=cuda_device)
    print(f'Loaded model in start=={start}')

    consecutive_failures = 0
    augmentations = {}  # sample id -> new sample
    sample_num = start - 1
    MAX_CONSECUTIVE_FAILURES = 8
    model_mistakes = 0
    skipped_samples_due_to_no_sig = 0
    skipped_due_no_attacks = 0
    n_successful_attacks = 0

    if updater:
        pbar = tqdm(total=n_samples)
    print('Starting attacks')
    while len(augmentations) < n_samples and sample_num < end - 1:

        t0 = time()
        analyzed_samples = []
        n_samples_to_analyze = min(n_samples - len(augmentations), end - 1 - sample_num, args.n_simultaneous_samples)
        # start collecting a batch of samples to propagate through the model together for higher efficiency
        while sample_num < end - 1 and len(analyzed_samples) < n_samples_to_analyze:
            # init sample robustness test
            sample_num += 1
            orig_sentence, label = dataset.get_sentence_to_perturb(sample_num)

            orig_sample_prob = model.predict_proba([dataset.generate_perturbed_sample(sample_num, orig_sentence)])[0]
            orig_correct = label == np.argmax(orig_sample_prob)
            if not orig_correct:  # no point in attacking samples that are already wrong
                consecutive_failures += 1
                if consecutive_failures == MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f'Something seems to be off with this model since it got wrong '
                                       f'{MAX_CONSECUTIVE_FAILURES} samples in a row')
                model_mistakes += 1
                continue
            consecutive_failures = 0
            analyzed_samples.append(AnalyzedSample(sample_num, orig_sentence, label, orig_sample_prob, args.num_attacks))
            try:
                strategy = strategies[len(analyzed_samples)-1]
                analyzed_sample = analyzed_samples[-1]
                strategy.init_attack(analyzed_sample.orig_sentence, analyzed_sample.label, args.num_attacks)
            except KeyError as e:
                if args.skip_failures:
                    # temp solution for imdb since it's attack space is not yet full
                    print(f'Warning - skipped sentence due to missing cache.')
                    skipped_samples_due_to_no_sig += 1
                    analyzed_samples = analyzed_samples[:-1]
                else:
                    print('failed to init attack and skipping is not permitted')
                    raise e

        strategies = strategies[:len(analyzed_samples)]

        max_budget = args.num_attacks

        # and start attacking to get the augmented input
        while max_budget > 0:
            new_attacks = []
            attack_strategies_indices = []
            edges = [0]
            for s_num, (strategy, analyzed_sample) in enumerate(zip(strategies, analyzed_samples)):
                explore_size = min(analyzed_sample.left_budget, args.explore_batch_size)
                attacks = strategy.explore(explore_size)
                edges.append(len(new_attacks) + len(attacks))
                if len(attacks) == 0:
                    continue
                analyzed_sample.left_budget = analyzed_sample.left_budget - len(attacks)
                new_attacks.extend(attacks)
                attack_strategies_indices.extend([s_num] * len(attacks))
            if len(new_attacks) == 0:
                break
            max_budget = max(ans.left_budget for ans in analyzed_samples)

            samples = [dataset.generate_perturbed_sample(analyzed_samples[s_num].sample_num, attack)
                       for s_num, attack in zip(attack_strategies_indices, new_attacks)]
            probs = model.predict_proba(samples)
            predicted_labels = np.argmax(probs, axis=1)
            for s_num, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
                if e > s:
                    label = analyzed_samples[s_num].label
                    strategies[s_num].update(list(map(lambda res: SampleResult(*res),
                                                      zip(new_attacks[s:e], probs[s:e, label], predicted_labels[s:e] != label))))

        final_attacks = [strategy.exploit(1) for strategy in strategies]  # random may have nothing to return here and cause an index out of bounds error
        had_attacks = [len(fa) == 1 and fa[0] != analyzed_sample.orig_sentence for fa, analyzed_sample in zip(final_attacks, analyzed_samples)]
        skipped_due_no_attacks += len(had_attacks) - sum(had_attacks)
        final_attacks = [fa[0] for fa, ha in zip(final_attacks, had_attacks) if ha]
        analyzed_samples = [ans for ans, ha in zip(analyzed_samples, had_attacks) if ha]
        ready_attacks = [dataset.generate_perturbed_sample(analyzed_sample.sample_num, final_attack) for analyzed_sample, final_attack in
                         zip(analyzed_samples, final_attacks)]
        best_attack_probs = []
        if sum(had_attacks) > 0:
            best_attack_probs = model.predict_proba(ready_attacks)
        assert len(final_attacks) == len(analyzed_samples)
        assert len(final_attacks) == len(best_attack_probs)

        added = 0
        for analyzed_sample, best_attack_prob, final_attack in zip(analyzed_samples, best_attack_probs, final_attacks):
            model_correct_on_attack = analyzed_sample.label == np.argmax(best_attack_prob)
            n_successful_attacks += int(not model_correct_on_attack)
            if not args.use_failed_attacks and model_correct_on_attack:
                continue
            augmentations[analyzed_sample.sample_num] = ' '.join(final_attack)
            added += 1
        if added > 0 and updater:
            pbar.update(added+skipped_due_no_attacks)

    if not mp_mode:
        # this is a single process job
        print(f'Analyzed {sample_num + 1} samples out of {len(dataset)} total samples, of which the model was originally wrong in '
              f'{model_mistakes}, and we were able to fool it on {n_successful_attacks} but adding {len(augmentations)} augmentations')
        print(f'Skipped {skipped_samples_due_to_no_sig} samples due to missing signatures, and {skipped_due_no_attacks} due to no attacks')
        use_failed_str = '_with_all' if args.use_failed_attacks else ''
        datasets_names = {'boolq': 'BoolQ', 'sst2': 'SST-2'}
        dataset_name = datasets_names.get(args.dataset.lower(), args.dataset.upper())
        out_path = os.path.join(args.glue_data_dir, dataset_name,
                                f'train_{len(augmentations)}_augs{use_failed_str}_{strategy}_{args.num_attacks}attacks'.replace('-', 'NEG'))
        out_path = out_path.replace('.', '_').replace(' ', '_').replace('=', '__').replace(',', '_')
        print('finished creating the attacks, now need to save them to', out_path)
        dataset.create_augmented_dataset(augmentations, out_path)
    else:
        return model_mistakes, n_successful_attacks, skipped_samples_due_to_no_sig, skipped_due_no_attacks, augmentations, str(strategy)


def _merge_outputs(outs):
    model_mistakes = sum(out[0] for out in outs)
    n_successful_attacks = sum(out[1] for out in outs)
    skipped_samples_due_to_no_sig = sum(out[2] for out in outs)
    skipped_due_no_attacks = sum(out[3] for out in outs)
    augmentations = {}
    for i, out in enumerate(outs):
        print(f'Process {i} create a {len(out[4])} augmentations')
        intersection_size = len(set(out[4].keys()).intersection(augmentations.keys()))
        if intersection_size > 0:
            raise RuntimeError(f'multiple process accounted for the same indices! (intersection size so far: {intersection_size}')
        augmentations.update(out[4])
    print(f'Total number of augmentations created {len(augmentations)} with min key: {min(augmentations.keys())} and max key: {max(augmentations.keys())}')
    return model_mistakes, n_successful_attacks, skipped_samples_due_to_no_sig, skipped_due_no_attacks, augmentations, outs[0][5]


def main():
    args = get_args()
    set_mock_sizes(True)

    t0 = time()
    if args.n_processes == 1:
        print('Running offline adversarial augmentation with a single process')
        run_analysis((0, None, '0', args))
        print(f'Finished creating results in {(time() - t0) / 3600:.1f} hours')
    else:
        print(f'Running offline adversarial augmentation with a {args.n_processes} processes')
        from multiprocessing import get_context
        import torch
        devices = torch.cuda.device_count()
        splits = np.linspace(0, args.num_samples, num=args.n_processes+1).astype(int)
        params_it = ((start, end, i % devices, args) for i, (start, end) in enumerate(zip(splits[:-1], splits[1:])))
        ctx = get_context('spawn')

        with ctx.Pool(args.n_processes) as executor:
            # args, starting_sample_num=0, last_sample_num=np.inf, cuda_device=0, partial_proc_num=None, return_dataset=True
            outs = list(executor.map(run_analysis, params_it))
        model_mistakes, n_successful_attacks, skipped_samples_due_to_no_sig, skipped_due_no_attacks, augmentations, strategy_str = _merge_outputs(outs)
        print(f'Finished creating results in {(time() - t0) / 3600:.2f} hours')
        t1 = time()
        dataset = get_dataset(args.dataset, train=True, **vars(args))
        dataset.shuffle(args.seed)
        print(f'Finished merging results in {(time() - t1):.2f} seconds')

        print(f'Analyzed {len(dataset)} samples out of {len(dataset)} total samples, of which the model was originally wrong in '
              f'{model_mistakes}, and we were able to fool it on {n_successful_attacks} but adding {len(augmentations)} augmentations')
        print(f'Skipped {skipped_samples_due_to_no_sig} samples due to missing signatures, and {skipped_due_no_attacks} due to no attacks')
        use_failed_str = '_with_all' if args.use_failed_attacks else ''
        datasets_names = {'boolq': 'BoolQ', 'sst2': 'SST-2'}
        dataset_name = datasets_names.get(args.dataset.lower(), args.dataset.upper())
        out_path = os.path.join(args.glue_data_dir, dataset_name, f'train_{len(augmentations)}_augs{use_failed_str}_{strategy_str}_{args.num_attacks}attacks'.replace('-', 'NEG'))
        out_path = out_path.replace('.', '_').replace(' ', '_').replace('=', '__').replace(',', '_')
        print('finished creating the attacks, now need to save them to', out_path)
        dataset.create_augmented_dataset(augmentations, out_path)


if __name__ == '__main__':
    main()
