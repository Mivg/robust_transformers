import operator as op
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import defaultdict
from time import time, sleep

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.interfaces import SampleResult
from attacks.synonyms_utils import set_cap_at, set_mock_sizes
from attacks.glue_datasets import get_dataset_names, get_dataset
from attacks.models import get_model_names, get_model
from attacks.plot_utils import dump_all_plots
from attacks.strategies import get_strategy

# TODO - what happens for samples that do not allow any attacks of required ed in strategies that require exact ed?


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--glue_data_dir', help='path to glue data', required=True)
    parser.add_argument('--out_dir', help='where to dump the results', required=True)

    parser.add_argument('--n_processes', help='if more than 1, split the computation into multiple processes, using all '
                                              'availble gpus if needed', type=int, default=1)

    parser.add_argument('--dataset', help='which dataset to use', choices=get_dataset_names(), required=True)
    parser.add_argument('--snli_pert_sent', help='which sentence to perturb (if using snli)', choices=(1, 2), required=False, default=1,
                        type=int)
    parser.add_argument('--strategies', help='which attack strategies to use, a path to a yaml file', type=str, default='strategies.yaml')
    parser.add_argument('--model', help='which model to test', choices=get_model_names(), required=True)

    parser.add_argument('--seed', help='seed to use before shuffling the test data', type=int, default=42)
    parser.add_argument('--num_samples', help='number of (correct) samples to test against', type=int, required=True)
    parser.add_argument('--include_wrong', help='if set, then samples that were wrong in the first place would be counted as well',
                        action='store_true')
    parser.add_argument('--num_attacks', help='MAXIMAL number of attacks to perform on each sample (budget)', type=int, required=True)
    parser.add_argument('--explore_batch_size', help='how many exploration samples to request at once.', type=int, required=False,
                        default=1)
    parser.add_argument('--explore_max_rounds', help='how many exploration rounds to perform.', type=int, required=False,
                        default=None)
    parser.add_argument('--exploit_every', help='every how many explorations rounds should test exploit results '
                                                '(note that if strategies do not make complete use of a round its their problem)',
                        type=int, required=False, default=5)
    parser.add_argument('--analyze_winners_locs', nargs='+', type=int, help='list of locations to in which perform winner analysis',
                        default=(50, 100, 250, 500))
    parser.add_argument('--cap_attack_size', help='If set to a positive number, will cap the attack space computation to that number '
                                                  'shortening time but reducing plots accuracy. if set to -1, will not compute attack '
                                                  'space size at all', type=int, default=0)
    parser.add_argument('--save_every', help='every how many explorations rounds should test exploit results be saved',
                        type=int, required=False, default=20)
    # TODO -allow multiple models on multiple GPUS

    parser.add_argument("--eval_batch_size", default=None, type=int, required=False)
    parser.add_argument("--model_path", default=None, type=str, required=False)

    parser.add_argument("--tfhub_cache_dir", type=str, help="path to which tensorflow hub should cache models",
                        default='/media/disk2/maori/tfhub_cache')
    parser.add_argument("--cache_dir", type=str, help="path to which hugging face should cache models",
                        default='/media/disk2/maori/data/hf_cache')
    parser.add_argument("--blacklist_orig_samples", help='If given, will skip those samples if presented, comma separated numbers', default='')

    args = parser.parse_args()

    if args.explore_max_rounds is None:
        args.explore_max_rounds = args.num_attacks
        # args.explore_max_rounds = int(args.num_attacks / args.explore_batch_size)
    # assert args.explore_batch_size <= args.num_attacks
    # assert args.explore_max_rounds * args.explore_batch_size <= args.num_attacks
    assert args.exploit_every <= args.explore_max_rounds
    assert args.save_every % args.exploit_every == 0
    args.analyze_winners_locs = sorted(args.analyze_winners_locs)
    assert args.analyze_winners_locs[-1] <= args.num_attacks
    assert args.analyze_winners_locs[0] >= args.exploit_every
    for awl in args.analyze_winners_locs:
        assert awl % args.exploit_every == 0

    os.environ['TFHUB_CACHE_DIR'] = args.tfhub_cache_dir
    return args


class RunningCounter:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.min = np.inf
        self.max = 0
        self.extras = defaultdict(int)

    def update(self, t, **extras):
        self.count += 1
        self.total += t
        self.min = min(self.min, t)
        self.max = max(self.max, t)
        for k, v in extras.items():
            self.extras[k] += v

    def merge_(self, running_counter):
        self.count += running_counter.count
        self.total += running_counter.total
        self.min = min(self.min, running_counter.min)
        self.max = max(self.max, running_counter.max)
        for k, v in running_counter.extras.items():
            self.extras[k] += v

    @property
    def mean(self):
        return self.total / self.count

    def to_dict(self):
        res = {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'total': self.total,
            'count': self.count,
        }
        res.update({k: v / self.count for k, v in self.extras.items()})
        return res


def run_analysis(args, starting_sample_num=0, last_sample_num=np.inf, cuda_device=0, n_samples=None, return_dataset=True, update_prog=True,
                 verbose=True):
    # last_sample_num is inclusive end of slice):
    if n_samples is None:
        n_samples = args.num_samples
    kwargs = vars(args)
    timings = defaultdict(RunningCounter)

    if args.cap_attack_size != 0:
        if args.cap_attack_size < 0:
            set_mock_sizes(True)
        else:
            set_cap_at(args.cap_attack_size)

    t = time()
    print('Initializing dataset... ')
    dataset = get_dataset(args.dataset, **kwargs)
    if len(dataset) < n_samples:
        raise ValueError(f'Requested to test {n_samples} samples but {args.dataset} contains only {len(dataset)} samples')
    dataset.shuffle(args.seed)
    timings['create_dataset'].update(time()-t)

    t = time()
    print('Initializing strategies... ')
    with open(args.strategies) as f:
        strategies_confs = yaml.load(f, Loader=yaml.FullLoader)
    strategies = [get_strategy(strategy_name, **strategy_values, dataset=args.dataset, train=False) for strategy in strategies_confs
                  for strategy_name, strategy_values in strategy.items()]
    timings['create_strategies'].update(time() - t)

    t = time()
    print('Initializing models... ')
    model = get_model(args.model, **kwargs, n_labels=len(dataset.get_labels()), cuda_device=cuda_device)
    timings['create_model'].update(time() - t)

    sample_num = starting_sample_num -1
    samples_orig_losses = []  # sample_index, orig_loss, row_in_unshuffled

    exploit_attacks = []
    exploit_strategy_indices = []
    exploit_left_budgets = []
    exploit_samples_indices = []
    exploit_labels = []
    exploit_save_mask = []
    attack_space_size_dict = defaultdict(dict)
    attack_positions_size_dict = defaultdict(dict)
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 8

    pbar = tqdm(total=n_samples)
    print('Starting attacks')
    if args.include_wrong:
        if n_samples < last_sample_num and starting_sample_num == 0 and args.n_processes == 1:
            last_sample_num = n_samples-2
        else:
            last_sample_num -= 1

    while len(samples_orig_losses) < n_samples and sample_num < len(dataset)-1 and sample_num <= last_sample_num:
        t0 = time()
        # init sample robustness test
        sample_num += 1
        orig_row_num = dataset.get_orig_row(sample_num)
        if orig_row_num in args.blacklist_orig_samples:
            print(f'Skipping sample number {sample_num} with orig row {orig_row_num} since it was blacklisted')
            continue

        orig_sentence, label = dataset.get_sentence_to_perturb(sample_num)

        orig_sample_prob = model.predict_proba([dataset.generate_perturbed_sample(sample_num, orig_sentence)])[0]
        orig_correct = label == np.argmax(orig_sample_prob)
        if not orig_correct:  # no point in attacking samples that are already wrong
            consecutive_failures += 1
            if consecutive_failures == MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f'Something seems to be off with this model since it got wrong '
                                   f'{MAX_CONSECUTIVE_FAILURES} samples in a row')
            if args.include_wrong:
                pbar.update(1)
            continue
        consecutive_failures = 0
        samples_orig_losses.append((sample_num, orig_sample_prob[label], dataset.get_orig_row(sample_num)))
        # want to minimize the probability the model gives to the correct label
        to_exploit_strategies = set()

        timings['init_sample'].update(time() - t0)

        t2 = time()

        try:
            for strategy_num, strategy in enumerate(strategies):
                strategy.init_attack(orig_sentence, label, args.num_attacks)
                attack_space_size_dict[sample_num][strategy_num], attack_positions_size_dict[sample_num][strategy_num] = \
                    strategy.attack_space_size()
        except KeyError:
            # temp solution for imdb since it's attack space is not yet full
            print(f'Warning - skipped sentence due to missing cache: {" ".join(orig_sentence)}')
            samples_orig_losses = samples_orig_losses[:-1]
            continue

        left_budget = {i: args.num_attacks for i in range(len(strategies))}

        timings['init_strategies'].update(time() - t2)

        # for now, only performing a single attack on a single sample at a time
        for j in range(1, args.explore_max_rounds+1):
            t1 = time()
            # init explore round
            attacks = []
            inds = [0]

            # gather exploration samples
            for strategy_num, strategy in enumerate(strategies):
                if left_budget[strategy_num] == 0:
                    new_attacks = []
                else:
                    explore_size = min(left_budget[strategy_num], args.explore_batch_size)
                    t2 = time()
                    new_attacks = strategy.explore(explore_size)
                    left_budget[strategy_num] = left_budget[strategy_num] - len(new_attacks)
                    attacks.extend(new_attacks)
                inds.append(inds[-1]+len(new_attacks))
                if len(new_attacks) > 0:
                    to_exploit_strategies.add(strategy_num)
                    timings[f'explore-{strategy}'].update(time() - t2, attacks=len(new_attacks))

            timings['get_explorations'].update(time() - t1, attacks=len(attacks))

            if len(attacks) > 0:
                t2 = time()
                # update strategies with the results
                samples = [dataset.generate_perturbed_sample(sample_num, attack) for attack in attacks]
                timings['perturb_explorations'].update(time() - t2, attacks=len(attacks))
                t2 = time()
                probs = model.predict_proba(samples)
                predicted_labels = np.argmax(probs, axis=1)
                timings['predict_explorations'].update(time() - t2, attacks=len(attacks))
                t2 = time()
                for strategy_num, strategy in enumerate(strategies):
                    start, end = inds[strategy_num], inds[strategy_num+1]
                    if start != end:
                        t3 = time()
                        # again, we want to minimize the label prob
                        strategy.update(list(map(lambda res: SampleResult(*res),
                                                 zip(attacks[start:end], probs[start:end, label], predicted_labels[start:end] != label))))
                        timings[f'update-{strategy}'].update(time() - t3, attacks=end-start)
                timings['update'].update(time() - t2, attacks=len(attacks))
            else:
                j = args.explore_max_rounds  # make sure we exploit before we break

            timings['full_explore_round'].update(time() - t1)
            # test exploitation (only prepare the samples
            if j % args.exploit_every == 0 or j == args.explore_max_rounds:
                t1 = time()

                # TODO - there is a huge bug here when explore batch size is large - we exploit based on number of rounds, instead of number of explorations, with ebs>1 j would not get to explore_max_rounds which also mean we won't exploit in the last iteration
                is_save = j % args.save_every == 0 or j == args.explore_max_rounds
                for strategy_num, strategy in enumerate(strategies):
                    if strategy_num not in to_exploit_strategies:
                        continue
                    exploit_strategy_indices.append(strategy_num)
                    leftover_budget = left_budget[strategy_num]
                    leftover_budget -= leftover_budget % args.explore_batch_size    # although not strictly correct, would help smoothness
                    # when analyzing the results, and its the strategy problem that it didn't used up athe batched
                    exploit_left_budgets.append(leftover_budget)
                    exploit_attacks.append(strategy.exploit(1)[0])
                    exploit_samples_indices.append(sample_num)
                    exploit_labels.append(label)
                    exploit_save_mask.append(is_save)

                timings['exploit'].update(time() - t1, exploited=len(exploit_attacks))
                to_exploit_strategies = set()

            if len(attacks) == 0:
                break  # no strategy wants to explore anymore
        pbar.update(1)

        timings['full_sample'].update(time() - t0)

    # test the exploitations
    t0 = time()
    print('Testing exploitations...')
    ready_exploit_attacks = [dataset.generate_perturbed_sample(sn, attack) for sn, attack in zip(exploit_samples_indices, exploit_attacks)]
    timings['prep_final_exploits'].update(time() - t0)
    t0 = time()
    best_attack_probs = model.predict_proba(ready_exploit_attacks)
    timings['predict_final_exploits'].update(time() - t0, attacks=len(ready_exploit_attacks))

    strategy_names = [str(strategy) for strategy in strategies]
    strategy_index_name_map = {i: sname for i, sname in enumerate(strategy_names)}

    if return_dataset:
        return dataset, strategy_names, strategy_index_name_map, exploit_save_mask, ready_exploit_attacks, exploit_samples_indices, \
               exploit_labels, best_attack_probs,  exploit_strategy_indices, samples_orig_losses, attack_space_size_dict, \
               exploit_left_budgets, attack_positions_size_dict, timings

    else:
        return strategy_names, strategy_index_name_map, exploit_save_mask, ready_exploit_attacks, exploit_samples_indices, exploit_labels, \
               best_attack_probs,  exploit_strategy_indices, samples_orig_losses, attack_space_size_dict, exploit_left_budgets, \
               attack_positions_size_dict, timings


def dump_results(args, dataset, strategy_names, strategy_index_name_map, exploit_save_mask, ready_exploit_attacks, exploit_samples_indices,
                 exploit_labels, best_attack_probs, exploit_strategy_indices, samples_orig_losses, attack_space_size_dict,
                 exploit_left_budgets, attack_positions_size_dict, timings):

    t0 = time()
    labels = np.array(exploit_labels)
    is_correct = np.argmax(best_attack_probs, axis=1) == labels
    results = pd.DataFrame.from_dict({
        'strategy_index': exploit_strategy_indices,
        'sample_index': exploit_samples_indices,
        'used_budget': args.num_attacks - np.array(exploit_left_budgets),
        'attack_loss': best_attack_probs[np.arange(len(labels)), labels],
        'is_correct': is_correct

    })

    print('Dumping results...')
    save_mask = np.array(exploit_save_mask, dtype=np.bool)
    attacks_array = np.array([' '.join(rea) for rea, sm in zip(ready_exploit_attacks, save_mask) if sm])
    is_strategy_success = ~np.array(is_correct)[save_mask]
    samples_to_save = np.array(exploit_samples_indices)[save_mask]
    orig_labels = np.array([dataset.get_sentence_to_perturb(sn)[1] for sn in samples_to_save])
    orig_rows = np.array([dataset.get_orig_row(sn) for sn in samples_to_save])
    orig_sentence = np.array([' '.join(dataset.generate_perturbed_sample(sn, dataset.get_sentence_to_perturb(sn)[0])) for sn in samples_to_save])
    strategy_names_arr = np.array(strategy_names)[np.array(exploit_strategy_indices)[save_mask]]
    data = np.concatenate((samples_to_save[:, None], orig_rows[:, None], orig_sentence[:, None], orig_labels[:, None],
                           strategy_names_arr[:, None], is_strategy_success[:, None], attacks_array[:, None]), axis=1)
    columns = ['SampleIndex', 'OrigIndex', 'OrigInput', 'OrigLabel', 'StrategyName', 'StrategySuccessful', 'Attack']

    os.makedirs(args.out_dir, exist_ok=True)
    results.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)
    pd.DataFrame(data=data, columns=columns).to_csv(os.path.join(args.out_dir, 'attacks.csv'), index=False)
    orig_results = pd.DataFrame(data=samples_orig_losses, columns=['sample_index', 'orig_loss', 'row_in_unshuffled'])
    orig_results.to_csv(os.path.join(args.out_dir, 'orig_results.csv'), index=False)
    last_sample = samples_orig_losses[-1][0]+1

    orig_wrong_samples = last_sample - len(samples_orig_losses)
    print(f'Model was wrong in {orig_wrong_samples} out of {last_sample} samples that were tested')
    fooled_samples = (results[['sample_index', 'is_correct']].astype(int).groupby('sample_index')['is_correct'].min() == 0).sum()
    robust_accuracy = 1 - (fooled_samples+orig_wrong_samples) / last_sample
    print(f'Out of {last_sample} samples tested, model achieved {robust_accuracy} robust accuracy')
    if len(args.blacklist_orig_samples) > 0:
        print('Don\'t trust the above number since black list samples were used (and would be counted as false examples if encountered')

    # columns are strategy index and rows are sample index
    attack_space_size = pd.DataFrame.from_dict(attack_space_size_dict, orient='index')
    attack_space_size_df = attack_space_size.copy()
    attack_space_size_df.columns = list(strategy_index_name_map[k] for k in range(len(strategy_index_name_map)))
    attack_space_size_df.to_csv(os.path.join(args.out_dir, 'attack_space.csv'), index=True)

    # columns are strategy index and rows are sample index
    attack_space_positions = pd.DataFrame.from_dict(attack_positions_size_dict, orient='index')
    attack_space_positions_df = attack_space_positions.copy()
    attack_space_positions_df.columns = list(strategy_index_name_map[k] for k in range(len(strategy_index_name_map)))
    attack_space_positions_df.to_csv(os.path.join(args.out_dir, 'attack_positions.csv'), index=True)

    # copy the strategies and configs used here
    shutil.copy2(args.strategies, os.path.join(args.out_dir, 'strategies.yaml'))
    with open(os.path.join(args.out_dir, 'conf.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    timings['dump_csvs'].update(time() - t0)
    t0 = time()

    # Now, create plots to depict results
    dump_all_plots(results, orig_results, attack_space_size, attack_space_positions, strategy_index_name_map, os.path.abspath(args.out_dir),
                   args.exploit_every, args.analyze_winners_locs)
    timings['dump_plots'].update(time() - t0)

    pd.DataFrame.from_dict({k: v.to_dict() for k, v in timings.items()}, orient='index').to_csv(os.path.join(args.out_dir, 'timings.csv'),
                                                                                                index=True)


def divide_dataset_to_splits(args):
    def _inner():
        devices = torch.cuda.device_count()
        kwargs = vars(args)
        dataset = get_dataset(args.dataset, **kwargs)
        dataset.shuffle(args.seed)

        if args.include_wrong:
            # the slice (start and end) is inclusive
            splits = np.linspace(-1, min(args.num_samples-1, len(dataset)), num=args.n_processes+1).astype(int)
            for start, end in zip(splits[:-1], splits[1:]):
                yield start+1, end, devices
        else:
            model = get_model(args.model, **kwargs, n_labels=len(dataset.get_labels()), cuda_device=0)

            sample_num = -1
            starting_sample_num = 0
            valid_samples = 0
            leftover = args.num_samples % args.n_processes
            if leftover > 0:
                args.num_samples += args.n_processes - leftover
            batch_size = int(args.num_samples / args.n_processes)

            while valid_samples < args.num_samples and sample_num < len(dataset):
                sample_num += 1
                orig_sentence, label = dataset.get_sentence_to_perturb(sample_num)

                orig_sample_prob = model.predict_proba([dataset.generate_perturbed_sample(sample_num, orig_sentence)])[0]
                orig_correct = label == np.argmax(orig_sample_prob)
                if not orig_correct:  # no point in attacking samples that are already wrong
                    continue
                valid_samples += 1
                if valid_samples % batch_size == 0 or sample_num == len(dataset):
                    yield starting_sample_num, sample_num, devices
                    starting_sample_num = sample_num + 1

            del model
        del dataset
        sleep(2)

    return list((args, ssn, lsn, i % devices, lsn-ssn+1, False, i==0)  # update prog only inthe first split to avoid deadlocks
                for i, (ssn, lsn, devices) in enumerate(_inner()))


def merge_outs(outs):
    outs = sorted(outs, key=lambda out: min(out[4]))
    attack_space_size_dict = {}
    attack_positions_size_dict = {}
    timings = defaultdict(RunningCounter)
    for out in outs:
        attack_space_size_dict.update(out[9])
        attack_positions_size_dict.update(out[11])
        for k, v in out[12].items():
            timings[k].merge_(v)
        # strategy_names, strategy_index_name_map, exploit_save_mask, ready_exploit_attacks, exploit_samples_indices, exploit_labels, \
        # best_attack_probs, exploit_strategy_indices, samples_orig_losses, attack_space_size_dict, exploit_left_budgets, \
        # attack_positions_size_dict, timings = out
    strategy_names = outs[0][0]
    strategy_index_name_map = outs[0][1]
    exploit_save_mask = sum(map(op.itemgetter(2), outs), [])
    ready_exploit_attacks = sum(map(op.itemgetter(3), outs), [])
    exploit_samples_indices = sum(map(op.itemgetter(4), outs), [])
    exploit_labels = sum(map(op.itemgetter(5), outs), [])
    best_attack_probs = np.concatenate([out[6] for out in outs])
    exploit_strategy_indices = sum(map(op.itemgetter(7), outs), [])
    samples_orig_losses = sum(map(op.itemgetter(8), outs), [])
    exploit_left_budgets = sum(map(op.itemgetter(10), outs), [])

    return strategy_names, strategy_index_name_map, exploit_save_mask, ready_exploit_attacks, exploit_samples_indices, exploit_labels, \
           best_attack_probs, exploit_strategy_indices, samples_orig_losses, attack_space_size_dict, exploit_left_budgets, \
           attack_positions_size_dict, timings


def unpack_run_analysis(args):
    try:
        return run_analysis(*args)
    except Exception as e:
        import traceback
        print(f'====> Process failed with: {e}')
        print('\n'.join(traceback.format_tb(e.__traceback__)))


def main():
    args = get_args()

    if args.cap_attack_size != 0:
        if args.cap_attack_size < 0:
            set_mock_sizes(True)
        else:
            set_cap_at(args.cap_attack_size)

    if args.blacklist_orig_samples != '':
        assert args.n_processes == 1
        assert not args.include_wrong
        args.blacklist_orig_samples = {int(i) for i in args.blacklist_orig_samples.split(',')}
    else:
        args.blacklist_orig_samples = set()

    t0 = time()
    if args.n_processes == 1:
        print('Running analysis with a single process')
        dataset, *out = run_analysis(args)
        print(f'Finished creating results in {(time()-t0)/3600:.1f} hours')
    else:
        print(f'Running analysis with a {args.n_processes} processes')
        from multiprocessing import get_context
        ctx = get_context('spawn')
        with ctx.Pool(1) as executor:
            # it seems that if it's not done in a child process, we later get a deadlock in the model inits of the analyses
            params_it = executor.map(divide_dataset_to_splits, [args])[0]
        # params_it = divide_dataset_to_splits(args)
        print(f'Finished creating params for processes in {(time() - t0):.1f} seconds')
        t0 = time()

        with ctx.Pool(args.n_processes) as executor:
            # args, starting_sample_num=0, last_sample_num=np.inf, cuda_device=0, partial_proc_num=None, return_dataset=True
            outs = list(executor.map(unpack_run_analysis, params_it))
        print(f'Finished creating results in {(time() - t0) / 3600:.2f} hours')
        t1 = time()
        out = merge_outs(outs)
        dataset = get_dataset(args.dataset, **vars(args))
        dataset.shuffle(args.seed)
        print(f'Finished merging results in {(time() - t1):.2f} seconds')

    t0 = time()
    dump_results(args, dataset, *out)
    print(f'Finished dumping results in {(time() - t0):.1f} seconds')


if __name__ == '__main__':
    main()
