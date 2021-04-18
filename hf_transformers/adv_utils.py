import random
from collections import namedtuple
from time import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from attacks.analyze_robustness import RunningCounter
from attacks.interfaces import SampleResult
from attacks.models import TransformerGLUEModel
from attacks.strategies import get_strategy
from attacks.synonyms_utils import set_mock_sizes
import numpy as np


Attacker = namedtuple('Attacker', 'features_func strategy_copies')
def get_attacker(max_seq_len, device, train_batch_size, adv_args, tokenizer):
    set_mock_sizes(True)  # it wasted a lot of time to compute the actual attack space size
    def samples_to_inputs(sample_a, samples_b, labels):
        if not isinstance(sample_a[0], list):
            sample_a = [sa.split() for sa in sample_a]
        if samples_b is not None and not isinstance(samples_b[0], list):
            if samples_b[0] is None:
                samples_b = None  # assuming if one of them is None, they all are
            else:
                samples_b = [sb.split() for sb in samples_b]
        features = TransformerGLUEModel.prep_samples_to_features(sample_a, tokenizer, max_seq_len, samples_b)
        batch = {k: torch.tensor(features[k]) for k in features}
        batch['labels'] = torch.tensor(labels)
        return batch
        # return batch_to_inputs(batch, args)
    try:
        n_copies = train_batch_size
        # n_copies = args.per_gpu_train_batch_size
        if adv_args.async_adv_batches:
            n_copies *= adv_args.n_adv_loaders
        strategy_copies = [get_strategy(adv_args.strategy_name, **adv_args.strategy_params, dataset=adv_args.syn_task, train=True) for
                           _ in range(n_copies)]
    except TypeError as e:
        raise ValueError(f'Wrong parameters given to strategy of type {adv_args.strategy_name}: {e}') from e
    return Attacker(samples_to_inputs, strategy_copies)


def prep_text(encoded_text, tokenizer):
    text = tokenizer.decode(encoded_text)
    end_ind = text.find(f'{tokenizer.pad_token}')
    if end_ind != -1:
        text = text[:end_ind].strip()
    text = text[len(tokenizer.cls_token):-len(tokenizer.sep_token)]  # remove the trailing "['CLS'] " and tailing " [SEP]"
    texts = [t.strip() for t in text.split(f'{tokenizer.sep_token}') if len(t.strip()) > 0]  # will return a list of length 1 or 2, depends on the task
    if len(texts) == 1:
        texts.append(None)
    return texts


def batch_to_inputs(batch, args):
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
    if args.model_type != 'distilbert':
        inputs['token_type_ids'] = batch[2] if args.model_type in \
                                               ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
    return inputs


def get_adversarial_inputs(adv_args, batch, model, tokenizer, attacker, per_device_eval_batch_size, prep_inputs_func,
                           timings: Optional[Dict[str, RunningCounter]] = None):
    t0 = time()

    def update_timings(key, **extras):
        if timings is None:
            return
        nonlocal t0
        timings[key].update(time()-t0, **extras)
        t0 = time()

    labels = batch['labels'].cpu().numpy()
    texts = [prep_text(b.tolist(), tokenizer) for b in batch['input_ids']]  # list of 2-tuples with (text_a, text_b) for each input
    texts_a, texts_b = list(zip(*texts))
    update_timings('prep_texts')

    if texts_b[0] is None:
        if adv_args.pert_text in ['b', 'random']:
            raise ValueError('Cannot use pert_text b/random when there is only one input text in the task!')
        adv_args.pert_text = 'a'

    text_to_pert = adv_args.pert_text if adv_args.pert_text != 'random' else random.choice(['a', 'b'])

    if text_to_pert == 'a':
        texts_b = np.array(texts_b)
        _to_inputs = lambda attacks, had_attacks: attacker.features_func(attacks, texts_b[had_attacks].tolist(), labels[had_attacks])
        pert_texts = texts_a
    else:
        texts_a = np.array(texts_a)
        _to_inputs = lambda attacks, had_attacks: attacker.features_func(texts_a[had_attacks].tolist(), attacks, labels[had_attacks])
        pert_texts = texts_b
    to_inputs = lambda attacks, had_attacks: prep_inputs_func(_to_inputs(attacks, had_attacks))

    for strategy, text, label in zip(attacker.strategy_copies, pert_texts, labels):
        strategy.init_attack(text.split(), label, adv_args.num_attacks, replace_orig_if_needed=True)
    update_timings('init_strategies', n_strategies=len(attacker.strategy_copies))

    n_strategies = len(attacker.strategy_copies)
    exploration_budgets = {sn: adv_args.num_attacks for sn in range(n_strategies)}
    done_strategies = 0
    for i in range(adv_args.num_attacks):
        # gather exploration samples
        attacks = []
        had_attacks = []
        edges = [0]
        for strategy_num, (strategy, _) in enumerate(zip(attacker.strategy_copies, labels)):  # go over labels in case the batch is not full
            if done_strategies == len(exploration_budgets):
                break  # everyone used up their budgets
            explore_batch_size = min(exploration_budgets[strategy_num], per_device_eval_batch_size)
            edges.append(edges[-1])
            if explore_batch_size == 0:
                continue
            new_attack = strategy.explore(explore_batch_size)
            actual_n_attacks = len(new_attack)
            if actual_n_attacks > 0:
                edges[-1] = edges[-1] + actual_n_attacks
                attacks.extend(new_attack)
                had_attacks.extend([strategy_num]*actual_n_attacks)
                exploration_budgets[strategy_num] -= actual_n_attacks
                if exploration_budgets[strategy_num] == 0:
                    done_strategies += 1
            update_timings('explored_strategy', n_new_attacks=actual_n_attacks)
        if len(attacks) == 0:
            break  # no more attacks left
        inputs = to_inputs(attacks, had_attacks)
        update_timings('prep_explored_inputs', n_attacks=len(attacks))

        # do we actually get here a the data parallel meaning we should use per_device * n_gpus?
        if len(had_attacks) <= per_device_eval_batch_size:
            outputs = model(**inputs)
            logits = outputs[1]
        else:
            logits = torch.cat([model(**mini_batch)[1] for mini_batch, _ in split_batch(inputs, per_device_eval_batch_size)])
        update_timings('predict_explored_inputs', n_attacks=len(attacks))

        probs = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probs, axis=1)
        assert edges[-1] == len(had_attacks)
        for strategy_num, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            if end == start:
                continue
            strategy = attacker.strategy_copies[strategy_num]
            label = labels[strategy_num]
            # again, we want to minimize the label prob
            strategy.update(list(map(lambda res: SampleResult(*res),
                                     zip(attacks[start:end], probs[start:end, label], predicted_labels[start:end] != label))))
            update_timings('updated_strategy')

    # now exploit (generate final attack)
    attacks = []
    had_attacks = []  # some samples may not have any attacks possible
    for strategy_num, (strategy, _) in enumerate(zip(attacker.strategy_copies, labels)):
        new_attack = strategy.exploit(adv_args.n_exploits)
        if len(new_attack) > 0:
            attacks.extend(new_attack)
            had_attacks.append(strategy_num)
        update_timings('exploited_strategy')
    if len(attacks) == 0:
        return None

    return to_inputs(attacks, had_attacks)


def split_batch(adv_inputs: dict, train_batch_size):
    keys = list(adv_inputs.keys())
    n_samples = adv_inputs[keys[0]].size(0)
    if n_samples <= train_batch_size:
        yield adv_inputs, 1.
    else:
        start, end = 0, train_batch_size
        while start < end:  # in the last iteration, start=end=n_samples
            yield {k: v[start:end] if v is not None else None for k, v in adv_inputs.items()}, (end - start) / n_samples
            start = end
            end = min(end+train_batch_size, n_samples)
