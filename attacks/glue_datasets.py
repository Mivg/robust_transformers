import json
import os
import re
from copy import deepcopy
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from attacks.constants import SEP
from attacks.interfaces import Dataset
from common.utils import read_glue_data


class GLUEDataset(Dataset):

    def __init__(self, glue_data_dir, task_name, label_list, train=False, dataset_suffix='', dev_set_name='dev', *__, **_):
        self._data = read_glue_data(glue_data_dir, task_name, (dev_set_name if not train else 'train') + dataset_suffix)
        self._filter_data()
        self._shuffled_data = self._data
        self._samples = {}
        self._shuffled = False
        self._label_list = label_list

    def _filter_data(self):
        pass

    def get_labels(self) -> List[str]:
        return self._label_list

    def shuffle(self, seed: int):
        if not self._shuffled:  # do it once
            np.random.seed(seed)
            self._shuffled_data = self._data.sample(frac=1., random_state=seed)
            self._shuffled = True

    def __len__(self):
        return len(self._data)

    def get_orig_row(self, i: int) -> int:
        return self._shuffled_data.iloc[i].name


class SentimentDataset(GLUEDataset):
    def __init__(self, glue_data_dir, train=False, task_=None, *__, is_tsv=True, **_):
        super().__init__(glue_data_dir, task_, ['0', '1'], train=train, *__, **_)
        self._is_tsv = is_tsv

    def get_unsplitted_sentence_to_perturb(self, i: int) -> str:
        sentnece = self._shuffled_data.iloc[i]['sentence']
        sentnece = re.sub(' +', ' ', sentnece).strip()  # remove multiple whitespace
        return sentnece

    def get_sentence_to_perturb(self, i: int) -> Tuple[List[str], int]:
        try:
            if i not in self._samples:
                sentnece, label = self._shuffled_data.iloc[i][['sentence', 'label']]
                sentnece = re.sub(' +', ' ', sentnece).strip()  # remove multiple whitespace
                self._samples[i] = (sentnece.split(' '), label)
            return self._samples[i]
        except IndexError:
            print(f'data contains only {len(self)} samples but dataset was required to return the {i}th sample in the shuffled data')
            raise

    def generate_perturbed_sample(self, i: int, perturbed_sentence: List[str]) -> List[str]:
        return perturbed_sentence

    def create_augmented_dataset(self, attacks: Dict[int, str], out_path: str):
        try:
            records = self._shuffled_data.iloc[list(attacks.keys())].copy()
        except IndexError:
            print(f'expected indices in [{0}, {len(self)-1}], but got indices in [{min(attacks.keys())}, {max(attacks.keys())}]')
            raise
        records['sentence'] = list(attacks.values())
        new_data = pd.concat((self._data.copy(), records)).reset_index(drop=True)
        print(f'Saving augmented data with {len(new_data)} n_samples compared to the original {len(self._data)}')
        if self._is_tsv:
            new_data.to_csv(out_path.split('.')[0] + '.tsv', sep='\t', index=False)
        else:
            new_data.to_csv(out_path.split('.')[0] + '.csv', index=False)


class SST2Dataset(SentimentDataset):
    def __init__(self, glue_data_dir, train=False, *__, **_):
        super().__init__(glue_data_dir, train, task_='SST-2', *__, **_)


class IMDBDataset(SentimentDataset):
    def __init__(self, glue_data_dir, train=False, *__, **_):
        super().__init__(glue_data_dir, train, task_='IMDB', dev_set_name='test', *__, is_tsv=False, **_)


class NLIDataset(GLUEDataset):
    def __init__(self, glue_data_dir, snli_pert_sent=1, train=False, task_=None, dataset_suffix='', *__, **_):
        super().__init__(glue_data_dir, task_, ['0', '1', '2'], train=train, dataset_suffix=dataset_suffix)
        assert snli_pert_sent in (1, 2)
        self._pert_sent = 'sentence1' if snli_pert_sent == 1 else 'sentence2'
        self._non_pert_sent = 'sentence2' if snli_pert_sent == 1 else 'sentence1'
        self._snli_pert_sent = snli_pert_sent
        self._label_map = {v: i for i, v in enumerate(["contradiction", "entailment", "neutral"])}
        self._label_col = 'label'

    def get_sentence_to_perturb(self, i: int) -> Tuple[List[str], int]:
        if i not in self._samples:
            sentnece, label = self._shuffled_data.iloc[i][[self._pert_sent, self._label_col]]
            sentnece = re.sub(' +', ' ', sentnece).strip()  # remove multiple whitespace
            self._samples[i] = (sentnece.split(' '), self._label_map[label])
        return self._samples[i]

    def generate_perturbed_sample(self, i: int, perturbed_sentence: List[str]) -> List[str]:
        non_pert_sent = self._shuffled_data.iloc[i][self._non_pert_sent]
        try:
            non_pert_sent = re.sub(' +', ' ', non_pert_sent).strip().split(' ')  # remove multiple whitespace
        except TypeError as e:
            print(f'Error when cleaning the non-pert-sent in mnli "{non_pert_sent}": {e}')
            raise
        return perturbed_sentence + [SEP] + non_pert_sent if self._snli_pert_sent == 1 else non_pert_sent + [SEP] + perturbed_sentence

    def create_augmented_dataset(self, attacks: Dict[int, str], out_path: str):
        records = self._shuffled_data.iloc[list(attacks.keys())].copy()
        records[self._pert_sent] = list(attacks.values())
        new_data = pd.concat((self._data.copy(), records))
        if not os.path.isdir(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        print(f'Saving augmented data with {len(new_data)} n_samples compared to the original {len(self._data)}')
        new_data.to_csv(out_path.split('.')[0] + '.tsv', sep='\t', index=False)


class SNLIDataset(NLIDataset):
    def __init__(self, glue_data_dir, snli_pert_sent=1, train=False, *__, **_):
        super().__init__(glue_data_dir, snli_pert_sent, train, task_='SNLI')


class MNLIDataset(NLIDataset):
    def __init__(self, glue_data_dir, snli_pert_sent=1, train=False, *__, **_):
        super().__init__(glue_data_dir, snli_pert_sent, train, task_='MNLI', dataset_suffix=('_matched' if not train else ''))


class SQUADDataset(Dataset):
    def __init__(self, squad_data_dir, *__, **_):
        self._data = os.path.join(squad_data_dir, 'dev.json')
        self._shuffled_data = self._data
        self._samples = {}
        self._shuffled = False
        # self._label_list = ???

    def _filter_data(self):
        pass

    def get_labels(self) -> List[str]:
        return self._label_list

    def shuffle(self, seed: int):
        if not self._shuffled:  # do it once
            np.random.seed(seed)
            self._shuffled_data = self._data.sample(frac=1., random_state=seed)
            self._shuffled = True

    def __len__(self):
        return len(self._data)

    def get_orig_row(self, i: int) -> int:
        return self._shuffled_data.iloc[i].name

    def get_sentence_to_perturb(self, i: int) -> Tuple[List[str], int]:
        pass

    def generate_perturbed_sample(self, i: int, perturbed_sentence: List[str]) -> List[str]:
        pass


class BoolQDDataset(Dataset):
    def __init__(self, glue_data_dir, train=False, *__, **_):
        with open(os.path.join(glue_data_dir, 'BoolQ', ('val' if not train else 'train') + '.jsonl'), encoding="utf-8") as lines:
            self._data = pd.DataFrame([json.loads(line) for line in lines])
        self._data['label'] = self._data['label'].apply(int)
        self._shuffled_data = self._data
        self._samples = {}
        self._shuffled = False
        self._label_list = ["0", "1"]

    def _filter_data(self):
        pass

    def get_labels(self) -> List[str]:
        return self._label_list

    def shuffle(self, seed: int):
        if not self._shuffled:  # do it once
            np.random.seed(seed)
            self._shuffled_data = self._data.sample(frac=1., random_state=seed)
            self._shuffled = True

    def __len__(self):
        return len(self._data)

    def get_orig_row(self, i: int) -> int:
        return self._shuffled_data.iloc[i].name

    def get_unsplitted_sentence_to_perturb(self, i: int) -> str:
        passage = self._shuffled_data.iloc[i]['passage']
        passage = re.sub(' +', ' ', passage).strip()  # remove multiple whitespace
        return passage

    def get_sentence_to_perturb(self, i: int) -> Tuple[List[str], int]:
        if i not in self._samples:
            passage, label = self._shuffled_data.iloc[i][['passage', 'label']]
            passage = re.sub(' +', ' ', passage).strip()  # remove multiple whitespace
            self._samples[i] = (passage.split(' '), label)
        return self._samples[i]

    def generate_perturbed_sample(self, i: int, perturbed_sentence: List[str]) -> List[str]:
        question = self.get_second_input(i).split(' ')
        return question + [SEP] + perturbed_sentence  # with the new HF, the question comes first

    def is_two_inputs_dataset(self):
        return True

    def get_second_input(self, i: int) -> str:
        question = self._shuffled_data.iloc[i]["question"]
        question = re.sub(' +', ' ', question).strip()  # remove multiple whitespace
        return question

    def create_augmented_dataset(self, attacks: Dict[int, str], out_path: str):
        records = {i: {
            'question': self._shuffled_data.iloc[sample_num]["question"],
            'passage': attack,
            'label': self._shuffled_data.iloc[sample_num]['label'],
            'idx': i
        } for i, (sample_num, attack) in enumerate(attacks.items(), len(self))}
        new_data = pd.concat((self._data.copy(), pd.DataFrame.from_dict(records, orient='index', columns=self._data.columns)))
        print(f'Saving augmented data with {len(new_data)} n_samples compared to the original {len(self._data)}')

        if not os.path.isdir(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        with open(out_path.split('.')[0] + '.jsonl', 'w', encoding="utf-8") as lines:
            new_data['label'] = new_data['label'].astype(bool)
            lines.write(new_data.to_json(orient='records', lines=True))


_dataset_names = {
    'sst2': SST2Dataset,
    'snli': SNLIDataset,
    # 'squad': SQUADDataset,
    'boolq': BoolQDDataset,
    'imdb': IMDBDataset,
    'mnli': MNLIDataset,
}

_dataset_name_to_dir = {
    'sst2': 'SST-2',
    'sst-2': 'SST-2',
    'sst': 'SST-2',
    'boolq': 'BoolQ'
}
def get_dataset_dir_name(dataset_name):
    dname = _dataset_name_to_dir.get(dataset_name.lower(), None)
    if dname is None:
        dname = dataset_name.upper()
    return dname

def get_dataset(dataset_name: str, *args, **kwargs) -> Dataset:
    mn = dataset_name.lower()
    if mn in _dataset_names:
        return _dataset_names[mn](*args, **kwargs)
    raise ValueError(f'Unknown dataset type: {dataset_name}')


def get_dataset_names():
    return list(_dataset_names.keys())


def estimate_dataset_attack_space(dataset, seed, n_samples=1000):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from attacks.synonyms_utils import load_synonym_for_attacks
    from attacks.text_fooler_filters import get_filter

    synonyms_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', 'text_fooler_synonyms.json')
    synonyms, _ = load_synonym_for_attacks(synonyms_path)

    dataset.shuffle(seed)
    n_samples = min(n_samples, len(dataset))
    n_words = []
    n_rep_words = []
    avg_n_syns = []
    n_rep_words_filter = []
    avg_n_syns_filter = []
    for i in range(n_samples):
        sent, _ = dataset.get_sentence_to_perturb(i)
        n_words.append(len(sent))
        n_rep = n_syns = 0
        attacks = []
        attacks_idx = []
        for i, w in enumerate(sent):
            if w in synonyms:
                rep = False
                for s in synonyms[w]:
                    if s != w:
                        sent_copy = deepcopy(sent)
                        sent_copy[i] = s
                        attacks.append(sent_copy)
                        attacks_idx.append(i)
                        n_syns += 1
                        rep = True
                n_rep += int(rep)
        n_rep_words.append(n_rep)
        if n_rep > 0:
            avg_n_syns.append(n_syns/n_rep)

            f = get_filter(filter_by_tf_pos=True, filter_by_sem=True, orig_sent=sent, return_mask=True, tfhub_cache_path=None,
                           sim_score_window=15, sim_score_threshold=0.7)
            mask = f(attacks, attacks_idx)
            attacks_idx = np.array(attacks_idx)[mask]
            n_rep_with_filter = len(np.unique(attacks_idx))
            n_rep_words_filter.append(n_rep_with_filter)
            if n_rep_with_filter > 0:
                avg_n_syns_filter.append(len(attacks_idx) / n_rep_with_filter)
        else:
            n_rep_words_filter.append(0)

    return {
        'n_words': f'{np.mean(n_words):.1f}',
        'rep_words': f'{np.mean(n_rep_words):.1f}',
        'n_syns': f'{np.mean(avg_n_syns):.1f}',
        'fil_rep_words': f'{np.mean(n_rep_words_filter):.1f}',
        'fil_n_syns': f'{np.mean(avg_n_syns_filter):.1f}',
    }



def estimate_all_datasets_sizes(glue_data_dir, seed=42):
    return pd.DataFrame.from_dict({dname: estimate_dataset_attack_space(get_dataset(dname, glue_data_dir, train=True), seed)
                                   for dname in get_dataset_names()})


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    os.environ['TFHUB_CACHE_DIR'] = 'media/disk1/maori/cache'
    print(estimate_all_datasets_sizes('/media/disk2/maori/data').transpose())
    # d = get_dataset(glue_data_dir='/media/disk2/maori/data', dataset_name='mnli')
    # sample = d.get_sentence_to_perturb(0)
    # pert = d.generate_perturbed_sample(0, 'this is my first attack'.split())
    # d.create_augmented_dataset({1: 'this is my first attack'}, '.')
