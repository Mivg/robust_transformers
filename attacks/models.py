import json
import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from attacks.constants import SEP
from attacks.interfaces import Model


class MockModel(Model):
    def __init__(self, n_labels, *__, **_):
        self.n_labels = n_labels

    def predict_proba(self, samples: List[List[str]]) -> np.array:
        # TODO - this is a mock of course
        ps = []
        for sample in samples:
            np.random.seed(abs(hash('$'.join(sample))) % 4294967295)  # make it deterministic (Seed must be between 0 and 2**32 - 1)
            ps.append(np.random.random(size=self.n_labels))
        ps = np.array(ps)
        return ps / ps.sum(axis=1)


# def preprocess_function(samples: List[List[str]], tokenizer, max_seq_len, samples_text_b: Optional[List[List[str]]] = None):
#     # Tokenize the texts
#     text_a = [' '.join(s) for s in samples]
#     if samples_text_b is None:
#         result = tokenizer(text_a, padding='max_length', max_length=max_seq_len, truncation=True)
#     else:
#         text_b = [' '.join(s) for s in samples_text_b]
#         result = tokenizer(text_a, text_b, padding='max_length', max_length=max_seq_len, truncation=True)
#
#     # Map labels to IDs (not necessary for GLUE tasks)
#     if label_to_id is not None and "label" in examples:
#         result["label"] = [label_to_id[l] for l in examples["label"]]
#     return result


class TransformerGLUEModel(Model):
    _guids = defaultdict(list)

    def __init__(self, model_path, label_list, cache_dir=None, eval_batch_size=None, cuda_device=0, *__, **_):
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        num_labels = len(label_list)
        use_fast = True

        config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, finetuning_task=None, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=use_fast)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path), config=config,
                                                                        cache_dir=cache_dir)
        with open(os.path.join(model_path, 'data_args.json'), 'r') as f:
            data_args_dict = json.load(f)

        training_args = torch.load(os.path.join(model_path, 'training_args.bin'))

        self.max_seq_length = data_args_dict['max_seq_length']
        self.device = f'cuda:{cuda_device}'
        self.model = self.model.to(self.device).eval()

        self.label_list = label_list
        if eval_batch_size is None:
            eval_batch_size = training_args.per_device_eval_batch_size or training_args.per_gpu_eval_batch_size
        self._eval_batch_size = eval_batch_size

    @classmethod
    def prep_samples_to_features(cls, samples: List[List[str]], tokenizer, max_seq_len,
                                 samples_text_b: Optional[List[List[str]]] = None):
        # Tokenize the texts
        sample_text = samples[0]
        text_b = None
        if SEP in sample_text:
            assert samples_text_b is None
            text_a, text_b = [], []
            for s in samples:
                sep_index = s.index(SEP)
                text_a.append(' '.join(s[:sep_index]))
                text_b.append(' '.join(s[sep_index + 1:]))
        else:
            text_a = [' '.join(s) for s in samples]
            if samples_text_b is not None:
                text_b = [' '.join(s) if s is not None else s for s in samples_text_b]
        args = (text_a, ) if text_b is None else (text_a, text_b)
        # TODO - see if we can avoid the joining above
        return tokenizer(*args, padding='max_length', max_length=max_seq_len, truncation=True)

    def predict_proba(self, samples: List[List[str]]) -> np.array:
        features = self.prep_samples_to_features(samples, self.tokenizer, max_seq_len=self.max_seq_length)

        # TOdo - moving to cuda must be later on to avoid memory error here if given large predictions. Need dataloader anyway
        # Convert to Tensors and build dataset
        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
        # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device)

        all_input_ids = torch.tensor(features['input_ids'], dtype=torch.long).to(self.device)
        all_attention_mask = torch.tensor(features['attention_mask'], dtype=torch.long).to(self.device)
        all_token_type_ids = torch.tensor(features['token_type_ids'], dtype=torch.long).to(self.device) if 'token_type_ids' in features else None

        with torch.no_grad():
            edges = np.arange(0, int(len(samples) / self._eval_batch_size) * self._eval_batch_size + 1, self._eval_batch_size).tolist()
            probs = []
            if edges[-1] != len(samples):
                edges.append(len(samples))
            for start, end in zip(edges[:-1], edges[1:]):
                inputs = {'input_ids': all_input_ids[start:end],
                          'attention_mask': all_attention_mask[start:end]}
                # # XLM, DistilBERT and RoBERTa don't use segment_ids
                if all_token_type_ids is not None:
                    inputs['token_type_ids'] = all_token_type_ids[start:end]

                # model outputs are always tuple in transformers (see doc) but here we will get only logits
                logits = self.model(**inputs)[0]
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
            return np.concatenate(probs)


class TransformerSentimentModel(TransformerGLUEModel):
    def __init__(self, model_path, cache_dir=None, *__, **kwargs):
        super().__init__(model_path, ['0', '1'], cache_dir, *__, **kwargs)


class TransformerNLIModel(TransformerGLUEModel):
    def __init__(self, model_path, cache_dir=None, *__, **kwargs):
        super().__init__(model_path, ['0', '1', '2'], cache_dir, *__, **kwargs)


class TransformerBoolQModel(TransformerGLUEModel):
    def __init__(self, model_path, cache_dir=None, *__, **_):
        super().__init__(model_path, ['0', '1'], cache_dir, *__, **_)


_model_names = {
    'mock': MockModel,
    'transformer_sst2': TransformerSentimentModel,
    'transformer_snli': TransformerNLIModel,
    'transformer_mnli': TransformerNLIModel,
    'transformer_boolq': TransformerBoolQModel,
    'transformer_imdb': TransformerSentimentModel,
}


def get_model(model_name: str, *args, **kwargs) -> Model:
    mn = model_name.lower()
    if mn in _model_names:
        return _model_names[mn](*args, **kwargs)
    raise ValueError(f'Unknown model type: {model_name}')


def get_model_names():
    return list(_model_names.keys())
