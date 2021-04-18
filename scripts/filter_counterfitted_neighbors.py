import json
from tqdm import tqdm
from transformers import BertTokenizer

if __name__ == '__main__':

    tokenzier = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True, cache_dir=None)
    tokenizer_vocab = set(tokenzier.vocab)

    NEIGHBORS_FILE = '../data/counterfitted_neighbors.json'
    FILTERED_NEIGHBORS_FILE = '../data/filtered_counterfitted_neighbors.json'
    with open(NEIGHBORS_FILE) as f:
        neighbors = json.load(f)

    filtered_neighbors = {key: [] if key not in tokenizer_vocab else [word for word in words if word in tokenizer_vocab]
                          for key, words in tqdm(neighbors.items())}
    with open(FILTERED_NEIGHBORS_FILE, 'w') as f:
        json.dump(filtered_neighbors, f)

