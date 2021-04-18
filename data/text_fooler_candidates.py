# based on code from https://github.com/jind11/TextFooler
import json

import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd  # pip install googledrivedownloader
import argparse
from tqdm import tqdm

from data.constants import stop_words


def download_counterfits_embeddings(embedding_path):
    # https://stackoverflow.com/a/47761459
    print('downloading embeddings file')
    gdd.download_file_from_google_drive(file_id='1bayGomljWb6HeYDMTDKXrh0HackKtSlx', dest_path=embedding_path, unzip=True)
    print('done')


def _compute_sim_mat_and_vocab(embedding_path):
    embeddings = []
    words = []
    print('loading embedding file')
    with open(embedding_path, 'r') as ifile:
        for line in ifile:
            line = line.strip().split()
            words.append(line[0])
            embedding = [float(num) for num in line[1:]]
            embeddings.append(embedding)
    print('computing similarity')
    embeddings = np.array(embeddings)
    print(embeddings.T.shape)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float32")
    product = np.dot(embeddings, embeddings.T)
    # np.save(out_path, product)
    return product, words


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count, threshold):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    # this is the original code fro mtext fooler, but it so unefficient i have to modify it
    # why the hell argsort and not argpartition??? and why negating the entire array every time instead of simply taking it in reverse??
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[i] for i in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def _create_synonyms_lists_per_index(cos_sim_mat, n_synonyms, min_sim, words):
    # since we don't really care for order now, can use argpartition which is much more efficient
    print('finding most similar synonyms')
    k = n_synonyms+1
    # due to issues with memory, have to do it in batches
    batch_size = 10000
    count = 0
    syns_dict = {}
    pbar = tqdm(total=len(cos_sim_mat))
    while cos_sim_mat.shape[0] > 0:
        # argpatition over axis 1 means each row will contain in its first k columns the indices of most similar words
        inds = np.argpartition(-cos_sim_mat[:batch_size, :], axis=1, kth=k)[:, :k]  # +1 because a word will also be identical to itself
        actual_batch_size = min(batch_size, len(cos_sim_mat))
        for src_idx in range(actual_batch_size):
            real_src_idx = count+src_idx
            if words[real_src_idx] in stop_words:
                continue  # they didn't replace 'stopwords' but they were okay with replacing to a stopword
            synonyms = [words[tgt_idx]
                        for tgt_idx in inds[src_idx, cos_sim_mat[src_idx][inds[src_idx]] >= min_sim] if tgt_idx != real_src_idx]
            if len(synonyms) > 0:
                syns_dict[words[real_src_idx]] = synonyms
        count += batch_size
        cos_sim_mat = cos_sim_mat[actual_batch_size:]
        del inds
        pbar.update(actual_batch_size)
    return syns_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_synonyms', type=int, default=50)
    parser.add_argument('--min_sim', type=float, default=0.5)  # they actually hardcoded that part
    args = parser.parse_args()
    embedding_path = '/tmp/counterfeits.txt'
    extracted_embedding_path = '/tmp/counter-fitted-vectors.txt'
    download_counterfits_embeddings(embedding_path)
    cos_sim_mat, words = _compute_sim_mat_and_vocab(extracted_embedding_path)
    syns_dict = _create_synonyms_lists_per_index(cos_sim_mat, args.n_synonyms, args.min_sim, words)
    print('dumping json file')
    with open('text_fooler_synonyms.json', 'w') as f:
        json.dump(syns_dict, f)


if __name__ == '__main__':
    # _create_synonyms_lists_per_index(np.load('/media/maor/Performance/Data/text_fooler/cos_sim_counter_fitting.npy'), 50, 0.5, [str(i) for i in range(65713)])
    main()
