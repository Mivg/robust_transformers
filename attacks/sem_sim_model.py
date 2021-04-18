# based on https://github.com/jind11/TextFooler

import os
from typing import List, Callable, Union

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


tf.disable_eager_execution()  # tensorflow-hub is based on v1 of tf which doesnot support eager mode


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  # note there are version 4 and 5 already, but this is what the paper used
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1: List[str], sents2: List[str]) -> np.ndarray:
        """
        Either two list with n strings each, in which case it will compute the similarity between each respective pair and return an array
        of length n, or sent1 must be a list of length 1, in which case it will compute the similarity between the string in it to each of
        the string in sents2
        :return: since it does cosine similarity, the results are in [-1, 1] where 1 is identical and -1 very dissimilar. note that values
        such as 0.5 are still very high
        """
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores[0]


_use_model = None


def get_semantic_sim_predictor(tfhub_cache_path) -> Callable[[List[str], List[str]], np.ndarray]:
    global _use_model
    if _use_model is None:
        if tfhub_cache_path is None:
            if not os.environ.get('TFHUB_CACHE_DIR', ''):
                print('Please initialize semantic sim predictor with a valid path to tfhub cache dir')
                exit(1)
            tfhub_cache_path = os.environ['TFHUB_CACHE_DIR']
        print(f'Loading USE model (cache_dir={tfhub_cache_path})... ')
        _use_model = USE(tfhub_cache_path)
        print('Done loading USE model!')
    return _use_model.semantic_sim


if __name__ == '__main__':
    sent1 = 'Hello there! my name is Maor and I want to understand how this works but it requires a lot of words for some reason'
    sent2 = 'Hello there! my name is Maor and I desire to comprehend how this operates but it needs many words for some reason'
    sent3 = 'This sentence has no relation whatsoever to the previous two'
    sent4 = 'sfkghkdfhg kjshdf gjkhsdfkgfn aldhnfg sdnf gjlnsdlf gfnskdnfgjkn sd;fn gksdf gdg'
    use_model = USE('/media/maor/Data/data/tfhub_cache')
    print(use_model.semantic_sim([sent1], [sent1, sent2, sent3, sent4]))
