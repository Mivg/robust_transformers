# RobustTransformers

This repository contains all code needed to train and evaluate the robustness of NLP transformers-based models on GLUE (and some other) 
datasets for the [Achieving Model Robustness through Discrete Adversarial Training](https://arxiv.org/abs/2104.05062) paper. It is based on HuggingFace and pytorch.
It also contains some auxiliary scripts and code from [TextFooler by Jin et. al](https://github.com/jind11/TextFooler)

**Updated version of the code base, with additional scripts needed to run the experiments as given in the final version that was presented in EMNLP 2021 will be uploaded here shortly** 


## Preparing the environment
To run the experiments you would need to create a virtual environment and download the data. You will also need access to GPUs

### Environment
To create an environment as used here, you should do the following (assuming use of conda environments and not virtualenv)

```bash
conda create -n robust python=3.7
source activate robust
pip install dash dash-bootstrap-components dash-table netifaces scikit-learn threadpoolctl spacy nltk pyyaml pandas attrs xlsxwriter==1.3.7
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html  # alternative for cuda 10.1: pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install psutil recordclass GPUtil sqlitedict boto3 sacremoses sentencepiece packaging filelock
pip install tensorboard==2.0.0 tensorflow==2.0.0 tensorflow-estimator==2.0.0 tensorflow_hub datasets==1.0.0
pip3 install --upgrade tensorflow-gpu  # for Python 3.n and GPU

pip install transformers==4.0.1

python3 -m spacy download en_core_web_sm
```

### Data
To download the datasets, run the following script:
```bash
python scripts/download_glue_data.py --tasks IMDB,BoolQ,SST --data_dir [path_to_target_dir]
```

To download the prepared synonyms dict, download this [file](https://drive.google.com/file/d/1Bp0xGybTLI24JDn4y3I7g6JGSAc7hZP6/view?usp=sharing) and then extract it under [data](data) (this will create three directories 
under [data](data), one for each dataset)

## Structure
Below is a very high level description of the structure of the repo. All relevant entry-point modules are listed here, and their 
dependencies many contain other modules of interest.

### Interfaces
The interfaces which are implemented and used throughout the project can be found in [attacks/interfaces.py](attacks/interfaces.py)
It contains the following:
- AttackStrategy: definition of an attack strategy and everything it should include (agnostic to dataset/model)
- Model: Definition of a model and the predict function it must implement (in evaluation only! training does not use this interface)
- Dataset: Definition of a dataset which can be used (in evaluation only! training does not use this interface)
 
### Datasets
The actual implementation of the different supported datasets can be found in [attacks/glue_datasets.py](attacks/glue_datasets.py).
For now, what you should care about is *SST2Dataset*, *BoolQDDataset* and *IMDBDataset*.
The datasets are created using the factory function `attacks/glue_datasets.get_dataset(...)` which get the dataset name 
(the key options can be found with `attacks/glue_datasets.get_dataset_names` while the args are specific to each dataset)

Dataset produce the input that should be perturbed, and upon getting a suggested attack, create the final input as should be consumed by 
the model to make the prediction

### Models
The only relevant model at the moment is TransformerGLUEModel which gets a cached model saved by HF and wraps it to implement the interface.
The models are implemented in [attacks/models.py](attacks/models.py).
There are few specific implementations for models for different datasets. Those can be found with `attacks/models.get_model_names()`
and you load a model with the factory function `attacks/models.get_model(...)`

### Attack Strategies
Strategies are implemented in [attacks/strategies.py](attacks/strategies.py).
The most relevant strategies are *RandomSynonymSwapAttackStrategy*, *GreedySynonymSwapAttackStrategy*, 
*InformedGreedySynonymSwapAttackStrategy* and *OptimisticGreedySynonymSwapAttackStrategy*. Note that TextFooler is a specific instance of 
*InformedGreedySynonymSwapAttackStrategy*.
The attack strategies contain several types of arguments. They include definitions of the attack-space (Future versions will have the 
attack space as an interface of its own and normalize the use of it), definition of how to attack and what to use (e.g. beam size) and 
finally, evaluation specific parameters such as budget and exploration batch size

The lifecycle of an attack strategy is as follows:
- Initialized with the global parameters, loads attack space and init all variables
- init_attack with the original input and label, pre-computes everything needed to start exploring
- Multiple steps of exploration (generate a new attack to predict on) and updates (use the new information to decide what to do next)
- When the budget ends (or the attack is satisfied with its attack), exploitation is triggered which retrieves the best possible attack it 
can based on its current information. Note that exploitation can used all the time, regardless the stage of exploration the attack is in.

### Evaluating
To evaluate a model's robustness, we use the [attacks/analyze_robustness.py](attacks/analyze_robustness.py) script. It gets many different 
parameters as can be described with `python attacks/analyze_robustness.py -h`
However, the most important ones are those that detail the dataset and model to be used, and the number of samples and budget for the 
attack. 
To detail the attacks strategies to use, we give it a path to a yaml file with the different strategies. Examples for such yaml files 
can be found in [strategies_configs](strategies_configs), where the specific yaml we use to define the ensemble with we which we evaluate 
robust accuracy is [strategies_configs/rob_eval.yaml](strategies_configs/rob_eval.yaml)


### Training
To train a model, we have the [hf_transformers/dat_glue.py](hf_transformers/dat_glue.py) script. It is based on HF run_glue.py example 
(which can be found in a slightly modified version in [hf_transformers/run_glue.py](hf_transformers/run_glue.py) if you wish to run 
sanity trainings). As in HF, it has many parameters it expects. Those include *ModelArguments* and *DataTrainingArguments* which are 
defined in [hf_transformers/dat_glue.py](hf_transformers/dat_glue.py), *TrainingArguments* from 
[hf_transformers/training_args.py](hf_transformers/training_args.py) and *AdversarialTrainingArguments* from 
[hf_transformers/adv_training_args.py](hf_transformers/adv_training_args.py).
The basic usage is just as in 
[HF run_glue example](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py), 
though the original trainer defined by HF is replaced by
[hf_transformers/adversarial_trainer.py](hf_transformers/adversarial_trainer.py) to also use support online adversarial/random training.

The only main difference (aside from adversarial training) is that we support non-glue datasets in the training, and thus using task_name 
is not advised, and instead, using **--train_file** and **--validation_file** is the correct usage

## Attack Spaces
To define and create attack spaces, we need to recall the many definitions we had for it over time e.g. simple dict-based word substitution, 
pre-computed POS tags with online POS evaluation (done globally on init attack), Filtering based on multiple possible filters 
(e.g. USE, NLTK POS tags) after every word substitution and finally $S_\phi'$ which uses many filters and caches the results. To add to it, 
we also had to account for difference in spacing, capitalization adn punctuations. All of this create something like geological layers of 
mess on what started of as tidy and well designed project. The many utils used to define and filter the attack space are referenced in the
next section and are pending a big refactor.

Still, the base candidates files are the one from Jia and Liang port of Alzantot's attack space which is in 
[data/counterfitted_neighbors.json](data/counterfitted_neighbors.json) and the one from TF that can be found here 
[data/text_fooler_synonyms.json](data/text_fooler_synonyms.json).
The creation of the TF synonyms is in the [data/text_fooler_candidates.py](data/text_fooler_candidates.py) which is based on TF 
implementation with ours USE and POS code (all based on theirs but had to be slightly modified due to mismatch in keras and pytorch 
versions)

The preparation of the cached attack space ($S_\phi'$) which we now use is done in 
[scripts/prepare_attack_space.py](scripts/prepare_attack_space.py) with 
`perform_spacy_pos_filtering`, `perform_gp2_ppl_filtering`, `perform_use_semantic_filtering` flags all active (with the default arguments 
for windows and thresholds)

## Training robust models
Aside from training a model with online methods (adversarial or random) which is done with 
[hf_transformers/dat_glue.py](hf_transformers/dat_glue.py)  as described above by supplying `--strategy_name`, `--strategy_params` and 
`--orig_lambda` (and potentially `--async_adv_batches`, all are described with `python hf_transformers/dat_glue.py -h`), we can also do 
offline random/adversarial augmentation. Those scripts (i.e. [scripts/random_augmentations.py](scripts/random_augmentations.py) and 
[scripts/offline_augmentation.py](scripts/offline_augmentation.py) respectively) create a new instance of the training set with the 
augmented results. Those will be passed to [hf_transformers/dat_glue.py](hf_transformers/dat_glue.py) with the `--train_file` argument to 
train on, as if it is the original dataset



## Complete mess that needs refactoring
There are multiple files which contain all types of utils and helpers that are used throughout the project. Unfortunately, due to 
historical reasons they are currently spread across multiple files, and lack in documentation. Future versions will be refactored to have 
better structure, and will include more documentation and inline comments. In the meantime, relevant util files are:
- [common/utils.py](common/utils.py): The main util files, with utils to pertrub sentences (the main one used everywhere is 
    `get_possible_perturbations`), clean and prepare sentences and compute size of attack spaces. It also contains util functions to read 
    glue data and to get the local ip. It contains the `get_spacy_pos` and `get_perplexity` functions used to created the 
    pre-computed filtered attack space $S_\phi'$
- [attacks/sem_sim_model.py](attacks/sem_sim_model.py): Defines and implement the USE semantic similarity function, modified from TF
- [attacks/synonyms_utils.py](attacks/synonyms_utils.py): This file contain utils to compute and store synonyms. In particular, 
    the function `load_synonym_for_attacks` which is the main entry point to get the global attack-space is there (Where the 
    `CachedSynonyms` class which implements $S_\phi'$ is ) is defined there
- [attacks/text_fooler_filters.py](attacks/text_fooler_filters.py): contains the definitions and implementations of attacks filters (based 
    on POS, Perplexity and Semantic-similarity)
- [hf_transformers/adv_utils.py](hf_transformers/adv_utils.py): This file contains utils used by online adversarial/random training to get 
    adversarial attacks out of encoded samples w.r.t. a current checkpoint of the loaded model
    

## Cite 
This repository contains a Tensorflow implementation for our [preprint paper](https://arxiv.org/abs/2104.05062).  
If you find this code useful in your research, please consider citing:

    @article{ivgi2021achieving,
      title={Achieving Model Robustness through Discrete Adversarial Training},
      author={Ivgi, Maor and Berant, Jonathan},
      journal={arXiv preprint arXiv:2104.05062},
      year={2021}
    }