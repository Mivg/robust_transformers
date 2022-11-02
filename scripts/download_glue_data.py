# from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e#file-download_glue_data-py
''' Script for downloading all GLUE data.
Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized,
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).
mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
1/30/19: It looks like SentEval is no longer hosting their extracted and tokenized MRPC data, so you'll need to download the data from the original source for now.
2/11/19: It looks like SentEval actually *is* hosting the extracted data. Hooray!
'''
import json
import os
import re
import sys
import shutil
import argparse
import tarfile
import tempfile
import urllib.request
import zipfile
import pandas as pd

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "RTE", "WNLI", "diagnostic", 'BoolQ', 'IMDB']

# updated download pathes to glue data from https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py
TASK2PATH = {"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
             "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
             "QQP":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
             "STS":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
             "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
             "QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
             "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
             "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
             "diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv',
             "IMDB":"http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
             'BoolQ': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip',
             }

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    url_path = TASK2PATH[task]
    is_targz = url_path.endswith('tar.gz')
    data_file = "%s.tar.gz" % task if is_targz else "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    print('finished downloading the file. Start extracting')
    if is_targz:
        with tarfile.open(data_file, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=os.path.join(data_dir,task))
    else:
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def normalize_sentence(sent):
    # remove multiple/tailing/trailing white space
    sent_ = re.sub('\s+', ' ', sent).strip()  # also takes care of other types of white space such as '\xa0' which is a non-breaking space
    # correct the location of punctuation marks (remove trailing white space and add tailing whitespace). Note - in BoolQ this cause one
    # issue with '... extension of .jpg' but it is still worth it
    sent_ = re.sub(' ([,\?\.!])', '\g<1>', sent_)
    sent_ = re.sub('(\w+) ([\.,])([a-zA-Z])', '\g<1>\g<2> \g<3>', sent_)
    # if sent_ != sent:
    #     print(sent_)
    #     print(sent)
    return sent_


def pre_process_imdb_data(glue_data_dir):
    base_data_path = os.path.join(glue_data_dir, 'IMDB', 'aclImdb')
    for dataset in ('train', 'test'):
        print(f'Start working on the {dataset} set... ', end='')
        records = []
        for label, label_dir in enumerate(('neg', 'pos')):
            print(f'starting {label_dir}... ', end=' ')
            samples_dir = os.path.join(base_data_path, dataset, label_dir)
            for fname in os.listdir(samples_dir):
                with open(os.path.join(samples_dir, fname)) as f:
                    records.append({
                        'idx': os.path.splitext(fname)[0],
                        'sentence': normalize_sentence(f.read()),
                        'label': label,

                    })
        print('Done! now dumping...')
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(glue_data_dir, 'IMDB', dataset + '.csv'), index=False)


def pre_process_boolq_data(glue_data_dir):
    base_data_path = os.path.join(glue_data_dir, 'BoolQ')
    for dataset in ('train', 'val'):
        with open(f'{base_data_path}/{dataset}.jsonl', encoding="utf-8") as lines:
            data = [json.loads(line) for line in lines]
        for d in data:
            d['passage'] = normalize_sentence(d['passage'])
        with open(f'{base_data_path}/{dataset}.jsonl', 'w', encoding="utf-8") as lines:
            lines.writelines(json.dumps(d) + '\n' for d in data)


def pre_process_sst_data(glue_data_dir):
    for dataset in ('train', 'dev'):
        file_path = os.path.join(glue_data_dir, 'SST-2', dataset + '.tsv')
        data = pd.read_csv(file_path, sep='\t')
        data.sentence = data.sentence.apply(normalize_sentence)
        data.to_csv(file_path, sep='\t', index=False)


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        print("Local MRPC data not specified, downloading data from %s" % MRPC_TRAIN)
        mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
        urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
        urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
    urllib.request.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
         open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding="utf8") as train_fh, \
         open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, \
            open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding="utf8") as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks


pre_processors = {
    'IMDB': pre_process_imdb_data,
    'BoolQ': pre_process_boolq_data,
    'SST': pre_process_sst_data,
}


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)
        if task in pre_processors:
            print('Pre-processing', task)
            pre_processors[task](args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))