import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile, gzip, re, copy, random, math
import sys, os, shutil
import numpy
from typing import TypeVar,Iterable
from multiprocessing import Pool
from allennlp.common.elastic_logger import ElasticLogger
from subprocess import Popen,call

T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.util import add_noise_to_dict_values

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import string

def parse_filename(filename):
    results_dict = {}
    match_results = re.match('(\S+)_dev_on_(\S+)_from_(\S+)_(\S+).json', filename)
    if match_results is not None:
        results_dict['eval_set'] = match_results[1]
        results_dict['target_dataset'] = match_results[2]
        results_dict['source_dataset'] = match_results[3]
        results_dict['type'] = match_results[4]
        return results_dict

    logger.error('could not find any parsing for the format %s',filename)
    return

def process_results(args):
    # for BERTlarge we process a precdiction file ...
    if False and args.predictions_file is not None:
        instance_list = []
        with open(args.predictions_file, 'r') as f:
            for line in f:
                try:
                    instance_list.append(json.loads(line))
                except:
                    pass

        instance_list = sorted(instance_list, key=lambda x: x['question_id'])
        intances_question_id = [instance['question_id'] for instance in instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in range(len(split_inds) - 1)]
        print(len(per_question_instances))
        results_dict = {'EM':0.0, 'f1': 0.0}
        for question_instances in per_question_instances:
            best_ind = numpy.argmax([instance['best_span_logit'] for instance in question_instances])
            results_dict['EM'] += question_instances[best_ind]['EM']
            results_dict['f1'] += question_instances[best_ind]['f1']
        results_dict['EM'] /= len(per_question_instances)
        results_dict['f1'] /= len(per_question_instances)
        results_dict['EM'] *= instance_list[0]['qas_used_fraction']
        results_dict['f1'] *= instance_list[0]['qas_used_fraction']

        # sanity test:
        if args.eval_path == '-1':
            pass
        elif args.eval_path is None:
            single_file_path = cached_path('s3://multiqa/datasets/' + args.eval_set  + '_' + args.split_type + '.jsonl.zip')
            all_question_ids = []
            with zipfile.ZipFile(single_file_path, 'r') as myzip:
                if myzip.namelist()[0].find('jsonl') > 0:
                    contexts = []
                    with myzip.open(myzip.namelist()[0]) as myfile:
                        header = json.loads(myfile.readline())['header']
                        for example in myfile:
                            context = json.loads(example)
                            contexts.append(context)
                            all_question_ids += [qa['id'] for qa in context['qas']]
            predictions_question_ids = list(set(intances_question_id))
            # print(set(all_question_ids) - set(predictions_question_ids))
            results_dict['qids_missing_frac'] = len(set(all_question_ids) - set(predictions_question_ids)) / len(set(all_question_ids))

        else:
            single_file_path = cached_path(args.eval_path)
            all_question_ids = []
            contexts = []
            with gzip.open(single_file_path) as myfile:
                header = json.loads(myfile.readline())['header']
                for example in myfile:
                    context = json.loads(example)
                    contexts.append(context)
                    all_question_ids += [qa['id'] for qa in context['qas']]


            predictions_question_ids = list(set(intances_question_id))
            #print(set(all_question_ids) - set(predictions_question_ids))
            results_dict['qids_missing_frac'] = len(set(all_question_ids) - set(predictions_question_ids)) / len(set(all_question_ids))

    else:
        # computing
        with open(args.eval_res_file, 'r') as f:
            results_dict = json.load(f)

    for field in args._get_kwargs():
        results_dict[field[0]] = field[1]
    ElasticLogger().write_log('INFO', 'EvalResults', context_dict=results_dict)

    if args.predictions_file is not None:
        if args.eval_path is not None:
            # uploading to cloud
            command = "aws s3 cp " + args.predictions_file + " " + args.prediction_path + " --acl public-read"
            Popen(command, shell=True, preexec_fn=os.setsid)
        else:
            # uploading to cloud
            command = "aws s3 cp " + args.predictions_file + " " + args.prediction_path + " --acl public-read"
            Popen(command, shell=True, preexec_fn=os.setsid)

def main():
    parse = argparse.ArgumentParser("Pre-process for DocumentQA/MultiQA model and datareader")
    parse.add_argument("--eval_res_file",default=None, type=str)
    parse.add_argument("--type", default=None, type=str)
    parse.add_argument("--source_dataset", default=None, type=str)
    parse.add_argument("--target_dataset", default=None, type=str)
    parse.add_argument("--eval_set", default=None, type=str)
    parse.add_argument("--split_type", default='dev', type=str)
    parse.add_argument("--model", default=None, type=str)
    parse.add_argument("--target_size", default=None, type=str)
    parse.add_argument("--seed", default=None, type=str)
    parse.add_argument("--num_of_epochs", default=None, type=int)
    parse.add_argument("--batch_size", default=None, type=int)
    parse.add_argument("--learning_rate", default=None, type=float)
    parse.add_argument("--gas", default=None, type=int)
    parse.add_argument("--experiment", default=None, type=str)
    parse.add_argument("--full_experiments_name", default=None, type=str)
    parse.add_argument("--predictions_file", default=None, type=str)
    parse.add_argument("--prediction_path", default="s3://aigame/predictions", type=str)
    parse.add_argument("--eval_path", default=None, type=str)
    parse.add_argument("--remove_serialization_dir", default=None, type=str)
    args = parse.parse_args()


    if args.eval_res_file is not None:
        process_results(args)

        if args.remove_serialization_dir is not None:
            logger.warning("removing the following dir %s" % (args.remove_serialization_dir))
            try:
                shutil.rmtree(args.remove_serialization_dir)
            except OSError as e:
                logger.warning("Error: %s - %s." % (e.filename, e.strerror))
    else:
        logger.error('No input provided')


if __name__ == "__main__":
    main()


