# coding: utf-8

# In[1]:


import json
import pika
import datetime
import time
import boto3
import copy
import os
import sys
from allennlp.common.file_utils import cached_path
from allennlp.common.elastic_logger import ElasticLogger
import pandas as pd

# In[2]:

class runners_control():
    def __init__(self):

    def get_s3_experiments(prefix):
        s3 = boto3.client("s3")
        all_objects = s3.list_objects(Bucket='multiqa', Prefix=prefix)
        all_keys = []
        if 'Contents' in all_objects:
            for obj in all_objects['Contents']:
                if obj['Key'].find('.tar.gz') > -1:
                    all_keys.append(obj['Key'])
        return all_keys

    def get_running_experiments():
        query = {
            "from": 0, "size": 100,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"message": "Job"}}
                    ]
                }
            },
            "sort": [
                {
                    "log_timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }
        running_exp = []
        res = ElasticLogger().es.search(index="multiqa_logs", body=query)
        curr_time = datetime.datetime.utcnow()
        for exp in res['hits']['hits']:
            exp_time = datetime.datetime.strptime(exp['_source']['log_timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
            if curr_time - exp_time < datetime.timedelta(0, 180):
                running_exp.append(exp['_source']['experiment_name'])
        return list(set(running_exp))

    def allennlp_include_packages():
        return ' --include-package allennlp.models.reading_comprehension.docqa++                --include-package allennlp.data.iterators.multiqa_iterator                --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+                --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+combine                --include-package allennlp.models.reading_comprehension.docqa++BERT                --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa_bert'

    def add_override_config(config_name):
        config_path = '/Users/alontalmor/Dropbox/Backup/QAResearch/MultiQA/experiment_configs/'
        with open(config_path + config_name, 'r') as f:
            return json.load(f)

    def add_model_dir(host):
        if host in ['savant', 'rack-gamir-g03', 'rack-gamir-g04', 'rack-gamir-g05']:
            return '/home/joberant/home/alontalmor/models/'
        else:
            return '../models/'

    def send_to_queue(name, queue, config):
        connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        config['retry'] = 0
        channel.basic_publish(exchange='',
                              properties=pika.BasicProperties(
                                  headers={'name': name}),
                              routing_key=queue,
                              body=json.dumps(config))
        connection.close()

    def dataset_specific_override(dataset, run_config):
        if dataset in run_config['dataset_specific_override']:
            for key1 in run_config['dataset_specific_override'][dataset].keys():
                for key2 in run_config['dataset_specific_override'][dataset][key1].keys():
                    if type(run_config['dataset_specific_override'][dataset][key1][key2]) == dict:
                        for key3 in run_config['dataset_specific_override'][dataset][key1][key2].keys():
                            run_config['override_config'][key1][key2][key3] = run_config['dataset_specific_override'][dataset][key1][key2][key3]
                    else:
                        run_config['override_config'][key1][key2] = run_config['dataset_specific_override'][dataset][key1][key2]

        return run_config

    def replace_tags(dataset, run_config, source_dataset=None):
        run_config['override_config']["train_data_path"] = run_config['override_config']["train_data_path"].replace('[DATASET]', dataset)
        run_config['override_config']["validation_data_path"] = run_config['override_config']["validation_data_path"].replace('[DATASET]',
                                                                                                                              dataset)

        run_config['output_file'] = run_config['output_file'].replace('[RUN_NAME]', run_name)
        run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[EXP_NAME]', experiment_name)
        run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[DATASET]', dataset)

        if 'source_model_path' in run_config:
            run_config['source_model_path'] = run_config['source_model_path'].replace('[SOURCE]', source_dataset)
            run_config['output_file'] = run_config['output_file'].replace('[SOURCE]', source_dataset)
            run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[SOURCE]', source_dataset)

        return run_config

    def replace_tag_params(run_config, params):
        for key in params.keys():
            if "train_data_path" in run_config['override_config']:
                run_config['override_config']["train_data_path"] = run_config['override_config']["train_data_path"].replace('[' + key + ']',
                                                                                                                            str(params[key]))
            if "validation_data_path" in run_config['override_config']:
                run_config['override_config']["validation_data_path"] = run_config['override_config']["validation_data_path"].replace(
                    '[' + key + ']', str(params[key]))

            if "post_proc_bash" in run_config:
                run_config["post_proc_bash"] = run_config["post_proc_bash"].replace('[' + key + ']', str(params[key]))

            if "model" in run_config:
                run_config["model"] = run_config["model"].replace('[' + key + ']', str(params[key]))
            if "eval_set" in run_config:
                run_config["eval_set"] = run_config["eval_set"].replace('[' + key + ']', str(params[key]))

            run_config['output_file'] = run_config['output_file'].replace('[' + key + ']', str(params[key]))
            run_config['output_file_cloud'] = run_config['output_file_cloud'].replace('[' + key + ']', str(params[key]))
        return run_config

    def replace_one_field_tags(value, params):
        for key in params.keys():
            value = value.replace('[' + key + ']', str(params[key]))

        return value

    def build_experiments_params(config):
        experiments = []
        iterators = []
        for iterator in config['nested_iterators'].keys():
            expanded_experiments = []
            if len(experiments) > 0:
                for experiment in experiments:
                    for value in config['nested_iterators'][iterator]:
                        new_expriment = copy.deepcopy(experiment)
                        new_expriment.update({iterator: value})
                        expanded_experiments.append(new_expriment)
            else:
                for value in config['nested_iterators'][iterator]:
                    new_expriment = {iterator: value}
                    expanded_experiments.append(new_expriment)
            experiments = expanded_experiments

        if len(config['list_iterators']) > 0:
            expanded_experiments = []
            for value in config['list_iterators']:
                if len(experiments) > 0:
                    for experiment in experiments:
                        new_expriment = copy.deepcopy(experiment)
                        new_expriment.update(value)
                        expanded_experiments.append(new_expriment)
                else:
                    new_expriment = value
                    expanded_experiments.append(new_expriment)
            experiments = expanded_experiments
        return experiments

    def build_evaluate_bash_command(run_config, run_name):
        bash_command = 'python -m allennlp.run evaluate ' + run_config['model'] + ' '
        bash_command += run_config['eval_set'] + ' '
        bash_command += '--output-file ' + runner_config['output_file'] + ' '
        bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
        bash_command += ' --cuda-device [GPU_ID]'
        bash_command += allennlp_include_packages()
        return bash_command

    def build_train_bash_command(run_config, run_name):
        bash_command = 'python -m allennlp.run train ' + run_config['master_config'] + ' '
        bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
        bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
        bash_command += allennlp_include_packages()
        return bash_command

    def build_finetune_bash_command(run_config, run_name):
        bash_command = 'python -m allennlp.run fine-tune -m ' + run_config['source_model_path'] + ' '
        bash_command += '-c ' + run_config['master_config'] + ' '
        bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
        bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
        bash_command += allennlp_include_packages()
        return bash_command




{"match": {"experiment": "020_BERT_exp1_evaluate"}},
"from": 0, "size": 500,

# ## delete from elastic results

# In[14]:


DRY_RUN = False

query = {
    "query": {
        "bool": {
            "should": [
                {"match": {"experiment": "063_BERT_evaluate_MRQA"}},
                {"match": {"eval_set": "SQuAD"}}
            ],
            "minimum_should_match": 2,
        }
    },
    "sort": [
        {
            "log_timestamp": {
                "order": "asc"
            }
        }
    ]
}
if DRY_RUN:
    results = ElasticLogger().es.search(index="multiqa_logs", body=query)
    print(results['hits']['total'])
    results = pd.DataFrame([res['_source'] for res in results['hits']['hits']])
    if len(results) > 0:
        results['log_timestamp'] = pd.to_datetime(results['log_timestamp'])
        results = results.set_index(['log_timestamp'])
else:
    print('DELETING %s' % (str(query)))
    time.sleep(10)
    results = ElasticLogger().es.delete_by_query(index="multiqa_logs", body=query)
results

# ## resources to spare

# In[11]:


queue = 'rack-gamir-g06'
# queue = 'savant'
queue = 'rack-jonathan-g06'
print(queue)
resources_to_spare = 1
name = 'resources to spare'
print('resources to spare %d', resources_to_spare)
config = {}
config['operation'] = "resources to spare"
config['resources_to_spare'] = resources_to_spare
send_to_queue(name, queue, config)

# ##  restart

# In[28]:


# hosts = ['rack-jonathan-g08','rack-jonathan-g02','rack-gamir-g03','rack-gamir-g04','rack-gamir-g05','savant']
hosts = ['rack-jonathan-g08', 'rack-jonathan-g02', 'rack-gamir-g03', 'rack-gamir-g04', 'rack-gamir-g05', 'savant']
# hosts = ['pc-jonathan1']
# hosts = ['rack-gamir-g03','rack-gamir-g04']
# hosts = ['savant']
config = {}
for queue in hosts:
    name = 'restart'
    print('restart %s' % queue)
    config['operation'] = "restart runner"
    config['clear_jobs'] = 'True'
    send_to_queue(name, queue, config)
    time.sleep(5)

# ## kill job

# In[7]:


# queue = 'pc-jonathan1'
queue = 'rack-jonathan-g08'
# queue = 'rack-gamir-g06'
# queue = 'savant'
# queue = 'GPUs'
config = {}
name = "kill job"
config['operation'] = "kill job"
config['experiment_name'] = "roberta-large/090_oLMpics_finetuned_LearningCurves_composition_v2_250_1_1225_1616"
print(queue)
send_to_queue(name, queue, config)

# # preproc

# In[8]:


config = {}
queue = 'pc-jonathan1'

# datasets = ['HotpotQA'];
# ,'TriviaQA-wiki2','SearchQA2'
datasets = ["HotpotQA2", "Squad", "NewsQA", "SearchQA2", "TriviaQA-G2", "TriviaQA-unfilt2", "WikiHop", "ComQA2", "ComplexQuestions2",
            "ComplexWebQuestions2", "DROP2"]
# datasets = ["HotpotQA2", "Squad", "NewsQA","SearchQA2","TriviaQA-G2","TriviaQA-unfilt2"]
# datasets = ['HotpotQA','HotpotQA_multi_title','HotpotQA_with_ents','HotpotQA_with_ents_multi_title']
datasets = ["HotpotQA_only_supfacts"]
DRY_RUN = False
num_of_docs = 10
MRQAStyle = False

params = [(True, ['train', 'dev'], '-1'),
          (False, ['train', 'dev'], '-1'),
          (True, ['train'], '15000'),
          (False, ['train'], '15000'),
          (True, ['train'], '75000'),
          (False, ['train'], '75000')]
require_answer_in_question = False
require_answer_in_doc = False
extra = ''

# dev with answers :
# params = [(True,['dev'],'-1')]
# require_answer_in_question = True
# require_answer_in_doc = True
# extra = 'only_answers_'

params = [(True, ['dev', 'train'], '-1')]
# require_answer_in_question = True
# require_answer_in_doc = True
# params = [(True,['test'],'-1')]


for BERT_format, splits, sample in params:

    if BERT_format:
        sort = False
        docsize = '512'
        only_answers = True
        use_rank = False
        # run_file = 'preprocess'
        run_file = 'preprocess_bert_all_answers'
    else:
        sort = True
        docsize = '400'
        only_answers = False
        use_rank = True
        run_file = 'preprocess_docqa'
    for dataset in datasets:
        for split in splits:
            output_name = "s3://multiqa/preproc/"
            if BERT_format and num_of_docs == 1 and MRQAStyle:
                output_name += 'MRQAData/'
            elif BERT_format:
                output_name += 'BERT/'
            else:
                output_name += 'DocQA/'

            if MRQAStyle:
                pass
            elif sample == '15000':
                output_name += 'small/'
            elif sample == '75000':
                output_name += '75000/'
                extra = '75000_'
            else:
                output_name += 'Full/'
                # extra = '75000_'
            if BERT_format and num_of_docs == 1:
                output_name += dataset.replace('2', '') + '_'
            else:
                output_name += dataset + '_'

            if num_of_docs < 10:
                extra += '5docs_'
            output_name += extra + docsize + "_" + split

            name = "preprocess_" + output_name
            output_name += ".jsonl.zip"

            command = "python scripts/multiqa/" + run_file + ".py s3://multiqa/datasets/" + dataset + "_" + split + ".jsonl.zip  " + output_name + " --n_processes 10 --ndocs " + str(
                num_of_docs) + " --docsize " + docsize + " --titles True --sample_size " + sample + " "
            if use_rank:
                command += ' --use_rank True '
            else:
                command += ' --use_rank False '

            if MRQAStyle:
                command += ' --MRQA_style True '

            if BERT_format:
                command += ' --BERT_format True '
            # else:
            #    command += ' --BERT_format False '

            if require_answer_in_doc:
                command += ' --require_answer_in_doc True '
            else:
                if only_answers and split == "train":
                    command += ' --require_answer_in_doc True '
                else:
                    command += ' --require_answer_in_doc False '

            if sort:
                command += ' --sort_by_question True '

            if require_answer_in_question:
                command += "--require_answer_in_question True"
            else:
                if split == "train":
                    command += "--require_answer_in_question True"
                else:
                    command += "--require_answer_in_question False"

            name += "_" + datetime.datetime.now().strftime("%m%d_%H%M")

            config['operation'] = "run job"
            config['bash_command'] = command
            config['resource_type'] = 'CPU'

            print(name)
            print('## bash_command= %s' % (config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
            print('--------------------------------\n')

            if not DRY_RUN:
                print('\n!!!! RUNNING !!!\n')
                time.sleep(5)
                send_to_queue(name, queue, config)
                time.sleep(1)

# ## preproc MRQA

# In[10]:


config = {}
queue = 'pc-jonathan1'

# datasets = ['HotpotQA'];
# ,'TriviaQA-wiki2','SearchQA2'
# datasets = ["HotpotQA2", "Squad", "NewsQA","SearchQA2","TriviaQA-G2","TriviaQA-unfilt2",\
#            "WikiHop","ComQA2","ComplexQuestions2","ComplexWebQuestions2","DROP2"]
datasets = ["SearchQA2", "TriviaQA-G2"]
DRY_RUN = False
num_of_docs = 2
MRQAStyle = False

params = [(True, ['train', 'dev'], '-1'),
          (False, ['train', 'dev'], '-1'),
          (True, ['train'], '15000'),
          (False, ['train'], '15000'),
          (True, ['train'], '75000'),
          (False, ['train'], '75000')]
require_answer_in_question = True
require_answer_in_doc = False
extra = ''

# dev with answers :
# params = [(True,['dev'],'-1')]
# require_answer_in_question = True
# require_answer_in_doc = True
# extra = 'only_answers_'

params = [(True, ['dev'], '-1')]
# require_answer_in_question = True
# require_answer_in_doc = True
# params = [(True,['test'],'-1')]


for BERT_format, splits, sample in params:

    if BERT_format:
        sort = False
        docsize = '512'
        only_answers = True
        use_rank = False
        # run_file = 'preprocess'
        run_file = 'preprocess_bert_all_answers'
    else:
        sort = True
        docsize = '400'
        only_answers = False
        use_rank = True
        run_file = 'preprocess_docqa'
    for dataset in datasets:
        for split in splits:
            output_name = "s3://multiqa/preproc/"
            # if BERT_format and num_of_docs == 2 and MRQAStyle:
            #    output_name += 'MRQAData/'
            # elif BERT_format:
            #    output_name += 'BERT/'
            # else:
            #    output_name += 'DocQA/'
            output_name += 'MRQA800/'

            if MRQAStyle:
                pass
            elif sample == '15000':
                output_name += 'small/'
            elif sample == '75000':
                output_name += '75000/'
                extra = '75000_'
            else:
                output_name += 'Full/'
                # extra = '75000_'
            if BERT_format and num_of_docs == 1:
                output_name += dataset.replace('2', '') + '_'
            else:
                output_name += dataset + '_'
            output_name += extra + docsize + "_" + split

            name = "preprocess_" + output_name
            output_name += ".jsonl.zip"

            command = "python scripts/multiqa/" + run_file + ".py s3://multiqa/datasets/" + dataset + "_" + split + ".jsonl.zip  " + output_name + " --n_processes 10 --ndocs " + str(
                num_of_docs) + " --docsize " + docsize + " --titles True --USE_TFIDF False --IGNORE_QAS_NOT_PROC True --sample_size " + sample + " "
            if use_rank:
                command += ' --use_rank True '
            else:
                command += ' --use_rank False '

            if MRQAStyle:
                command += ' --MRQA_style True '

            if BERT_format:
                command += ' --BERT_format True '
            # else:
            #    command += ' --BERT_format False '

            if require_answer_in_doc:
                command += ' --require_answer_in_doc True '
            else:
                if only_answers and split == "train":
                    command += ' --require_answer_in_doc True '
                else:
                    command += ' --require_answer_in_doc False '

            if sort:
                command += ' --sort_by_question True '

            if require_answer_in_question:
                command += "--require_answer_in_question True"
            else:
                if split == "train":
                    command += "--require_answer_in_question True"
                else:
                    command += "--require_answer_in_question False"

            name += "_" + datetime.datetime.now().strftime("%m%d_%H%M")

            config['operation'] = "run job"
            config['bash_command'] = command
            config['resource_type'] = 'CPU'

            print(name)
            print('## bash_command= %s' % (config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
            print('--------------------------------\n')

            if not DRY_RUN:
                print('\n!!!! RUNNING !!!\n')
                time.sleep(5)
                send_to_queue(name, queue, config)
                time.sleep(1)

# # Run Rxperiment

# In[13]:


# queue = 'rack-jonathan-g02'
# queue = 'rack-gamir-g03'
FORCE_RUN = False
queue = 'GPUs'
print('Running new job on queue = %s', queue)

# experiment_name = '017_BERT_exp2_finetune_from_partial_exp'
# experiment_name = '018_BERT_exp2_finetune_from_full_exp'
# experiment_name = '016_GloVe_exp3_finetune_target_sizes'
experiment_name = '019_BERT_finetune_Full'
job_type = 'finetune'
DRY_RUN = True

config = add_override_config(job_type + '/' + experiment_name + '.json')
print('Description: %s \n\n' % (config['description']))
config['override_config'] = config['allennlp_override']

s3_done_experiments = get_s3_experiments('/'.join(config['output_file_cloud'].split('/')[3:6]))
currently_running_experiments = get_running_experiments()

experiments = build_experiments_params(config)

for params in experiments:
    run_config = copy.deepcopy(config)
    run_config = dataset_specific_override(dataset, run_config)

    params['EXPERIMENT'] = experiment_name
    # adding execution time name
    run_name_no_date = replace_one_field_tags(run_config['run_name'], params)
    run_name = run_name_no_date + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
    params['RUN_NAME'] = run_name
    params['OUTPUT_FILE_CLOUD'] = replace_one_field_tags(run_config['output_file_cloud'], params)
    run_config = replace_tag_params(run_config, params)

    # checking if this run has already been run
    if not FORCE_RUN:
        if job_type == 'evaluate':  # evaluate stores results in elastic..
            pass
        else:
            if '/'.join(run_config['output_file_cloud'].split('/')[3:]) in s3_done_experiments:
                print('!! %s already found in s3, NOT RUNNING. \n' % (run_name))
                continue
        if len([exp for exp in currently_running_experiments if exp.lower().startswith(run_name_no_date.lower())]) > 0:
            print('!! %s currently running , NOT RUNNING. \n' % (run_name_no_date))
            continue

    # Building command
    runner_config = {'output_file': run_config['output_file']}
    runner_config['operation'] = "run job"

    if "post_proc_bash" in run_config:
        runner_config['post_proc_bash'] = run_config['post_proc_bash']
    runner_config['resource_type'] = 'GPU'
    runner_config['override_config'] = run_config['override_config']
    if job_type == 'evaluate':
        runner_config['bash_command'] = build_evaluate_bash_command(run_config, run_name)
    elif job_type == 'train':
        runner_config['bash_command'] = build_train_bash_command(run_config, run_name)
    elif job_type == 'finetune':
        runner_config['bash_command'] = build_finetune_bash_command(run_config, run_name)
    runner_config['bash_command'] = replace_one_field_tags(runner_config['bash_command'], params)
    print(run_name)
    print('## bash_command= %s' % (runner_config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
    print('## output_file_cloud= %s' % (run_config['output_file_cloud']))
    print('## output_file= %s' % (runner_config['output_file']))
    if "post_proc_bash" in run_config:
        print('## post_proc_bash= %s' % (runner_config['post_proc_bash']))
    print('--------------------------------\n')

    if not DRY_RUN:
        send_to_queue(run_name, queue, runner_config)
        time.sleep(1)
    # break

# In[ ]:


queue = 'GPUs'
print('queue = %s', queue)
count = 0
s3 = boto3.client("s3")

all_keys = []
all_objects = []
# ,'trans_3e','trans_6e','trans_9e'
# ,['75000','small','mix']
# ,['finetuned']
for train_type in ['75000', 'small', 'mix', 'finetuned']:
    search_res = s3.list_objects(Bucket='multiqa', Prefix='models/' + train_type + '/')
    if 'Contents' in search_res:
        all_objects += search_res['Contents']
    search_res = s3.list_objects(Bucket='multiqa', Prefix='eval/' + train_type + '/')
    if 'Contents' in search_res:
        all_objects += search_res['Contents']

    models = []

    for obj in all_objects:
        all_keys.append(obj['Key'])
        if obj['Key'].startswith('models/' + train_type + '/') and obj['Key'].find('.tar.gz') > -1:
            models.append(obj['Key'].replace('models/' + train_type + '/', '').replace('.tar.gz', ''))

    eval_sets = ['Squad', 'NewsQA', 'HotpotQA', 'TriviaQA-G', 'SearchQA', 'ComplexWebQuestions', 'ComQA', 'ComplexQuestions', 'WikiHop']
    # eval_sets = ['Squad','NewsQA','HotpotQA', 'TriviaQA-G','SearchQA','ComplexWebQuestions']

    experiments = []
    for model in models:
        for tar in eval_sets:
            experiments.append((model, tar))

    # experiments = [('NewsQA_to_TriviaQA-G', 'ComplexWebQuestions')]

    for model, tar in experiments:

        if train_type == 'finetuned' and not model.startswith(tar.replace('_dev', '')):
            continue
        base_model = 's3://multiqa/models/' + train_type + '/' + model + '.tar.gz '
        tar_set = 's3://multiqa/preproc/' + tar + '_400_dev.jsonl.zip '
        config['override_config'] = add_override_config('MultiQA_Override.json')
        config['override_config']['trainer']["cuda_device"] = '[GPU_ID]'
        config['master_config'] = 's3://multiqa/config/MultiQA_GloVe.json '
        if train_type != 'preproc':
            eval_name = tar + '_dev_on_' + model + '_' + train_type
        else:
            eval_name = tar + '_dev_on_' + model

        if len([key for key in all_keys if key.find(train_type + '/' + tar + '_dev_on_' + model) > -1]) > 0:
            continue

        name = eval_name + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
        config['operation'] = "run job"
        bash_command = 'python -m allennlp.run evaluate ' + base_model + tar_set
        bash_command += '--output-file results/' + eval_name + '.json '
        bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
        bash_command += allennlp_include_packages()
        bash_command += ' --cuda-device [GPU_ID]'

        config[
            'post_proc_bash'] = 'aws s3 cp results/' + eval_name + '.json s3://multiqa/eval/' + train_type + '/' + eval_name + '.json --acl public-read'
        config['resource_type'] = 'GPU'
        config['bash_command'] = bash_command

        print(eval_name)
        count += 1

        send_to_queue(name, queue, config)

        time.sleep(1)
print(count)

# # Archived

# ## BERT NEW

# In[ ]:


# queue = 'rack-jonathan-g04'
queue = 'rack-gamir-g05'
# queue = 'GPUs'
print('queue = %s', queue)

model = 'SearchQA'
preproc_type = '_512'
# train_type = '_req-answers'
train_type = ''

config = {}
config_path = '/Users/alontalmor/Dropbox/Backup/QAResearch/MultiQA/configs/'
with open(config_path + 'MultiQA_BERT_override.json', 'r') as f:
    config['override_config'] = json.load(f)
config['override_config']['trainer']["cuda_device"] = '[GPU_ID]'
# config['override_config']['trainer']["cuda_device"] = 4
config['override_config']["train_data_path"] = 's3://multiqa/preproc/BERT/' + model + train_type + preproc_type + '_train.jsonl.zip'
config['override_config']["validation_data_path"] = 's3://multiqa/preproc/BERT/' + model + preproc_type + '_dev.jsonl.zip'
# config['override_config']['dataset_reader']['type'] = 'multiqa+'
config['master_config'] = 's3://multiqa/config/MultiQA_BERT.json '
# config['override_config']["trainer"]["optimizer"]["lr"] = 0.0003
# config['override_config']["trainer"]["optimizer"]["weight_decay"] = 0.0005
if model == 'Squad':
    config['override_config']["trainer"]["num_epochs"] = 2
    config['override_config']["trainer"]["optimizer"]["t_total"] = 29199
elif model == 'HotpotQA':
    config['override_config']["trainer"]["num_epochs"] = 1
    config['override_config']["trainer"]["optimizer"]["t_total"] = 27600
elif model == 'SearchQA':
    config['override_config']["trainer"]["num_epochs"] = 1
    config['override_config']["trainer"]["optimizer"]["t_total"] = 36000

config['override_config']['model']['shared_norm'] = True
opt_type = config['override_config']["trainer"]["optimizer"]["type"]
lr = config['override_config']["trainer"]["optimizer"]["lr"]
eval_name = model

name = eval_name + '/BERT_new_preproc_' + train_type + preproc_type + '_' + opt_type + '_lr' + str(
    lr) + '_' + datetime.datetime.now().strftime("%m%d_%H%M")

config['operation'] = "run job"
bash_command = 'python -m allennlp.run train ' + config['master_config']
bash_command += '-s ' + '[MODEL_DIR]' + name + ' '
bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
bash_command += allennlp_include_packages()

config[
    'post_proc_bash'] = 'aws s3 cp [MODEL_DIR]' + name + '/model.tar.gz s3://multiqa/models/bert/' + eval_name + '.tar.gz --acl public-read'
config['resource_type'] = 'GPU'
config['output_file'] = '[MODEL_DIR]' + name
config['bash_command'] = bash_command
print(name)
print(config['override_config'])

send_to_queue(name, queue, config)

# ## Train model

# In[ ]:


# queue = 'rack-jonathan-g02'
queue = 'GPUs'
print('queue = %s', queue)

model = 'TriviaQA-web'
dataset_type = 'req-answers'
# dataset_type = 'small'


config['override_config'] = add_override_config('MultiQA_Override.json')
config['override_config']['trainer']["cuda_device"] = '[GPU_ID]'

## TODO!!!!!!!!!!!!!!!!!!
if dataset_type == 'small':
    config['override_config']["train_data_path"] = 's3://multiqa/15000/' + model + '_15000_400_train.jsonl.zip'
else:
    config['override_config']["train_data_path"] = 's3://multiqa/preproc/' + model + '_250_train.jsonl.zip'
config['override_config']["validation_data_path"] = 's3://multiqa/preproc/' + model + '_250_dev.jsonl.zip'

config['override_config']['dataset_reader']['type'] = 'multiqa+'
config['master_config'] = 's3://multiqa/config/MultiQA_GloVe_no_char_tokens.json '

config['override_config']['model']['shared_norm'] = False
# config['override_config']["trainer"]["optimizer"]["lr"] = 0.001
# config['override_config']["trainer"]["optimizer"]["weight_decay"] = 0.0001
eval_name = model
name = eval_name + '/GloVe_' + dataset_type + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
config['operation'] = "run job"
bash_command = 'python -m allennlp.run train ' + config['master_config']
bash_command += '-s ' + '[MODEL_DIR]' + name + ' '
bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
bash_command += allennlp_include_packages()

config[
    'post_proc_bash'] = 'aws s3 cp [MODEL_DIR]' + name + '/model.tar.gz s3://multiqa/models/' + dataset_type + '/' + eval_name + '.tar.gz --acl public-read'
config['resource_type'] = 'GPU'
config['bash_command'] = bash_command
print(name)
print(bash_command)

send_to_queue(name, queue, config)

# ## preprocess

# In[ ]:


queue = 'pc-jonathan1'
set = 'train'
dataset = 'TriviaQA-web'
docsize = '250'
sample = '-1'
sort = True

# In[ ]:


config = {}
queue = 'pc-jonathan1'
set = 'train'
dataset = 'TriviaQA-web'
docsize = '400'
sample = '-1'
sort = True
only_answers = True

output_name = "s3://multiqa/preproc/" + dataset + "_"
if sample != '-1':
    output_name += sample + "_"
if not sort:
    output_name += "unsorted_"
if only_answers:
    output_name += 'req-answers'
output_name += "_" + docsize + "_" + set + "_"

name = "preprocess_" + output_name
output_name += ".jsonl.zip"

command = "python scripts/multiqa/preprocess.py s3://multiqa/datasets/" + dataset + "_" + set + ".jsonl.zip  " + output_name + " --n_processes 10 --ndocs 10 --docsize " + docsize + " --titles True --use_rank True --sample_size " + sample + " "

if only_answers:
    command += ' --require_answer_in_doc True '
else:
    command += ' --require_answer_in_doc True '

if sort:
    command += ' --sort_by_question True '

if set == "train":
    command += "--require_answer_in_question True"
else:
    command += "--require_answer_in_question False"

name += "_" + datetime.datetime.now().strftime("%m%d_%H%M")

config['operation'] = "run job"
config['bash_command'] = command
config['resource_type'] = 'CPU'

print(name)
print(command)

send_to_queue(name, queue, config)

# ## epoch ablations

# In[ ]:


queue = 'GPUs'
# queue = 'rack-gamir-g05'
print('queue = %s', queue)

source_models = ['Squad', 'NewsQA', 'HotpotQA', 'TriviaQA-G', 'SearchQA']
tar_sets = ['Squad', 'NewsQA', 'HotpotQA', 'TriviaQA-G', 'SearchQA', 'ComplexWebQuestions', 'ComQA', 'ComplexQuestions']

# hyper params:
epoches = [3, 6, 9]
lr = 0.0005

s3 = boto3.client("s3")
all_objects = s3.list_objects(Bucket='multiqa')
existing_trans_models = []
all_keys = []
for num_of_epoches in epoches:
    transfer_type = 'tr_e' + str(num_of_epoches) + '_lr' + str(lr)
    for obj in all_objects['Contents']:
        all_keys.append(obj['Key'])
        if obj['Key'].find(transfer_type) > -1 and obj['Key'].find('.tar.gz') > -1:
            existing_trans_models.append(obj['Key'].replace('models/' + transfer_type + '/', '').replace('.tar.gz', ''))

experiments = []
for num_of_epoches in epoches:
    for model in source_models:
        for tar in tar_sets:
            if model != tar:
                experiments.append((model, tar, num_of_epoches))

# experiments = [('SearchQA','NewsQA')]

for model, tar, num_of_epoches in experiments:
    experiment_tag = 'tr_e' + str(num_of_epoches) + '_lr' + str(lr) + '/' + tar + '_from_' + model
    if len([key for key in all_keys if key.find(experiment_tag) > -1]) > 0:
        continue

    base_model = 's3://multiqa/models/75000/' + model + '.tar.gz '
    config['override_config'] = add_override_config('MultiQA_Override.json')
    config['override_config']['trainer']["cuda_device"] = '[GPU_ID]'
    config['override_config']["train_data_path"] = 's3://multiqa/15000/' + tar + '_15000_400_train.jsonl.zip'
    config['override_config']["validation_data_path"] = 's3://multiqa/preproc/' + tar + '_400_dev.jsonl.zip'
    config['override_config']["trainer"]["num_epochs"] = num_of_epoches
    config['override_config']["trainer"]["optimizer"]["lr"] = lr
    # config['override_config']["trainer"]["optimizer"]["weight_decay"] = 0.0
    # config['override_config']['model']["text_field_embedder"] = {"allow_unmatched_keys":True}
    config['master_config'] = 's3://multiqa/config/MultiQA_GloVe_no_char_tokens.json '
    name = experiment_tag + '/' + datetime.datetime.now().strftime("%m%d_%H%M")
    config['operation'] = "run job"
    bash_command = 'python -m allennlp.run fine-tune -m ' + base_model + ' '
    bash_command += '-c ' + config['master_config']
    bash_command += '-s ' + '[MODEL_DIR]' + name + ' '
    bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    # bash_command += ' --extend-vocab '
    bash_command += allennlp_include_packages()

    config['post_proc_bash'] = 'aws s3 cp [MODEL_DIR]' + name + '/model.tar.gz s3://multiqa/models/tr_e' + str(
        num_of_epoches) + '_lr' + str(lr) + '/' + tar + '_from_' + model + '.tar.gz --acl public-read'
    config['resource_type'] = 'GPU'
    config['bash_command'] = bash_command

    print(experiment_tag)

    continue
    send_to_queue(name, queue, config)
    time.sleep(1)

# In[ ]:


queue = 'GPUs'
# queue = 'rack-gamir-g03'
# queue = 'rack-jonathan-g02'
# queue = 'test'
print('queue = %s', queue)

source_models = ['mix', 'NewsQA', 'Squad', 'HotpotQA', 'TriviaQA-G', 'SearchQA']
# source_models = ['Squad','TriviaQA-G']
# tar_sets = ['NewsQA','Squad','HotpotQA', 'TriviaQA-G','SearchQA','ComplexWebQuestions','ComQA','ComplexQuestions']
tar_sets = ['WikiHop']

s3 = boto3.client("s3")
all_objects = s3.list_objects(Bucket='multiqa', Prefix='models/finetuned')
existing_finetuned_models = []
for obj in all_objects['Contents']:
    existing_finetuned_models.append(obj['Key'])

experiments = []
for model in source_models:
    for tar in tar_sets:
        if model != tar:
            experiments.append((model, tar))

count = 0
for model, tar in experiments:
    experiment_tag = 'finetuned/' + tar + '_from_' + model
    if len([key for key in existing_finetuned_models if key.find(experiment_tag) > -1]) > 0:
        continue
    base_model = 's3://multiqa/models/75000/' + model + '.tar.gz '
    config['override_config'] = add_override_config('MultiQA_finetune_Override.json')
    config['override_config']['trainer']["cuda_device"] = '[GPU_ID]'
    config['override_config']["train_data_path"] = 's3://multiqa/15000/' + tar + '_15000_400_train.jsonl.zip'
    config['override_config']["validation_data_path"] = 's3://multiqa/preproc/' + tar + '_400_dev.jsonl.zip'
    config['master_config'] = 's3://multiqa/config/MultiQA_GloVe_no_char_tokens.json '
    name = experiment_tag + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
    config['operation'] = "run job"
    bash_command = 'python -m allennlp.run fine-tune -m ' + base_model + ' '
    bash_command += '-c ' + config['master_config']
    bash_command += '-s ' + '[MODEL_DIR]' + name + ' '
    bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()

    config[
        'post_proc_bash'] = 'aws s3 cp [MODEL_DIR]' + name + '/model.tar.gz s3://multiqa/models/finetuned/' + tar + '_from_' + model + '.tar.gz --acl public-read'
    config['resource_type'] = 'GPU'
    config['bash_command'] = bash_command

    print(experiment_tag)
    count += 1
    continue
    send_to_queue(name, queue, config)
    time.sleep(1)
print(count)

# # Run train experiment

# In[ ]:


# queue = 'rack-jonathan-g04'
# queue = 'rack-gamir-g03'
queue = 'GPUs'
print('Running new job on queue = %s', queue)

experiment_name = '003_2Epoches_One_Instance_Per_Question_train_BERT'

config = add_override_config(experiment_name + '.json')
print('Description: %s \n\n' % (config['description']))
config['override_config'] = config['allennlp_override']

for dataset in config['datasets']:
    run_config = copy.deepcopy(config)

    run_config = dataset_specific_override(dataset, run_config)

    # adding execution time name
    run_name = dataset + '/' + experiment_name + '_'
    run_name += 'LR' + str(run_config['override_config']["trainer"]["optimizer"]["lr"]) + '_'
    run_name += 'E' + str(run_config['override_config']["trainer"]["num_epochs"]) + '_'
    run_name += datetime.datetime.now().strftime("%m%d_%H%M")
    run_config = replace_tags(dataset, run_config)

    # Building command
    runner_config = {'output_file': run_config['output_file'], 'output_file_cloud_target': run_config['output_file_cloud_target']}
    runner_config['operation'] = "run job"
    bash_command = 'python -m allennlp.run train ' + run_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()
    runner_config['post_proc_bash'] = 'aws s3 cp ' + runner_config['output_file'] + ' ' + runner_config[
        'output_file_cloud_target'] + ' --acl public-read'
    runner_config['resource_type'] = 'GPU'
    runner_config['override_config'] = run_config['override_config']
    runner_config['bash_command'] = bash_command
    print(run_name)
    print('## bash_command= %s' % (runner_config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
    print('## output_file= %s' % (runner_config['output_file']))
    print('## post_proc_bash= %s' % (runner_config['post_proc_bash']))
    print('--------------------------------\n')

    send_to_queue(run_name, queue, runner_config)
    time.sleep(1)

# # Run fine-tune experiment

# In[ ]:


queue = 'rack-jonathan-g02'
# queue = 'rack-gamir-g03'
# queue = 'GPUs'
print('Running new job on queue = %s', queue)

experiment_name = '005_GloVe_FineTuning_Curves'

config = add_override_config(experiment_name + '.json')
print('Description: %s \n\n' % (config['description']))
config['override_config'] = config['allennlp_override']

# building experiments:
finetune_experiments = []
if config['1source_to_1target']:
    for source, tar in zip(config['source_datasets'], config['target_datasets']):
        finetune_experiments.append((source, tar))
else:
    for source in config['source_datasets']:
        for tar in config['target_datasets']:
            if model != tar:
                finetune_experiments.append((source, tar))

for source_dataset, dataset in finetune_experiments:
    run_config = copy.deepcopy(config)
    run_config = dataset_specific_override(dataset, run_config)

    # adding execution time name
    run_name = dataset + '_from_' + source_dataset + '/' + experiment_name + '_'
    # run_name += 'LR' + str(run_config['override_config']["trainer"]["optimizer"]["lr"]) +'_'
    # run_name += 'E' + str(run_config['override_config']["trainer"]["num_epochs"]) +'_'
    run_name += '75000_'
    run_name += datetime.datetime.now().strftime("%m%d_%H%M")
    run_config = replace_tags(dataset, run_config, source_dataset)

    # Building command
    runner_config = {'output_file': run_config['output_file'], 'output_file_cloud_target': run_config['output_file_cloud_target']}
    runner_config['operation'] = "run job"
    bash_command = 'python -m allennlp.run fine-tune -m ' + run_config['source_model_path'] + ' '
    bash_command += '-c ' + run_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()

    runner_config['post_proc_bash'] = 'aws s3 cp ' + runner_config['output_file'] + ' ' + runner_config[
        'output_file_cloud_target'] + ' --acl public-read'
    runner_config['resource_type'] = 'GPU'
    runner_config['override_config'] = run_config['override_config']
    runner_config['bash_command'] = bash_command
    print(run_name)
    print('## bash_command= %s' % (runner_config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
    print('## output_file= %s' % (runner_config['output_file']))
    print('## post_proc_bash= %s' % (runner_config['post_proc_bash']))
    print('--------------------------------\n')

    send_to_queue(run_name, queue, runner_config)
    time.sleep(1)


