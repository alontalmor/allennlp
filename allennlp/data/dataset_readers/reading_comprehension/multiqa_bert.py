import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random
import numpy as np

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

def sample_contexts(instance_list,sample_size):
    random.seed(2)

    instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
    intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
    split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                              range(len(split_inds) - 1)]

    random.shuffle(per_question_instances)

    sampled_contexts = []
    num_of_qas = 0
    for question_instances in per_question_instances:
        if num_of_qas >= sample_size:
            break
        sampled_contexts += question_instances
        num_of_qas += 1
    return sampled_contexts


@DatasetReader.register("multiqa_bert")
class BERTQAReader(DatasetReader):
    """
    Reads a JSON-formatted Quesiton Answering in Context (QuAC) data file
    and returns a ``Dataset`` where the ``Instances`` have four fields: ``question``, a ``ListField``,
    ``passage``, another ``TextField``, and ``span_start`` and ``span_end``, both ``ListField`` composed of
    IndexFields`` into the ``passage`` ``TextField``.
    Two ``ListField``, composed of ``LabelField``, ``yesno_list`` and  ``followup_list`` is added.
    We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_text_lists'] and ``metadata['token_offsets']``.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_one_inst_per_question = False,
                 lazy: bool = False,
                 all_question_instances_in_batch = False,
                 sample_size: int = -1,
                 ) -> None:
        super().__init__(lazy)
        random.seed(0)
        logger.info('----------------- NEW SEED ---------------')
        self._tokenizer = tokenizer or WordTokenizer()
        self._sample_size = sample_size
        self._use_one_inst_per_question = use_one_inst_per_question
        self._all_question_instances_in_batch = all_question_instances_in_batch
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def build_instances(self, header, instances):

        # bucketing by QuestionID
        instance_list = instances
        instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
        intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                                  range(len(split_inds) - 1)]

        # sorting
        #sorting_keys = ['question_tokens', 'tokens']
        #instances_with_lengths = []
        #for instance in per_question_instances:
        #    padding_lengths = {key: len(instance[0][key]) for key in sorting_keys}
        #    instance_with_lengths = ([padding_lengths[field_name] for field_name in sorting_keys], instance)
        #    instances_with_lengths.append(instance_with_lengths)
        #instances_with_lengths.sort(key=lambda x: x[0])
        # random shuffle training
        #random.seed(8)
        #random.shuffle(instances_with_lengths)

        #per_question_instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

        # selecting instaces to add
        filtered_instances = []
        for question_instances in per_question_instances:
            if self._all_question_instances_in_batch:
                instances_to_add = question_instances
            else:
                # choose at most 2 instances from the same question:
                if len(question_instances) > 1 and self._use_one_inst_per_question:
                    inst_with_answers = [inst for inst in question_instances if inst['answers'] != []]
                    instances_to_add = random.sample(inst_with_answers, 1)
                elif len(question_instances) > 2:
                    # This part is inspired by Clark and Gardner, 17 - oversample the highest ranking documents.
                    # In thier work they use only instances with answers, so we will find the highest
                    # ranking instance with an answer (this also insures we have at least one answer in the chosen instances)

                    inst_with_answers = [inst for inst in question_instances if inst['answers'] != []]
                    instances_to_add = random.sample(inst_with_answers[0:2], 1)
                    # we assume each question will be visited once in an epoch
                    question_instances.remove(instances_to_add[0])
                    instances_to_add += random.sample(question_instances, 1)

                else:
                    instances_to_add = question_instances

                # Require at least one answer:
                if not any(inst['answers'] != [] for inst in instances_to_add):
                    continue

            filtered_instances += instances_to_add

        #logger.info("multiqa+: yielding %d instances ", len(filtered_instances))
        for inst_num, inst in enumerate(filtered_instances):
            tokenized_paragraph = [Token(text=t[0], idx=t[1]) for t in inst['tokens']]
            question_tokens = [Token(text=t[0], idx=t[1]) for t in inst['question_tokens']]
            new_passage = inst['text']
            new_answers = inst['answers']
            instance = util.make_reading_comprehension_instance_multiqa(question_tokens,
                                                                                 tokenized_paragraph,
                                                                                 self._token_indexers,
                                                                                 new_passage,
                                                                                 new_answers,
                                                                                 inst['metadata'],
                                                                                 header)

            yield instance

    @profile
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading the dataset")

        # supporting multi dataset training:
        instances = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            # if `file_path` is a URL, redirect to the cache
            logger.info("Reading file at %s", single_file_path)

            if single_file_path.find('jsonl') > 0:
                single_file_path_cached = cached_path(single_file_path)
                with zipfile.ZipFile(single_file_path_cached, 'r') as myzip:
                    with myzip.open(myzip.namelist()[0]) as myfile:
                        header = json.loads(myfile.readline())['header']
                        number_of_yielded_instances = 0
                        for line,example in enumerate(myfile):
                            # header
                            curr_instance = json.loads(example)
                            instances.append(curr_instance)


                            # making sure not to take all instances of the same question
                            if self._sample_size == -1 and len(instances)>10000 and instances[-1]['metadata']['question_id'] \
                                                    != instances[-2]['metadata']['question_id']:
                                remainder = instances[-1]
                                instances = instances[:-1]
                                for instance in self.build_instances(header, instances):
                                    number_of_yielded_instances += 1
                                    yield instance
                                instances = [remainder]

                        # per dataset sampling
                        if self._sample_size > -1:
                            instances = sample_contexts(instances, self._sample_size)

                        # yielding the remainder
                        for instance in self.build_instances(header, instances):
                            yield instance

