from typing import Dict, List
import json
import logging
import gzip
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("transformer_mc_qa")
class TransformerMCQAReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 num_choices: int = 4,
                 add_prefix: int = 0,
                 sample: int = -1) -> None:
        super().__init__()

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model)
        self._tokenizer_internal = self._tokenizer._tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._num_choices = num_choices
        self._add_prefix = add_prefix

        for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
            if model in pretrained_model:
                self._model_type = model
                break

    @overrides
    def _read(self, file_path: str):
        cached_file_path = cached_path(file_path)

        if file_path.endswith('.gz'):
            data_file = gzip.open(cached_file_path, 'rb')
        else:
            data_file = open(cached_file_path, 'r')


        logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
        item_jsons = []
        for line in data_file:
            item_jsons.append(json.loads(line.strip()))

        if self._sample != -1:
            item_jsons = random.sample(item_jsons, self._sample)
            logger.info("Sampling %d examples", self._sample)

        for item_json in Tqdm.tqdm(item_jsons,total=len(item_jsons)):
            item_id = item_json["id"]

            statement_text = item_json["phrase"]

            yield self.text_to_instance(
                    item_id=item_id,
                    question=statement_text,
                    answer_id=item_json["answer"])

        data_file.close()

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         answer_id: int = None) -> Instance:
        fields: Dict[str, Field] = {}

        qa_tokens, segment_ids = self.transformer_features_from_qa(question)
        qa_field = TextField(qa_tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(segment_ids, qa_field)

        fields['phrase'] = qa_field
        fields['segment_ids'] = segment_ids_field
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "correct_answer_index": answer_id
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def transformer_features_from_qa(self, question: str):

        tokens = self._tokenizer.tokenize(question)
        segment_ids = [0] * len(tokens)

        return tokens, segment_ids