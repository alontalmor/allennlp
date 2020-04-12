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

@DatasetReader.register("transformer_binary_qa")
class TransformerBinaryReader(DatasetReader):
    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 add_prefix: bool = False,
                 sample: int = -1) -> None:
        super().__init__()

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._sample = sample
        self._add_prefix = add_prefix
        self._debug_prints = -1

    @overrides
    def _read(self, file_path: str):
        self._debug_prints = 5
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
            self._debug_prints -= 1
            if self._debug_prints >= 0:
                logger.info(f"====================================")
                logger.info(f"Input json: {item_json}")
            item_id = item_json["id"]

            statement_text = item_json["phrase"]
            context = item_json["context"] if "context" in item_json else None

            yield self.text_to_instance(
                    item_id=item_id,
                    question=statement_text,
                    answer_id=item_json["answer"],
                    context = context)

        data_file.close()

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         answer_id: int = None,
                         context: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        qa_tokens = self.transformer_features_from_qa(question, context)
        qa_field = TextField(qa_tokens, self._token_indexers)

        fields['phrase'] = qa_field
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)
        metadata = {
            "id": item_id,
            "question_text": question,
            "context": context,
            "correct_answer_index": answer_id
        }
        if self._debug_prints >= 0:
            logger.info(f"Tokens: {qa_tokens}")
            logger.info(f"Label: {answer_id}")
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def transformer_features_from_qa(self, question: str, context: str):
        if self._add_prefix:
            question = "Q: " + question
            if context is not None:
                context = "C: " + context
        if context is not None:
            tokens = self._tokenizer.tokenize_sentence_pair(question, context)
        else:
            tokens = self._tokenizer.tokenize(question)
        return tokens
