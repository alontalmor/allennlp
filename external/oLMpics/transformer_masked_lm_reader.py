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

@DatasetReader.register("transformer_masked_lm_qa")
class TransformerMaskedLMReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 num_choices: int = 4,
                 add_prefix: Dict[str, str] = None,
                 sample: int = -1) -> None:
        super().__init__()

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model)
        self._tokenizer_no_special_tokens = PretrainedTransformerTokenizer(pretrained_model, add_special_tokens=False)
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._num_choices = num_choices
        self._add_prefix = add_prefix or {}

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

            question_text = item_json["question"]["stem"]

            choice_label_to_id = {}
            choice_text_list = []
            choice_context_list = []
            choice_label_list = []
            choice_annotations_list = []

            any_correct = False
            choice_id_correction = 0

            for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                choice_label = choice_item["label"]
                choice_label_to_id[choice_label] = choice_id - choice_id_correction
                choice_text = choice_item["text"]

                choice_text_list.append(choice_text)
                choice_label_list.append(choice_label)

                if item_json.get('answerKey') == choice_label:
                    if any_correct:
                        raise ValueError("More than one correct answer found for {item_json}!")
                    any_correct = True


            if not any_correct and 'answerKey' in item_json:
                raise ValueError("No correct answer found for {item_json}!")


            answer_id = choice_label_to_id.get(item_json.get("answerKey"))
            # Pad choices with empty strings if not right number
            if len(choice_text_list) != self._num_choices:
                choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                if answer_id is not None and answer_id >= self._num_choices:
                    logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                    continue

            yield self.text_to_instance(
                    item_id=item_id,
                    question=question_text,
                    choice_list=choice_text_list,
                    answer_id=answer_id)

        data_file.close()

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None) -> Instance:
        fields: Dict[str, Field] = {}

        qa_fields = []
        all_masked_index_ids = []
        masked_labels_fields = []
        segment_ids_fields = []
        qa_tokens_list = []
        annotation_tags_fields = []
        for idx, choice in enumerate(choice_list):
            qa_tokens, segment_ids, masked_labels, masked_index_ids = \
                self.transformer_features_from_qa(question, choice)
            qa_field = TextField(qa_tokens, self._token_indexers)
            segment_ids_field = SequenceLabelField(segment_ids, qa_field)
            qa_fields.append(qa_field)
            qa_tokens_list.append(qa_tokens)
            segment_ids_fields.append(segment_ids_field)
            masked_labels_field = SequenceLabelField(masked_labels, qa_field)
            all_masked_index_ids.append(masked_index_ids)
            masked_labels_fields.append(masked_labels_field)

        if answer_id is not None:
            fields['phrase'] = qa_fields[answer_id]
            fields['masked_labels'] = masked_labels_fields[answer_id]
        else:
            fields['phrase'] = qa_fields[0]
            fields['masked_labels'] = masked_labels_fields[answer_id]

        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "all_masked_index_ids": all_masked_index_ids,
            "question_tokens_list": qa_tokens_list,
        }

        if len(annotation_tags_fields) > 0:
            fields['annotation_tags'] = ListField(annotation_tags_fields)
            metadata['annotation_tags'] = [x.array for x in annotation_tags_fields]

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def transformer_features_from_qa(self, question: str, answer: str):
        question = self._add_prefix.get("q", "") + question
        answer = self._add_prefix.get("a",  "") + answer

        # Alon changing mask type:
        if self._model_type in ['roberta','xlnet']:
            question = question.replace('[MASK]','<mask>')
        elif self._model_type in ['albert']:
            question = question.replace('[MASK]', '[MASK]>')

        question_tokens = self._tokenizer.tokenize(question)

        choice_tokens = [answer]
        choice_ids = self._tokenizer_no_special_tokens.tokenizer.encode('a ' + answer, add_special_tokens=False)[1:]

        if len(choice_ids) > 1:
            logger.error('more than one word-piece answer!')

        tokens = []
        segment_ids = []
        current_segment = 0
        # Alon, if the question is empty don't add seprators.
        masked_tokens = [(i, t) for i, t in enumerate(question_tokens) if t.text == self._tokenizer.tokenizer.mask_token]

        if len(masked_tokens) > 0:
            tokens += question_tokens
            segment_ids += len(tokens) * [current_segment]

        masked_labels = [-1] * len(tokens)
        masked_index_ids = []
        for i, t in enumerate(masked_tokens):
            masked_labels[t[0]] = choice_ids[i]
            masked_index_ids.append((t[0], choice_ids[i]))

        return tokens, segment_ids, masked_labels, masked_index_ids
