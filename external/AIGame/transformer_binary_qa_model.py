from typing import Dict, Optional, List, Any

from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlnet import XLNetModel
from transformers.modeling_bert import BertModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import SequenceSummary
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy


@Model.register("transformer_binary_qa")
class TransformerBinaryQA(Model):
    """
    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._pretrained_model = pretrained_model
        if 'roberta' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'xlnet' in pretrained_model:
            self._padding_value = 5  # The index of the XLNet padding token
            self._transformer_model = XLNetModel.from_pretrained(pretrained_model)
            self.sequence_summary = SequenceSummary(self._transformer_model.config)
        elif 'albert' in pretrained_model:
            self._transformer_model = AlbertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'bert' in pretrained_model:
            self._transformer_model = BertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        else:
            assert (ValueError)

        for name, param in self._transformer_model.named_parameters():
            if layer_freeze_regexes and requires_grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            else:
                grad = requires_grad
            if grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        transformer_config = self._transformer_model.config
        transformer_config.num_labels = 1
        self._output_dim = self._transformer_model.config.hidden_size

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, 1)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()

        self._accuracy = BooleanAccuracy()
        self._loss = torch.nn.BCEWithLogitsLoss()

        self._debug = -1


    def forward(self,
                    phrase: Dict[str, torch.LongTensor],
                    segment_ids: torch.LongTensor = None,
                    label: torch.LongTensor = None,
                    metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = phrase['tokens']
        batch_size = input_ids.size(0)

        question_mask = (input_ids != self._padding_value).long()

        # Segment ids are not used by RoBERTa
        if 'roberta' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                         # token_type_ids=util.combine_initial_dims(segment_ids),
                                                                         attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        if 'albert' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                         # token_type_ids=util.combine_initial_dims(segment_ids),
                                                                         attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        elif 'xlnet' in self._pretrained_model:
            transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                          token_type_ids=util.combine_initial_dims(segment_ids),
                                                          attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self.sequence_summary(transformer_outputs[0])

        elif 'bert' in self._pretrained_model:
            last_layer, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                token_type_ids=util.combine_initial_dims(segment_ids),
                                                                attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        else:
            assert (ValueError)

        label_logits = self._classifier(cls_output)
        label_logits_flat = label_logits.squeeze(1)
        label_logits = label_logits.view(-1, 1)

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.sigmoid(label_logits_flat)
        output_dict['answer_index'] = label_logits_flat > 0


        if label is not None:
            loss = self._loss(label_logits, label.view(-1,1).float())
            self._accuracy(label_logits > 0 , label.view(-1,1) > 0)
            output_dict["loss"] = loss

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'EM': self._accuracy.get_metric(reset),
        }

