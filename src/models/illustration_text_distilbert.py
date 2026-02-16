import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertNLIClassifier(nn.Module):
    """
    Lightweight Natural Language Inference (NLI) classifier
    on top of pretrained DistilBERT.

    The model is used purely for illustration purposes
    (no finetuning, frozen encoder).
    """

    def __init__(self, num_classes: int = 3):
        """
        Parameters
        ----------
        num_classes : int, default=3
            Number of NLI classes:
            {entailment, contradiction, neutral}
        """
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.classifier = nn.Linear(768, num_classes)

        # Freeze transformer parameters
        for p in self.distilbert.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for sentence-pair NLI classification.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len)
            containing [CLS] premise [SEP] hypothesis [SEP].

        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).

        Returns
        -------
        logits : torch.Tensor
            Output logits of shape (batch_size, 3).
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS representation
        cls_rep = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        logits = self.classifier(cls_rep)
        return logits
