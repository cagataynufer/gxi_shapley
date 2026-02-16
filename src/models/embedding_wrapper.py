import torch
import torch.nn as nn


class EmbeddingWrapper(nn.Module):
    """
    Wraps a text model so it accepts embeddings instead of token_ids.

    This is required because Gradient×Input expects the first argument
    to be differentiable. The wrapper is head-agnostic and supports
    both sentiment and NLI classifiers.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()

        self.base_model = base_model
        self.transformer = base_model.distilbert

        # Resolve classifier head automatically
        if hasattr(base_model, "fc"):
            self.classifier = base_model.fc
        elif hasattr(base_model, "classifier"):
            self.classifier = base_model.classifier
        else:
            raise AttributeError(
                "Base model must define either `.fc` or `.classifier`."
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # embeddings: (L, D) or (1, L, D)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        outputs = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits
