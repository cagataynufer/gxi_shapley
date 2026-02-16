"""
Text illustration for GXI-Shapley (single-feature game, NLI).

This script:
- loads a pretrained DistilBERT NLI classifier,
- selects a fixed (premise, hypothesis) pair,
- constructs a GXI-Shapley game on the CLS (decision) token,
- approximates token-level Shapley influences,
- saves a sorted influence table to disk.

The goal is to illustrate how contextual tokens influence
the decision-level representation in a transformer model.
"""

from pathlib import Path

import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

from shapiq.approximator import PermutationSamplingSV

from src.models.illustration_text_distilbert import DistilBertNLIClassifier
from src.gxi.game import GXIShapleyGame
from src.masking.text import TextMasker
from src.models.embedding_wrapper import EmbeddingWrapper
from src.data.illustration_text_samples import TEXT_SAMPLES


def main() -> None:
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results/illustration/text")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = "distilbert-base-uncased"
    max_len = 128

    sample_key = "nli_contradiction_contextual"

    top_k = 15
    budget = 2000
    batch_size = 40
    random_state = 0

    print("Running GXI-Shapley text illustration (NLI, CLS-centered)")
    print(f"Sample key: {sample_key}")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    model = DistilBertNLIClassifier().to(device)
    model.eval()

    wrapped_model = EmbeddingWrapper(model).to(device)

    # Prepare NLI text sample
    premise, hypothesis = TEXT_SAMPLES[sample_key]

    enc = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    print("Tokenizing sentence pair...")
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Sequence length: {len(tokens)}")

    # Target feature: CLS token (decision representation)
    token_index = 0
    target_token = tokens[token_index]  # should be [CLS]

    print(f"Target feature index i: {token_index}")
    print(f"Target feature token: '{target_token}'")

    # Obtain token embeddings
    with torch.no_grad():
        base_embeddings = model.distilbert.embeddings(input_ids)

    emb_single = base_embeddings[0]  # (L, D)
    seq_len, embed_dim = emb_single.shape

    # MASK-token baseline embedding
    mask_id = tokenizer.mask_token_id
    baseline_embedding = model.distilbert.embeddings(
        torch.tensor([[mask_id]], device=device)
    )[0, 0]

    masker = TextMasker(baseline_embedding=baseline_embedding)

    # Construct GXI-Shapley game
    game = GXIShapleyGame(
        n_players=seq_len,
        model=wrapped_model,
        x=emb_single,
        masker=masker,
        device=device,
        aggregation="signed",
        normalize=False,
        forward_args=(attention_mask,),
        forward_kwargs=None,
    )

    game.feature_index = token_index
    game.target_index = None  # infer predicted NLI class

    # Approximate Shapley values
    approximator = PermutationSamplingSV(
        n=seq_len,
        random_state=random_state,
    )

    ivs = approximator.approximate(
        game=game,
        budget=budget,
        batch_size=batch_size,
    )

    shapley_vals = ivs.values[1:]  # drop null player
    assert shapley_vals.shape[0] == seq_len

    # Build influence table
    df = pd.DataFrame({
        "token_index_j": list(range(seq_len)),
        "token": tokens,
        "shapley_influence_on_CLS": shapley_vals.astype(float),
    })

    df["abs_influence"] = df["shapley_influence_on_CLS"].abs()
    df = df.sort_values("abs_influence", ascending=False).drop(columns="abs_influence")

    df_top = df.head(top_k)

    # Save results
    out_csv = results_dir / f"token_influence_{sample_key}_CLS.csv"
    out_txt = results_dir / f"token_influence_{sample_key}_CLS.txt"

    df_top.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("GXI-Shapley text illustration (NLI, CLS-centered)\n")
        f.write(f"sample_key: {sample_key}\n")
        f.write(f"target_feature: CLS (index 0)\n")
        f.write(f"premise: {premise}\n")
        f.write(f"hypothesis: {hypothesis}\n\n")
        f.write(df_top.to_string(index=False))
        f.write("\n")

    print("Saved outputs:")
    print(f" - {out_csv}")
    print(f" - {out_txt}")


if __name__ == "__main__":
    main()
