# Proposal for Contribution: Integrating SIMBA‑UQ (Similarity‑Based Aggregation for Uncertainty Quantification) into Mellea

## Summary
This proposal describes the integration of SIMBA‑UQ (Similarity‑Based Aggregation for Uncertainty Quantification) into the Mellea generative programming library. SIMBA‑UQ is a black‑box, non‑verbalized framework that estimates confidence in LLM generations by aggregating pairwise similarities across multiple sampled outputs, avoiding reliance on model internals or self‑reported probabilities (i.e., logprobs).

The proposed contribution introduces SIMBA‑UQ as a first‑class uncertainty module in Mellea, implemented using existing abstractions such as sampling strategies and verifiers. The integration aims to provide calibrated, model‑agnostic confidence estimates that can be used for selection, abstention, fallback routing, and other reliability‑oriented generative workflows.

Our papers:
- EMNLP 2025: [SIMBA-UQ: Similarity-based Aggregation for Uncertainty Quantification](https://arxiv.org/pdf/2510.13836)
- UAI 2025: [The Consistency Hypothesis for Uncertainty Quantification in LLMs](https://arxiv.org/pdf/2506.21849)

## Overview of SIMBA-UQ
SIMBA‑UQ defines a general pipeline for confidence estimation:

- **Sampling**: Generate multiple outputs for the same input query, typically using temperature‑based sampling.
- **Similarity Computation**: Compute pairwise similarities between generated outputs using a task‑appropriate similarity metric.
- **Aggregation**: Aggregate similarities for each output to produce a confidence score interpreted as the probability of correctness.

The framework unifies and generalizes a broad class of consistency‑based uncertainty quantification methods and supports both unsupervised and lightly supervised instantiations.

## Proposed Algorithms to Integrate
The contribution proposes integrating three concrete SIMBA‑UQ algorithms, each mapping cleanly to Mellea abstractions.

### Simple Similarity Aggregation (Unsupervised)
**Description**: Confidence for a generation is computed as the arithmetic mean of its pairwise similarities to all other sampled generations.

**Properties**:
- Fully black‑box
- No training data required
- Deterministic aggregation
- Low computational overhead

### Bayesian Similarity Aggregation (Lightweight Supervised)
**Description**: A Bayesian posterior probability of correctness is computed by treating similarities as evidence and modeling conditional similarity distributions (e.g., using Beta distributions).

**Properties**:
- Requires a small labeled calibration set
- Produces probabilistic, interpretable confidence estimates
- Remains black‑box with respect to the LLM

### Aggregation by Classification (Supervised)
**Description**: Confidence estimation is formulated as a probabilistic classification task. Features are derived from similarity values (e.g., all pairwise similarities or summary statistics), optionally augmented with generative scores. Random forest classifiers have been shown to perform particularly well.

**Properties**:
- Strong empirical performance on calibration metrics
- Flexible feature design
- Naturally extensible to task‑specific signals


## Proposed Design and API Integration
The integration is designed to reuse existing Mellea concepts rather than introduce new primitives.

### Sampling Integration
SIMBA‑UQ relies on multiple generations per query. This is naturally supported via Mellea’s existing sampling strategies (e.g., temperature sweeps).

### Similarity Metrics
The contribution introduces a pluggable similarity interface, with initial support for:

- Token‑based similarities (e.g., Jaccard)
- Text overlap metrics (e.g., ROUGE‑L)
- Others

The design allows future extensions to embedding‑based or task‑specific similarity metrics.

### Aggregators and Verifiers
SIMBA‑UQ aggregators are exposed as verifiers that:

- Consume multiple sampled outputs
- Produce a confidence score per output
- Can be used for ranking, selection, or threshold‑based decision‑making

```python
from mellea.sampling import TemperatureSweep
from mellea.uq import MeanSimilarityAggregator
from mellea.uq.similarity import JaccardSimilarity

m = start_session()
result = m.instruct(
    "Summarize the following document in one sentence: ...",
    requirements=[
        "Keep it under 50 words",
    ]
    with_confidence=True,
    strategy=TemperatureSweep(
        temperatures=[0.25, 0.5, 0.75],
        n_per_temp=4
    ),
    verifier=MeanSimilarityAggregator(
        similarity=JaccardSimilarity()
    )
)

# Output best result and confidence
best_output = result.select_max_confidence()
confidence = result.confidence(best_output)
```
