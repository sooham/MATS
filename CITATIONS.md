# Research citations

This is an annotated ledger of primary papers that materially informed an experiment or
interpretation in this repository. Search results and papers mentioned only second-hand are not
included. Status values mean:

- **Applied** — directly determines an implemented design or analysis choice.
- **Background** — informs interpretation but is not yet operationalized.
- **Considered but not adopted** — reviewed, but its method was intentionally not used.

## Applied

### `LieberumEtAl2023` — Multiple-choice label mechanisms

Tom Lieberum, Matthew Rahtz, János Kramár, Neel Nanda, Geoffrey Irving, Rohin Shah, and
Vladimir Mikulik. “Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice
Capabilities in Chinchilla.” arXiv:2307.09458, 2023.
[Paper](https://arxiv.org/abs/2307.09458) ·
[DOI](https://doi.org/10.48550/arXiv.2307.09458)

- **Status:** Applied
- **Consulted:** 2026-08-27
- **Relevant result:** The paper separates knowing answer content from emitting the corresponding
  multiple-choice label and finds mechanisms sensitive to enumeration and label format. Its prompt
  mutations also show why tokenization and randomized answer labels require explicit validation.
- **Use here:** Motivates all six semantic-to-A/B/C mappings, recording the full-vocabulary greedy
  token and A/B/C probability mass, the Stage-0 elicitation control, and treating label mapping as a
  repeated measurement rather than evidence of six independent decisions.
- **Applied in:** `notebooks/03_controlled_posterior_behavior.ipynb`, model-scoring and compliance
  sections.

### `ZhangNanda2024` — Activation-patching methodology

Fred Zhang and Neel Nanda. “Towards Best Practices of Activation Patching in Language Models:
Metrics and Methods.” *International Conference on Learning Representations (ICLR)*, 2024.
arXiv:2309.16042.
[Paper](https://arxiv.org/abs/2309.16042) ·
[DOI](https://doi.org/10.48550/arXiv.2309.16042)

- **Status:** Background; mechanistic follow-up is explicitly deferred.
- **Consulted:** 2026-08-27
- **Relevant result:** Patching conclusions can change substantially with the corruption method and
  evaluation metric, so the intervention and metric must be chosen for the causal question rather
  than treated as interchangeable defaults.
- **Use here:** Records methodological constraints for a possible later project. The present work
  performs no activation patching and makes no localization claim.
- **Referenced in:** Behavioral stopping-point section of
  `notebooks/03_controlled_posterior_behavior.ipynb`.

### `GeigerEtAl2025` — Causal abstraction

Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang,
Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard. “Causal
Abstraction: A Theoretical Foundation for Mechanistic Interpretability.” arXiv:2301.04709v4,
2025 (first version 2023).
[Paper](https://arxiv.org/abs/2301.04709) ·
[DOI](https://doi.org/10.48550/arXiv.2301.04709)

- **Status:** Background; mechanistic follow-up is explicitly deferred.
- **Consulted:** 2026-08-27
- **Relevant result:** A faithful high-level variable should participate in the appropriate causal
  transformations of the low-level model, not merely be decodable from correlated activations.
- **Use here:** Clarifies why a future decodability result would not by itself establish mechanism.
  The present experiment remains behavioral and does not train representation probes.
- **Referenced in:** Behavioral stopping-point section of
  `notebooks/03_controlled_posterior_behavior.ipynb`.

## Maintenance rule

When further literature is consulted, add a paper only after reading the primary source and deciding
that it changes the design, analysis, or interpretation. Record the date, specific result used, exact
repository location affected, and whether the idea was applied. Do not reconstruct this ledger from
the bibliography of another paper at the end of the project.
