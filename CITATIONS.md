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

- **Status:** Applied to the mechanistic research agenda.
- **Consulted:** 2026-08-27
- **Relevant result:** Patching conclusions can change substantially with the corruption method and
  evaluation metric, so the intervention and metric must be chosen for the causal question rather
  than treated as interchangeable defaults.
- **Use here:** Determines the paired corruption design, bidirectional patching, multiple baselines,
  and raw causal metrics in the proposed follow-up. Existing notebook results remain behavioral.
- **Applied in:** Behavioral stopping-point section of
  `notebooks/03_controlled_posterior_behavior.ipynb` and `reports/research_agenda/REPORT.md`.

### `GeigerEtAl2025` — Causal abstraction

Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang,
Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard. “Causal
Abstraction: A Theoretical Foundation for Mechanistic Interpretability.” arXiv:2301.04709v4,
2025 (first version 2023).
[Paper](https://arxiv.org/abs/2301.04709) ·
[DOI](https://doi.org/10.48550/arXiv.2301.04709)

- **Status:** Applied to the mechanistic research agenda.
- **Consulted:** 2026-08-27
- **Relevant result:** A faithful high-level variable should participate in the appropriate causal
  transformations of the low-level model, not merely be decodable from correlated activations.
- **Use here:** Makes posterior log-odds, evidence agreement, and reliability explicit high-level
  variables whose roles must be tested with held-out interchange interventions. Existing notebook
  results remain behavioral.
- **Applied in:** Behavioral stopping-point section of
  `notebooks/03_controlled_posterior_behavior.ipynb` and `reports/research_agenda/REPORT.md`.

### `FalckEtAl2024` — Necessary tests for Bayesian in-context learning

Fabian Falck, Ziyu Wang, and Christopher C. Holmes. “Is In-Context Learning in Large Language
Models Bayesian? A Martingale Perspective.” *ICML*, 2024.
[Paper](https://proceedings.mlr.press/v235/falck24a.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Bayesian learning over exchangeable data must satisfy martingale and
  uncertainty-scaling conditions; tested LLMs systematically violate them.
- **Use here:** Prevents the agenda from treating behavioral success on a small task as evidence
  that ICL is generically Bayesian and motivates exact, variable-level causal tests.
- **Applied in:** `reports/research_agenda/REPORT.md`, motivation and falsification criteria.

### `GuptaEtAl2025` — Behavioral Bayesian updating on coin flips

Ritwik Gupta, Rodolfo Corona, Jiaxin Ge, Eric Wang, Dan Klein, Trevor Darrell, and David M. Chan.
“Enough Coin Flips Can Make LLMs Act Bayesian.” *ACL*, 2025.
[Paper](https://aclanthology.org/2025.acl-long.377/)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** LLMs can track Bayesian-like updates after sufficient simple evidence while
  retaining biased priors; attention magnitude has little relation to the update.
- **Use here:** Motivates separating prior, likelihood update, and posterior, and rejecting
  attention heatmaps as mechanistic evidence.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiments 1 and 3.

### `KimEtAl2025` — Evidence quality and belief

Minsu Kim, Sangryul Kim, and James Thorne. “From Evidence to Belief: A Bayesian Epistemology
Approach to Language Models.” *NAACL*, 2025.
[Paper](https://aclanthology.org/2025.naacl-long.531/)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Responses and confidence do not consistently respect confirmation,
  disconfirmation, irrelevance, and source reliability across elicitation methods.
- **Use here:** Establishes the behavioral gap and motivates an exact, semantically controlled
  reliability intervention rather than source-quality prose alone.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 1 motivation.

### `WiegreffeEtAl2025` — Semantic answer versus answer symbol

Sarah Wiegreffe, Oyvind Tafjord, Yonatan Belinkov, Hanna Hajishirzi, and Ashish Sabharwal.
“Answer, Assemble, Ace: Understanding How LMs Answer Multiple Choice Questions.” *ICLR*, 2025.
[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c248154176c08147e82c0b30961604f7-Abstract-Conference.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Activation patching separates middle-layer answer-symbol selection from
  later amplification, and symbol mechanisms vary with format and model.
- **Use here:** Motivates paired semantic margins and distinct causal tests for candidate preference
  and label routing.
- **Applied in:** `reports/research_agenda/REPORT.md`, shared protocol and Experiment 2.

### `HewittLiang2019` — Probe control tasks

John Hewitt and Percy Liang. “Designing and Interpreting Probes with Control Tasks.”
*EMNLP-IJCNLP*, 2019.
[Paper](https://aclanthology.org/D19-1275/)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** High-capacity probes can learn the target task or memorize incidental input
  properties; selectivity requires matched control tasks.
- **Use here:** Determines grouped OOD splits, low-capacity probes, random-label controls, and the
  rule that probing is only a localization screen.
- **Applied in:** `reports/research_agenda/REPORT.md`, shared protocol.

### `MakelovEtAl2024` — Interpretability illusions from subspace patching

Aleksandar Makelov, Georg Lange, Atticus Geiger, and Neel Nanda. “Is This the Subspace You Are
Looking for? An Interpretability Illusion for Subspace Activation Patching.” *ICLR*, 2024.
[Paper](https://openreview.net/forum?id=Ebt7JgMHv1)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** A subspace intervention can produce the intended behavior by activating a
  dormant pathway rather than revealing the naturally used mechanism.
- **Use here:** Requires necessity, natural interchange, bidirectional dose response, and off-target
  controls before calling a posterior or reliability direction causal.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiments 1 and 2.

### `HannaEtAl2024` — Faithful circuit discovery with EAP-IG

Michael Hanna, Sandro Pezzelle, and Yonatan Belinkov. “Have Faith in Faithfulness: Going Beyond
Circuit Overlap When Finding Model Mechanisms.” *COLM*, 2024.
[Paper](https://openreview.net/forum?id=TZ0CCGDcuT) ·
[arXiv](https://arxiv.org/abs/2403.17806)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** EAP with integrated gradients finds more faithful circuits than first-order
  EAP in studied settings, but circuit overlap is not evidence of faithfulness.
- **Use here:** EAP-IG is restricted to post-localization discovery on 4B, with exact path patching
  and whole-circuit ablation required for validation.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 1 and compute plan.

### `TurpinEtAl2023` — Unfaithful visible reasoning

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. “Language Models Don't Always Say
What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.” *NeurIPS*, 2023.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Models can rationalize answers caused by prompt biases without disclosing
  those causal influences.
- **Use here:** Visible derivations are treated as an experimental condition whose atomic steps
  must be intervened on, not as transparent evidence of the underlying computation.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 2.

### `HalawiEtAl2024` — Correct early state overwritten by late imitation

Danny Halawi, Jean-Stanislas Denain, and Jacob Steinhardt. “Overthinking the Truth: Understanding
How Language Models Process False Demonstrations.” *ICLR*, 2024.
[Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bb63841e1ad12370a34504f15c60db4f-Abstract-Conference.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Early predictions can remain correct under false demonstrations before late
  false-induction heads overwrite them; ablation reduces the harmful effect.
- **Use here:** Supplies a concrete rival mechanism for the notebook-08 depth-30-to-final alias
  collapse and motivates final-block ablation and patching.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 2.

### `WardEtAl2023` — Causal definition of deception

Francis Ward, Francesca Toni, Francesco Belardinelli, and Tom Everitt. “Honesty Is the Best Policy:
Defining and Mitigating AI Deception.” *NeurIPS*, 2023.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Deception is defined through goal-directed causal influence in structural
  causal games, not merely by a false output.
- **Use here:** Requires the strategic-source condition to include an explicit objective and
  reporting policy, and keeps i.i.d. noise distinct from deception.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 3.

### `GolechhaGarrigaAlonso2025` — Decodability without causal control in deception

Satvik Golechha and Adrià Garriga-Alonso. “Among Us: A Sandbox for Measuring and Detecting Agentic
Deception.” *NeurIPS*, 2025.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/105c4de1195135fae4974aa8c5e27bbf-Abstract-Conference.html)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Linear deception probes generalize with AUROC above 95%, while detected SAE
  features fail to steer deception away.
- **Use here:** Reinforces the requirement that source-policy and posterior probes be followed by
  causal listener-side tests.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 3.

### `ZhaoCoulson2026` — Qwen lie expectancy under strategic incentives

Xingyuan Zhao and Seana Coulson. “Graded Expectations: Do Large Language Models Show Human-like
Sensitivity to the Likelihood of Deceptive Speech Acts?” *SCiL*, 2026.
[Paper](https://aclanthology.org/2026.scil-main.46/)

- **Status:** Applied
- **Consulted:** 2026-09-01
- **Relevant result:** Qwen3 continuation likelihoods track human lie expectancy, but post-trained
  models are weaker and strategic-gain or self-protection lies are underpredicted.
- **Use here:** Motivates a controlled Qwen listener-side test where goal-conditioned report
  likelihoods and exact posteriors are available.
- **Applied in:** `reports/research_agenda/REPORT.md`, Experiment 3.

## Considered but not adopted

### `BelroseEtAl2023` — Tuned lens

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella
Biderman, and Jacob Steinhardt. “Eliciting Latent Predictions from Transformers with the Tuned
Lens.” arXiv:2303.08112, 2023.
[Paper](https://arxiv.org/abs/2303.08112)

- **Status:** Considered but not adopted as a core method
- **Consulted:** 2026-09-01
- **Relevant result:** Per-layer affine translators are less biased and more predictive than the
  ordinary logit lens, with causal-basis tests connecting some lens-important directions to model
  behavior.
- **Use here:** Retained only as an exploratory visualization; it predicts the eventual vocabulary
  distribution and cannot establish the proposed Bayesian causal abstraction by itself.
- **Referenced in:** `reports/research_agenda/REPORT.md`, Experiment 2.

## Maintenance rule

When further literature is consulted, add a paper only after reading the primary source and deciding
that it changes the design, analysis, or interpretation. Record the date, specific result used, exact
repository location affected, and whether the idea was applied. Do not reconstruct this ledger from
the bibliography of another paper at the end of the project.
