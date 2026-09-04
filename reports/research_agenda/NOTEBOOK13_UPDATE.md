# Notebook 13 update: locating the missing reliability-sign gate

## Bottom line

[`notebooks/13_noisy_channel_bayesian_agreement_logits.ipynb`](../../notebooks/13_noisy_channel_bayesian_agreement_logits.ipynb)
changes the lead mechanistic hypothesis. Qwen3.5-9B's immediate answer-boundary state contains a
robust **unsigned agreement** signal, but very little evidence that this signal is multiplied by the
sign of source reliability. When the source is wrong more often than right, the direct model still
treats agreement as support. A visible explanation causes the correct sign inversion to appear.

The strongest current working model is therefore:

> The prompt-only pathway accumulates “candidate matches the reports” with a positive gain. The
> below-chance reliability rule is either represented but not composed with that statistic, or is
> not operationalized until visible text explicitly turns matches into likelihoods.

This is more specific than “the model cannot do Bayes” and more cautious than “reasoning reveals a
latent posterior.” The present evidence does not distinguish a missing computation from a routing
failure. It also does not show that the free-form explanation is faithful: the final reasoning-on
logits are measured after the model has often written its likelihood calculation and conclusion.

Both notebook-13 conditions use `enable_thinking=False`. “Reasoning on” means ordinary visible
assistant text before `ANSWER:`; it is not Qwen's native thinking mode. All proposed work below
keeps native thinking disabled.

## What the artifact establishes

The saved 9B run contains 1,600 generations: 10 independently sampled question schedules, all
eight report patterns, five reliability values, two visible-reasoning settings, and both candidate
orders. The two answer numerals are single tokens (`2` is token 17 and `7` is token 22). Candidate
order is averaged before the reported correlations.

For the interior symmetric reliability pair:

| Condition | $r=.1$: correlation / slope | $r=.9$: correlation / slope | Paired correlation $m_{.1},m_{.9}$ | Non-tie generated accuracy |
|---|---:|---:|---:|---:|
| Immediate answer | Spearman .634 / +.225 | Spearman .687 / +.319 | +.940 | 34.2% / 70.0% |
| Visible explanation | Spearman -.859 / -6.830 | Spearman .906 / +7.106 | -.994 | 100% / 100% |

Here a slope is the order-balanced `logit(C1) - logit(C2)` regressed on
$\Delta a=a_{C1}-a_{C2}$. The exact Bayesian slope is
$\log(r/(1-r))$: -2.197 at $r=.1$ and +2.197 at $r=.9$. Thus visible reasoning restores the
correct ordering but produces an overconfident, saturating answer margin; it is not a calibrated
posterior-log-odds readout.

A paired decomposition makes the failure especially clear:

\[
m_{\mathrm{even}}(d)=\frac{m(d,.9)+m(d,.1)}{2},\qquad
m_{\mathrm{odd}}(d)=\frac{m(d,.9)-m(d,.1)}{2}.
\]

The even component is reliability-sign invariant; the odd component is the desired reliability
interaction. A 10,000-resample percentile bootstrap over the 10 question-schedule clusters gives:

| Condition | slope of $m_{\mathrm{even}}$ on $\Delta a$ | slope of $m_{\mathrm{odd}}$ on $\Delta a$ |
|---|---:|---:|
| Immediate answer | .272 [.159, .394] | .047 [.031, .062] |
| Visible explanation | .138 [.106, .168] | 6.968 [5.834, 8.926] |

This supports an unsigned-agreement heuristic in the direct condition. It does not support the
stronger claim that reliability is wholly ignored: the positive slope grows from .225 at $r=.1$
to .452 at $r=1$. A plausible finer hypothesis is that reliability is treated as nonnegative
confidence—weak reliability damps agreement, but never turns it into counterevidence.

### Audit qualifications

- `allow_same=False` removes `SAME` from the answer vocabulary, not from the stimuli. At $r=.5$,
  all 80 scenarios are exact posterior ties. At $r=.1$ and $r=.9$, 20/80 have
  $\Delta a=0$ and are ties. These rows are legitimate for a continuous null test but cannot be
  counted as two-class accuracy observations.
- The endpoint cells are degenerate. At each of $r=0$ and $r=1$, 42/80 scenarios tie and 20/80
  have zero total evidence under the exact model, leaving only 18 non-ties. Endpoints should not be
  mixed with ordinary log-odds calibration.
- The reasoning-on arm has 797/800 compliant completions; three hit the generation cap or lack a
  usable terminal boundary. The immediate arm has 800/800 compliance.
- The effective sampling unit for generalizing across random set schedules is 10, not 80. The eight
  report patterns are exhaustive repeated conditions within each schedule. Uncertainty must be
  clustered by schedule.
- Candidate order is controlled, but candidate identity remains fixed at 2 versus 7. A confirmatory
  bank must cross several single-token numeral pairs or token aliases.

## What the literature changes

The existing citation ledger already identifies the right distinction. Behavioral success is not
enough to claim Bayesian inference: necessary martingale tests find systematic violations
([Falck et al., ICML 2024](https://proceedings.mlr.press/v235/falck24a.html)), while simple
coin-flip tasks can elicit approximately Bayesian updating after enough evidence
([Gupta et al., ACL 2025](https://aclanthology.org/2025.acl-long.377/)). Source-quality effects are
also inconsistent across elicitation methods
([Kim et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.531/)). Notebook 13 contributes
a sharper diagnosis: the model extracts the sufficient agreement statistic but fails the
below-chance sign interaction at the immediate answer boundary.

Three mechanistic lessons determine the next experiments:

1. Answer semantics and answer symbols can be implemented by distinct mechanisms
   ([Wiegreffe et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c248154176c08147e82c0b30961604f7-Abstract-Conference.html)).
   Position balancing should therefore be retained and extended to identity/alias swaps.
2. A plausible explanation is not a faithful process trace. Early answering, inserted mistakes,
   filler, and paraphrase interventions reveal strong task-dependent variation in how much models
   use their written chain of thought
   ([Lanham et al., 2023](https://arxiv.org/abs/2307.13702)). Activation patching in distilled
   reasoning models provides evidence that reasoning-token states can affect answers, but also
   finds larger effects when answer-format fragments recur—exactly the answer-anchoring confound
   that conclusion-free audits should remove
   ([Zhang et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.198/)).
3. Decoding a variable is not a mechanism. Causal abstraction requires the neural state to undergo
   the counterfactual transformations of the proposed high-level variable
   ([Geiger et al., 2025](https://arxiv.org/abs/2301.04709)). Distributed alignment search can find
   non-axis-aligned candidate variables, but held-out interchange accuracy is the test
   ([Geiger et al., CLeaR 2024](https://proceedings.mlr.press/v236/geiger24a.html)). Patching
   metrics/corruptions need multiple baselines
   ([Zhang and Nanda, ICLR 2024](https://arxiv.org/abs/2309.16042)), and a learned subspace can
   activate a dormant rather than naturally used pathway
   ([Makelov et al., ICLR 2024](https://openreview.net/forum?id=Ebt7JgMHv1)).

Qwen3.5-4B and 9B both have 32 language layers arranged as eight repetitions of three Gated
DeltaNet blocks followed by one full-attention block; their hidden widths are 2,560 and 4,096,
respectively ([official 4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[official 9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)). This makes an
unsigned-accumulator-plus-late-gate division of labor testable, but the architecture should guide
the intervention grid only after a residual-stream effect is localized.

## Ranked hypotheses

### H1 — Unsigned agreement accumulator, missing sign composition

The model computes $\Delta a$ but uses a nonnegative reliability gain $h(r)\geq 0$, approximately
$m\approx b+h(r)\Delta a$. Visible text constructs the missing signed product.

Predictions:

- $\Delta a$ transfers across reliability values in prompt-only activations.
- $r$, $1[r>.5]$, or $g(r)$ may be separately decodable, but a held-out causal representation
  of $z^*=\Delta a g(r)$ is absent or unused.
- A direct candidate-evidence prompt that supplies AGREES/DISAGREES relations still fails at
  $r<.5$.
- A fixed audit sentence stating that mismatches are more likely causes the odd interaction to
  appear.

### H2 — Signed posterior exists but is ignored or overwritten

A correct $z^*$ state exists before the answer, but the final answer pathway reads the stronger
unsigned-agreement feature.

Predictions:

- A cross-reliability, cross-alias $z^*$ probe succeeds before the final layer under immediate
  answering.
- Interchanging that state between paired $(\Delta a,r)$ cells gives the exact counterfactual
  answer on held-out histories.
- Late module ablation or routing patches improve low-reliability behavior without erasing
  above-chance behavior.

### H3 — Visible text computes the signed posterior

There is no causally usable $z^*$ at the prompt boundary. It emerges only after the model writes
the channel inversion and candidate-specific likelihood factors.

Predictions:

- Early-answer margins change sign immediately after the first explicit “mismatch is more likely”
  or likelihood-comparison step.
- Corrupting one audit fact changes the final margin by the corresponding likelihood increment;
  neutral paraphrases do not.
- Conclusion-free teacher-forced audits rescue direct behavior, while length-matched filler does
  not.

### H4 — Reliability-language parsing, not Bayesian composition, is the bottleneck

The model treats the phrase “reliability 0.1” as weak positive trust but can directly use an
equivalent “wrong 9 times out of 10” instruction.

Prediction: direct odd interaction changes sharply across truth-rate, error-rate, “usually lies,”
and explicit likelihood-weight renderings even though the exact task is unchanged. This is a
lower-priority hypothesis because notebook 13 already says both that the source is flipped with
probability .9 and that a mismatch contributes $1-r$; any rescue would be a framing/compilation
effect, not recovery of omitted information.

## Experiment 1 — A clean sign-gate factorial

This is the immediate next experiment and should precede activation work.

Construct every question so exactly one of the two candidates belongs to its set. Use odd
$K\in\{1,3,5\}$. Then $\Delta a$ is odd and cannot be zero, eliminating structural ties without
discarding observations. Use only interior symmetric pairs

\[
(r,1-r)\in\{(.1,.9),(.25,.75),(.4,.6)\}.
\]

Keep $r=.5$ as a separately labeled continuous-null suite; omit $r=0,1$ from the primary test.
Sample at least 32 independent question schedules, exhaust the report patterns within each, and
cross candidate order. Use four verified single-token candidate pairs on a confirmation subset.

Run a four-step computational ladder, all with native thinking disabled and no stated winner:

1. the raw set-membership prompt from notebook 13;
2. a reduced table giving candidate-specific AGREES/DISAGREES relations;
3. a candidate-specific factor list, such as `C1: .9, .1, .9`, which instantiates the reliability
   rule but does not aggregate it;
4. the two exact likelihood products or log-likelihood totals, which leaves only comparison and
   answer routing.

Within the raw and relation conditions, cross a truth-rate rendering with the equivalent error-rate
rendering (“SOURCE is wrong with probability .9”) as a secondary framing control.

Primary model:

\[
m=\beta_0+\beta_a\Delta a+\beta_g g(r)+\beta_{ag}\Delta a g(r)
  +\text{paired nuisance effects}.
\]

The Bayesian signature is $\beta_a=0$, $\beta_{ag}>0$, odd symmetry
$m(d,r)=-m(d,1-r)$, and monotonic magnitude in $|g(r)|$. The current direct result instead has a
large positive $\beta_a$ and a small interaction. Report the even and odd components directly,
along with calibration slope/intercept, conditional candidate probability, valid-answer mass,
greedy accuracy, and position/identity effects. Bootstrap schedules, not rendered rows.

Interpretation:

- Rescue by AGREES/DISAGREES localizes failure to membership/report binding.
- Rescue by factor lists localizes failure to applying signed reliability weights to already
  computed agreement relations.
- Rescue by likelihood totals localizes failure to aggregation; failure even with totals points to
  comparison or answer routing.
- A large truth-rate versus error-rate difference identifies a framing/semantic-compilation
  problem, but should not be described as missing task information.

## Experiment 2 — Find and causally test the missing interaction

Use 4B for discovery on the non-tie factorial. At the final prompt token and a small set of aligned
sites—the end of the reliability rule, each report, the end of each candidate-evidence row, and the
answer prefix—capture residual pre/post across all 32 layers. Probe five preregistered variables:

- $\Delta a$;
- $g(r)$ and the binary sign $1[r>.5]$;
- $|g(r)|$;
- the Bayesian product $z^*=\Delta a g(r)$;
- the heuristic product $z_h=\Delta a |g(r)|$ or simply $\Delta a$.

Use grouped splits that hold out entire schedules, one reliability pair, one prompt rendering, and
candidate identities. A linear probe is only a localization screen. The decisive test is a crossed
2-by-2 interchange design with $\Delta a\in\{-d,+d\}$ and
$g(r)\in\{-g,+g\}$: patch the candidate variable, reliability variable, or proposed product, and
ask whether the answer margin matches the high-level counterfactual in every quadrant.

Start with whole residual-state patches at the prompt boundary. If an effect exists, narrow to the
reliability-line state and report-summary states, then to module outputs. Only after a held-out
effect survives should DAS search for a low-dimensional distributed subspace. Require both patch
directions, resampled and mean baselines, random subspaces, off-target controls, and raw margin
changes. This distinguishes a naturally used sign gate from a steerable dormant direction.

The key layerwise plot is not ordinary logit lens accuracy. Plot the slopes of
$m_{\mathrm{even}}$ and $m_{\mathrm{odd}}$ across residual depth. H1 predicts an early/middle
even component and no prompt-only odd component. H2 predicts an odd component that appears and is
then lost or bypassed near the answer.

## Experiment 3 — Is visible reasoning computation or answer anchoring?

Free-form reasoning is useful behaviorally but poorly aligned for interventions. Replace it with a
teacher-forced, fixed-format audit that never states the winning candidate before `ANSWER:`:

1. candidate membership bits;
2. report-match bits;
3. the reliability rule (“match weight” and “mismatch weight”);
4. candidate log-likelihood totals;
5. `ANSWER:`.

Score candidate logits after each prefix. Then apply minimal paired corruptions:

- flip exactly one membership or match bit;
- swap the two reliability weights;
- change only the sentence interpreting $r<.5$;
- swap the two likelihood totals;
- add a correct or incorrect explicit conclusion as a separate final condition;
- paraphrase correct audit statements and use length-matched neutral text.

In parallel, perform early answering on the original free explanations: truncate at sentence
boundaries, append a fresh neutral `ANSWER:` suffix, and score both candidates. Annotate the first
sentence that states the reliability inversion, completes both candidate likelihoods, or announces
a conclusion.

Evidence for genuine text-mediated computation is a graded, normatively signed response to atomic
audit corruptions before any conclusion is named. If only the explicit conclusion controls the
answer—or repeating an answer-format fragment dominates—the visible trace is primarily an answer
commitment/induction cue. This experiment directly implements the intervention logic of Lanham et
al. while exploiting exact per-step effects unavailable in ordinary benchmarks.

## Experiment 4 — Does Gated DeltaNet carry the wrong sufficient statistic?

Run only after Experiment 2 identifies a causal residual site. Use report sequences of lengths
$K=1,3,5,7$, recording states after each observation. At step $t$, compare two candidate state
models:

\[
A_t=\sum_{i\leq t}\Delta a_i,
\qquad
Z_t=g(r)A_t.
\]

Test whether states with the same $A_t$ but different report orders are interchangeable, and
whether changing only $r$ turns $A_t$ into $-Z_t$ below .5. Patch Gated DeltaNet token-mixer
outputs and full-attention outputs separately only in the causal layer window. A particularly clean
outcome would be that Gated DeltaNet carries order-invariant $A_t$, while a later attention/MLP
path should apply—but fails to apply—the reliability sign.

Do not infer this division of labor from decodability or the 3:1 architecture pattern. Require
equal-state interchange, opposite-state swaps, necessity ablations, and preservation of unrelated
set-membership performance.

## RTX 5090 execution plan

The installed RTX 5090 has 32,607 MiB. Both local checkpoints are BF16 (about 8.8 GB for 4B and
19 GB for 9B) and have already run in this environment. Use one model at a time, Transformers
generation, greedy decoding, `ENABLE_MTP=False`, and `enable_thinking=False`. Pin the checkpoint
revisions and use the same tokenizer serialization for both models.

1. **Behavioral replication:** run the exact notebook-13 sweep on 4B. Then run the non-tie
   sign-gate factorial on 4B and 9B. This is cheap because the primary condition emits only the
   answer line.
2. **4B discovery capture:** store residual pre/post only at preregistered token positions. For 512
   prompts, 33 depths, eight positions, BF16 residual storage is roughly 0.7 GB at width 2,560.
   Avoid all-token/all-stream dumps.
3. **4B causal scan:** scan layers with whole residual patches, lock the sites and metrics, then
   test module outputs only in the surviving window.
4. **9B confirmation:** repeat only the locked sites and interventions. The analogous selected-
   position residual cache is roughly 1.1 GB at width 4,096; use capture batch size 1–2 and stream
   tensors to CPU.
5. **Reasoning mediation:** generate free explanations only for a balanced diagnostic subset.
   Teacher-forced audit prefixes give aligned, reproducible states and are much cheaper than
   repeatedly generating 700-token explanations.

At the time of this audit, only about 3.6 GB of GPU memory was free, so an existing process is
occupying most of the card. Finish or stop that process before loading another checkpoint; do not
attempt concurrent 4B and 9B runs.

## Decision rules

1. If the 4B replication does not show the positive low-reliability direct correlation, treat the
   effect as scale-specific and do not pool mechanisms across sizes.
2. If the non-tie factorial reproduces a dominant even component in both sizes, promote H1 and
   begin causal localization.
3. If prompt-only $z^*$ passes held-out interchange tests, promote H2 and localize the late routing
   failure.
4. If $z^*$ appears only after conclusion-free audit steps and atomic corruptions have exact
   signed effects, promote H3.
5. If wording changes rescue the odd interaction, promote H4 and study the semantic compilation
   of source descriptions before searching for a generic Bayesian circuit.
6. If a probe succeeds but interchange fails, report decodability only and stop subspace/circuit
   claims.

The first publishable target is therefore not “Qwen contains a Bayesian direction.” It is a causal
answer to a narrower question: **where does an unsigned agreement statistic acquire—or fail to
acquire—the sign of source reliability?**
