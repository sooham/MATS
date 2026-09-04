# Mechanistic research agenda: Bayesian evidence grounding under unreliable and strategic sources

> **Notebook 13 update (2026-09-04):** The full Qwen3.5-9B reliability sweep shows a robust
> reliability-invariant agreement signal under immediate answering and a correctly signed signal
> only after visible explanation. The focused interpretation, revised hypotheses, non-tie
> factorial, and RTX 5090 execution plan are in
> [`NOTEBOOK13_UPDATE.md`](NOTEBOOK13_UPDATE.md). That update supersedes the ordering and immediate
> next steps below; this document remains the broader three-project agenda.

## Executive recommendation

The project should ask one overarching question:

> When Qwen3.5 makes a bad decision from an unreliable source, did it fail to extract the
> evidence, infer the source likelihood, compose the likelihood with its prior, or route an
> already-correct posterior into the answer?

The current notebooks make this unusually tractable. The normative computation is exact, the
inputs can be counterfactually paired, and the answer-surface failure is already characterized.
The most promising sequence is:

1. **Test a causal factorization of the posterior update** into report agreement and source
   reliability.
2. **Determine whether the direct-answer failure is inference or readout**, and whether visible
   work creates the posterior rather than merely revealing it.
3. **Test for a recurrent Bayesian sufficient state and then change the source from stochastic to
   strategic**, while keeping an exact listener-side posterior.

These are stronger than a generic search for a “Bayes direction.” Each proposes an explicit
high-level causal model and can be rejected by held-out interchange interventions. All model calls
must use `enable_thinking=False`. The visible-derivation arm in Experiment 2 is ordinary generated
text, not Qwen thinking mode.

## What the existing experiments establish

| Evidence | Result | Consequence for the agenda |
|---|---|---|
| Notebook 01 | Raw A/B/C scoring is dominated by label priors. Counterbalancing, warnings, and prompt rewrites move the collapse but do not recover the missing semantic class. The raw-evidence symmetry probe never predicts the right candidate. | Accuracy on one answer vocabulary is not a valid mechanistic target. Semantic margins must be estimated with paired label swaps and total answer mass must be reported separately. |
| Notebook 05 | A single no-thinking call with a visible concise derivation reaches 107/112 on held-out examples (95.5%; 58/60 on non-ties). Answer-only variants are much worse. | The model has the task capability, but the computation may occur while producing visible intermediate tokens. A correct rationale is not evidence that the same computation existed in the prompt-only forward pass. |
| Notebook 06 | Token-exact filler followed by `FINAL: ` does not reliably improve performance. Filler identity and count change the readout; the selected underscore/F=11 setting validates at 62.5% but never predicts ties. Because validation non-ties contain 25 target-2 versus 15 target-7 rows, an always-2 rule also scores 62.5%; balanced candidate accuracy is 63.3%. | Filler tokens are a readout perturbation, not “latent reasoning time.” Use balanced effects and do not use filler length as a compute axis. |
| Notebook 07 | Scaling the domain leaves candidate-only accuracy at 50.8% and produces an extreme surface bias: 2,799/2,816 predictions are `Y`, with no ties. Reliability is part of the RNG seed, so reliability cells do not replay the same evidence. | Candidate identity, position, and label must be independently crossed. Larger domains do not by themselves create a reasoning test, and future reliability contrasts must be paired. |
| Notebook 08 | Under alias swaps, underscore has 1.9% semantic consistency and 98.1% surface-token invariance; period has 12.5% and 70.6%, respectively. More sharply, aggregate semantic consistency rises to 78.4% at residual depth 30 and collapses to 7.2% at the final depth; for underscore it falls from 85.0% to 1.9%. Depth-30 semantic accuracy is only 55.5%. | The final block is a sharply motivated causal target for alias overwrite, but the earlier alias-equivariant state is not yet an accurate posterior. |
| Notebook 09 | Visible derivation is near ceiling for 4B/9B/27B, but direct no-scratchpad non-tie accuracy is 51.7%/55.0%/63.3%. Filler-grid raw means are 0.508/0.530/0.618 and balanced means are 0.505/0.514/0.561. Across 84,840 filler-grid decisions and 336 direct held-out decisions, no model ever selects `=`. | Scale helps modestly in the direct condition but does not remove the interface problem. Tie measurement is broken, and the 27B result is confounded by GPTQ-Int4 and backend differences. |
| Notebook 13 | For 9B immediate answers, agreement/logit correlation stays positive from $r=0$ through $r=1$, including Spearman .634 at $r=.1$; paired $r=.1/.9$ margins correlate +.940. Visible explanations invert the low-reliability slope and make the paired margins correlate -.994. | The lead target is now the missing interaction between agreement and reliability sign. Locate the reliability-invariant and signed components separately, then test them with crossed interchange interventions. |

A new, no-GPU screen joined the existing direct F=0 rows to the exact analytic target. For the 64
interior-reliability examples per model, Spearman correlation between the final `2`-minus-`7` logit
margin and exact posterior log-odds was 0.104 for 4B, 0.137 for 9B, and 0.211 for 27B. Simple
row-bootstrap 95% intervals were [-0.165, 0.363], [-0.127, 0.387], and [-0.077, 0.465]. Among the
52 nonzero-margin targets, sign accuracy was 46.2%, 48.1%, and 57.7%. This weakens the easiest
version of “latent Bayes at the output,” but it does not test intermediate, nonlinear, or causally
unused representations.

The reliability split is more diagnostic than aggregate scale. On interior non-ties, accuracy above
one-half reliability rises from 57.7% to 69.2% to 92.3% across 4B/9B/27B, while accuracy below
one-half falls from 46.2% to 38.5% to 30.8%. Agreement with the incorrect “more matches wins” rule
below one-half rises from 53.8% to 61.5% to 69.2%. Scale may be strengthening evidence matching
without strengthening the sign gate. At the endpoints, 16 paired `r=1`/`r=0` visible-derivation
examples have identical targets: every model is 16/16 at `r=1`, but 4B/9B/27B fall to 12/16,
13/16, and 14/16 at `r=0`. This is unusually clean evidence for an inversion-composition failure.

Several uncertainty estimates in the notebooks are anti-conservative for a mechanistic claim.
Notebook 05's 112 test rows contain only two independent question schedules; notebook 06 selects
among 505 measurements of the same 56 development examples; notebook 08 repeats 16 base examples
across 40 prompt conditions and 33 depths. Future intervals and tests must cluster by independent
schedule/history, and discovery versus confirmation must be separated at the level of those
clusters.

The literature makes the gap precise. Behavioral ICL is not generically Bayesian under necessary
martingale tests ([Falck et al., ICML 2024](https://proceedings.mlr.press/v235/falck24a.html)),
although simple coin-flip behavior can approach a discounted Bayesian filter after enough evidence
([Gupta et al., ACL 2025](https://aclanthology.org/2025.acl-long.377/)). Models also respond
inconsistently to evidence quality ([Kim et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.531/)).
None of these studies separates the internal stages above. Mechanistic work shows that answer-symbol
selection can be distinct from semantic answering ([Wiegreffe et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c248154176c08147e82c0b30961604f7-Abstract-Conference.html)),
but decodability or steerability alone is not causal evidence
([Hewitt and Liang, EMNLP 2019](https://aclanthology.org/D19-1275/);
[Makelov et al., ICLR 2024](https://openreview.net/forum?id=Ebt7JgMHv1)).

## Shared formalism and experimental discipline

For candidates \(x\) and \(y\), let \(a_x\) and \(a_y\) be the numbers of reports that agree with
the truthful answers implied by each candidate. Under the current symmetric channel and a uniform
prior,

\[
z^* = \log \frac{P(x\mid D)}{P(y\mid D)}
    = (a_x-a_y)\,g(r),\qquad
g(r)=\log\frac{r}{1-r}.
\]

This gives three high-level variables with exact interventions: the evidence statistic
\(\Delta a=a_x-a_y\), the reliability gain \(g(r)\), and posterior log-odds \(z^*\). A later
nonuniform-prior extension adds \(z_0\) as \(z^*=z_0+\Delta a\,g(r)\).

Every experiment should follow these rules:

- Reuse identical raw histories across reliability values; do not resample reports when changing
  \(r\). Include paired \(r\) and \(1-r\), plus \(r=0.5\), so the normative answer must invert or
  vanish.
- Cross candidate identities, candidate order, semantic-to-token mapping, prompt paraphrase, and
  source name. Split by the entire base history, never by rendered row.
- Use a paired semantic margin that cancels first-order label bias. If one prompt maps \(x\to X\)
  and \(y\to Y\), and its pair swaps the mapping, use
  \[
  m_{sem}=\tfrac12[(\ell_X-\ell_Y)_{x\to X}+(\ell_Y-\ell_X)_{x\to Y}].
  \]
- Report raw semantic margin, conditional candidate probability, total valid-answer mass, greedy
  accuracy, per-class recall, tie and invalid rates, calibration slope/intercept, rank correlation,
  Brier/log score, and paired perturbation effects. Do not collapse them into one accuracy.
- Cache only preregistered positions: the end of the reliability statement, each report, the end of
  the question, and the answer-prefix token. At each site begin with residual pre/post, module
  output, and MLP output. Only inspect Q/K/V, full-attention patterns, or Gated DeltaNet gates and
  recurrent state after a coarse causal effect appears.
- A probe is a localization screen. Use ridge/linear probes with train-only normalization, grouped
  cross-validation, held-out reliability and template transfer, random-label controls, and
  reliability-only/evidence-only baselines. A representation claim requires a held-out causal test.
- For patching, run both source-to-base and base-to-source directions, report raw changes in
  \(m_{sem}\), and use both resampled counterfactual and within-cell mean baselines. Activation-
  patching conclusions are sensitive to metric and corruption choice
  ([Zhang and Nanda, ICLR 2024](https://openreview.net/forum?id=9eJv5PS27Q)).

## Experiment 1 — Is reliability a causal evidence gate?

### Question and hypothesis

**Question:** Does Qwen implement the actual factorization \(z^*=\Delta a\,g(r)\), or does it use
lexical “trust/distrust” heuristics?

**Hypothesis:** Report agreement and reliability are separately represented, then composed into a
signed evidence update in a small set of pathways. Swapping the reliability representation while
holding the observations fixed should produce the quantitative counterfactual associated with the
new \(g(r)\): magnitude should change monotonically, the preference should vanish at 0.5, and its
sign should reverse below 0.5.

This is important because it distinguishes genuine source-sensitive evidence grounding from an
answer that happens to be correct. It is timely because the present data already show that explicit
low reliability is behaviorally brittle, while the field has behavioral evidence-quality studies
but no causal decomposition of a continuous Bayesian sufficient statistic in an instruction-tuned
LLM.

### Dataset

Start from `controlled_posterior.py`, rather than the natural-distribution generator used by
notebook 07. It already supplies exact rational posteriors, a controlled N=2/N=4 accumulation
ladder, and a fixed N=8 bank with 32 independent schedules, reliability replays, candidate
reversals, partitions, and explicit heuristic labels (1,280 examples and 5,120 probes). Derive a
mechanistic subset of at least 256 independent base transcript skeletons, stratified over
\(K\in\{1,3,5,8\}\), \(\Delta a\), report balance, candidate order, and set-membership pattern.
Render each with
\(r\in\{0.1,0.3,0.45,0.5,0.55,0.7,0.9\}\), two paired answer mappings, and two preregistered
paraphrases. This is 7,168 inexpensive behavioral forwards. Use a balanced 512-prompt subset for
activation discovery and 128 held-out matched pairs for interventions. Reserve entire history
skeletons and one paraphrase for test. Analyze \(r=0\) and \(r=1\) separately because finite
log-odds and ordinary calibration metrics do not apply there.

### Method

1. Establish behavioral odd symmetry and monotonicity in the paired semantic margin.
2. At each selected position and layer, probe \(\Delta a\), \(g(r)\), and \(z^*\) separately.
   Cross-test the \(z^*\) probe on unseen reliabilities, histories, candidate numerals, label pairs,
   and paraphrases.
3. Patch reliability-statement residuals between prompts that differ only in \(r\). Test whether
   the causal change in semantic margin tracks \(\Delta z^*=\Delta a[g(r')-g(r)]\). Patch report
   states between prompts differing in one report and test the exact one-step update.
4. If whole-residual patches work, learn the smallest reliability/posterior subspace using
   distributed alignment search and evaluate held-out interchange-intervention accuracy. Require
   necessity, sufficiency, bidirectional dose response, and low off-target damage; steering alone
   is insufficient evidence of natural use.
5. Localize the surviving effect to residual-to-module paths. Qwen3.5 has three Gated DeltaNet
   layers for every full-attention layer
   ([official Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)),
   making a concrete division-of-labor test possible: do
   recurrent layers carry agreement while full-attention blocks bind the reliability statement to
   candidate identity? Use EAP-IG on 4B only as a discovery approximation, then validate top paths
   with exact patching and circuit ablation. EAP-IG circuits must be evaluated for faithfulness,
   not just overlap ([Hanna et al., COLM 2024](https://openreview.net/forum?id=TZ0CCGDcuT)).

### Falsification and interpretation

- **Strong support:** all three variables transfer across nuisance factors, and reliability/report
  interchanges produce the high-level counterfactual on held-out examples.
- **Partial support:** \(\Delta a\) and \(g(r)\) decode, but the product does not, or reliability
  swaps behave categorically rather than quantitatively. This identifies composition as the
  bottleneck.
- **Reject:** posterior decoding disappears under grouped/OOD splits, or a steerable direction
  fails necessity and natural interchange tests.

## Experiment 2 — Is the posterior latent, overwritten, or created by visible work?

### Question and hypothesis

**Question:** In a prompt-only forward pass, is a correct semantic posterior present before the
surface-token circuit takes over, or is the posterior computed only as the model writes a visible
derivation?

**Hypothesis:** A label-invariant posterior state exists in middle layers under no-scratchpad
inference, but a later symbol-selection pathway overwrites or fails to route it. The rival hypothesis
is that no such state exists until visible membership checks create it autoregressively.

This matters for latent-knowledge claims and for the interpretation of notebook 05. Chain-of-thought
can rationalize prompt-induced answers ([Turpin et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html)),
so 95.5% accuracy with a plausible derivation does not show that prompt-only activations contained
the answer. Conversely, work on false demonstrations shows that correct early predictions can be
overwritten later ([Halawi et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bb63841e1ad12370a34504f15c60db4f-Abstract-Conference.html)).

### Conditions

Use the same held-out histories in four no-thinking conditions:

1. immediate one-token answer at `FINAL: `;
2. model-generated visible concise derivation, matching notebook 05;
3. a teacher-forced, fixed-format correct audit containing only atomic candidate memberships and
   report comparisons;
4. minimally corrupted audits in which exactly one membership or SAME/DIFFERENT field is wrong,
   plus length- and token-matched neutral controls.

Cross at least four verified single-token label pairs and both mappings per pair. The primary set
should exclude ties; a separately powered zero-log-odds set tests tie recognition.

### Method

1. Train a low-capacity semantic \(z^*\) probe on one set of answer symbols and test on unseen
   symbols, candidate values, templates, and filler conditions. Compare prompt-only trajectories
   with trajectories after each visible audit item.
2. Interchange-patch the putative semantic state between matched histories with opposite
   posteriors. Separately patch label-mapping states between alias-swapped prompts. A useful
   decomposition predicts that the first intervention changes which candidate is preferred, while
   the second changes only which symbol expresses that preference.
3. Begin late localization at the final block: notebook 08's alias semantic consistency falls from
   78.4% at residual depth 30 to 7.2% at depth 32/final (85.0% to 1.9% for underscore). Patch the
   final full-attention and MLP branches separately, then their inputs and selected heads. Test
   whether ablation improves semantic consistency without erasing the only 55.5%-accurate depth-30
   candidate signal. Evaluate the proposed semantic and symbol subcircuits jointly for necessity,
   sufficiency, and faithfulness.
4. In the teacher-forced arm, corrupt one atomic audit fact at a time. If visible work is causal,
   the answer-margin change should have the sign and approximate magnitude predicted by that
   report's likelihood increment. Paraphrasing a correct audit should have a much smaller effect.
5. Use a tuned lens only as an exploratory visualization; ordinary logit lens is already known to
   be biased at intermediate layers ([Belrose et al.](https://arxiv.org/abs/2303.08112)). Do not
   make the new Jacobian-lens/J-space method a dependency for the core result: it is a recent
   preprint, its readout is oriented toward verbalizable single-token concepts, and continuous
   log-odds are a poor match.

### Falsification and interpretation

- **Latent-and-overwritten:** cross-symbol posterior decoding and causal interchange both work in
  prompt-only states; late symbol-path interventions selectively restore semantic behavior.
- **Computed-during-visible-work:** prompt-only causal tests fail, but \(z^*\) emerges incrementally
  after audit items and atomic corruptions produce the normative incremental effect.
- **Rationalization:** visible derivations remain fluent when corrupted, their stated calculations
  do not mediate the answer, or label-path interventions explain both direct and derivation arms.
- **No strong mechanism claim:** a probe decodes \(z^*\) but its subspace is neither necessary nor
  naturally interchangeable.

The preliminary final-logit screen makes the first outcome uncertain, which is a virtue: this is a
real discriminator rather than an experiment designed only to confirm a favored story.

## Experiment 3 — Does recurrent state implement Bayesian accumulation, and can it model a strategic source?

### Question and hypothesis

**Question:** Does Qwen3.5 compress a sequence of reports into a source-model-conditioned
sufficient statistic, and does that mechanism generalize when reports are generated to manipulate
the listener rather than by i.i.d. noise?

**Hypothesis:** Gated DeltaNet recurrence carries an approximately order-invariant cumulative
log-likelihood ratio. In the i.i.d. condition it should support posterior-equivalent state swaps.
Under a strategic policy, either the same accumulator is correctly gated by a goal/policy
representation or the model falls back to a scalar “trustworthy/untrustworthy” heuristic.

This is the most architecture-specific and safety-relevant hypothesis. Most mechanistic evidence-
accumulation work studies ordinary attention, whereas Qwen3.5's recurrent state is an unusually
direct candidate for a running sufficient statistic. Deception research largely studies the
speaker's behavior or detects lies; it does not test whether a listener applies the right likelihood
for a goal-directed reporting policy. Importantly, a false report is not by itself deception: the
strategic condition must include a source objective and causal policy
([Ward et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract.html)).

### Phase A: sufficient-state test

Use the existing N=2/N=4 accumulation ladder as the first causal screen, then generate histories
with \(K=1\ldots 8\), including multiple report orders and lexically different question sets that
yield the same \(\Delta a\) and \(z^*\). Record residuals and selected Gated
DeltaNet recurrence/gate states after each report.

- Decode the running \(\Delta a_t\) and \(z_t^*\), testing longer-K and permutation OOD splits.
- Patch a state after report \(t\) into another history, then append the same remaining reports. A
  true sufficient state predicts the same downstream answer for posterior-equivalent swaps and the
  exact shifted answer for posterior-different swaps.
- Contrast Gated DeltaNet and every-fourth full-attention layers using output patching, not
  attention heatmaps. Attention weights alone are not reliable causal explanations
  ([Jain and Wallace, NAACL 2019](https://aclanthology.org/N19-1357/)).
- Test bounded memory directly: preserve \(z_t^*\) while varying order, redundant questions, and
  lexical distance. A scalar accumulator should generalize; a bag-of-phrases heuristic should not.

### Phase B: exact strategic source

Introduce a transparent goal-conditioned mixture policy. The source observes the true answer
\(T_s(Q)\) to query \(Q\), wants the listener to believe a displayed decoy \(c\), and independently
chooses on each turn:

- with probability \(1-\lambda\), report the truthful answer \(T_s(Q)\);
- with probability \(\lambda\), report the answer \(T_c(Q)\) that the decoy would imply.

Thus

\[
P(Y=y\mid s,Q,c,\lambda)
=(1-\lambda)\mathbf 1[y=T_s(Q)]
+\lambda\mathbf 1[y=T_c(Q)],
\]

with a small preregistered symmetric lapse if zero likelihoods are undesirable. This is intentional,
goal-directed persuasion but remains exactly enumerable. The same transcript can be replayed under
different \(c\) and \(\lambda\), producing different exact posteriors without changing any report
token.

1. Pair strategic prompts with i.i.d. channels matched as closely as possible on marginal truth
   rate. Cross decoy identity, \(\lambda\), candidate order, and wording.
2. Probe source goal, policy strength, expected truthfulness, and listener posterior separately.
3. Patch only goal/policy-span activations across otherwise identical transcripts. Test whether
   report-state updates change according to the exact strategic likelihood rather than average
   reliability.
4. Path-patch goal and report information into the Phase-A accumulator. Test whether the i.i.d.
   circuit transfers, is augmented by a goal-conditioned gate, or is replaced by a distinct
   lexical deception pathway.
5. Include decoy-irrelevant and \(\lambda=0\) controls. Measure whether interventions improve the
   target Bayesian decision while preserving unrelated set membership, arithmetic, and language
   modeling.

Recent Qwen-specific behavioral work finds that continuation likelihoods track graded human lie
expectancy but underpredict deception driven by strategic gain
([Zhao and Coulson, SCiL 2026](https://aclanthology.org/2026.scil-main.46/)). Meanwhile, high-AUROC
deception probes and SAE features can fail to steer behavior
([Golechha and Garriga-Alonso, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/105c4de1195135fae4974aa8c5e27bbf-Abstract-Conference.html)).
Those results make listener-side causal updating, rather than another deception classifier, the
high-value gap.

### Rényi–Ulam follow-up

Do not merge the primary strategic experiment with the Rényi–Ulam liar game. The present i.i.d.
channel has a likelihood, the strategic mixture has a goal-conditioned likelihood, and the
Rényi–Ulam game has an adversarial lie budget and a worst-case survivor set. After Phase B, a clean
follow-up is to ask whether the recurrent state changes from scalar log-odds to a representation of
per-candidate remaining lie budgets. That would be a distinct high-level causal model and deserves
its own stopping rule.

## Compute and execution plan

| Stage | 4B BF16 | 9B BF16 | 27B GPTQ-Int4 |
|---|---|---|---|
| Behavioral factorial bank | Full bank | Full preregistered replication | Full or reduced replication |
| Selected-position activations | Discovery model; comfortable | Stream to CPU, small batches | Forward-only at preselected sites |
| Residual/module patching | Full discovery and causal validation | Replicate only selected layers/sites | Small confirmatory set only |
| Learned subspaces / DAS | Primary model | Pilot if 4B succeeds | Do not use |
| EAP-IG / backward search | 4B only, after coarse localization | Avoid unless sharply reduced | Do not use |
| SAE or full J-lens fitting | Not first-line | Not first-line | Not appropriate |

The 4B and 9B checkpoints both have 32 language layers; the 27B has 64. Each follows the 3:1
linear/full-attention schedule. The practical workflow is:

1. Freeze prompts, exact evaluator, semantic-margin metric, splits, patch directions, and stopping
   rules before inspecting module-level results.
2. Discover on 4B. Require held-out causal interchange before head/gate-level work.
3. Lock layers, positions, subspace dimensions, and intervention coefficients; replicate on 9B.
4. Use the quantized 27B only for behavioral replication and a small number of forward or exact
   activation patches. Report it as a size-plus-precision-plus-backend comparison, never a clean
   scaling law.
5. Release every prompt pair, token ID, exact high-level intervention, raw model margin, and patch
   result. Bootstrap by independent base history and report per-example effects, not only averages.

## Priority and stopping rules

1. **Experiment 1 is the lead project.** It has the cleanest causal model, strongest novelty, and
   lowest implementation risk. Stop or redesign if posterior probes fail grouped transfer and
   whole-state reliability/report patches have no held-out effect.
2. **Experiment 2 is the decisive competence-versus-interface study.** Run it in parallel only
   after the factorial bank and semantic metric are stable. A negative latent-state result is
   publishable if the visible-work arm reveals when and how the computation appears.
3. **Experiment 3 is the bridge to deception.** Begin Phase A after a causal accumulator site is
   identified; begin Phase B only if the listener can be shown to use some source model in the i.i.d.
   case. Otherwise strategic failures will be uninterpretable.

The highest-value possible result is not “Qwen has a Bayesian feature.” It is a causal account of
which exact update variables are represented, where they are composed, when the output interface
breaks them, and whether the same mechanism survives a source whose reports are chosen to change
the listener's belief.
