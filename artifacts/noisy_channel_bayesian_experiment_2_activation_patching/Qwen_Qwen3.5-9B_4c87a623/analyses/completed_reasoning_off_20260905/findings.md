# Notebook 18: completed reasoning-off probe findings

Frozen at 2026-09-05T02:09:29.780481+00:00: 9 complete sites,
1728 existing probes. Training used 56 schedules / 4,480 rows;
test uses eight held-out schedules / 640 rows. Layer indices are zero-based.

| site | target | layer | best_cv_score | r2 | auroc |
| --- | --- | --- | --- | --- | --- |
| report_3_answer | z_bayes | 0 | -0.0000 | 0.0000 | — |
| report_3_answer | gain_abs | 1 | 0.9921 | 0.9930 | — |
| report_3_answer | delta_a | 3 | 0.3591 | 0.5286 | — |
| report_3_answer | z_heuristic | 7 | 0.2252 | 0.1333 | — |
| report_3_answer | reliability_sign | 28 | 1.0000 | — | 1.0000 |
| report_3_answer | gain | 31 | 0.9794 | 0.9796 | — |
| final_prompt | z_bayes | 2 | 0.0024 | -0.0009 | — |
| final_prompt | reliability_sign | 2 | 1.0000 | — | 1.0000 |
| final_prompt | z_heuristic | 20 | 0.2370 | 0.3793 | — |
| final_prompt | delta_a | 21 | 0.3773 | 0.4553 | — |
| final_prompt | gain | 30 | 0.9964 | 0.9974 | — |
| final_prompt | gain_abs | 30 | 0.9948 | 0.9944 | — |
| answer_line | reliability_sign | 2 | 1.0000 | — | 1.0000 |
| answer_line | gain_abs | 3 | 0.9977 | 0.9983 | — |
| answer_line | gain | 12 | 0.9942 | 0.9930 | — |
| answer_line | z_heuristic | 19 | 0.3757 | 0.4417 | — |
| answer_line | delta_a | 20 | 0.5347 | 0.6780 | — |
| answer_line | z_bayes | 28 | 0.1887 | 0.2625 | — |


Reliability and its sign are readily decoded. Agreement and the unsigned heuristic
are better represented than the signed Bayesian interaction near the prompt boundary.
This is consistent with experiment 2's H1: separately available ingredients are not
successfully composed into the signed product used for a direct answer. It is not
proof that the product is absent: a nonlinear or distributed code can evade a linear
probe. Nor does decoding alone show the model causally uses a variable.

The generated answer-line site includes the answer numeral itself. Its stronger
decodability cannot establish that a Bayesian state existed before the answer.
The bank holds out schedules, but not reliability values, prompt renderings or
candidate identities; all rows use candidates 2/7 in one order. It therefore does
not meet experiment 2's complete crossed generalization design. No causal rescue
claim follows from these probes.

The original report's below-chance failure can be tested against the saved behavior
plots: a positive slope of C1−C2 margin on agreement at r<0.5 has the wrong Bayesian
sign. The exact slope is log(r/(1−r)), which is negative in that regime. CSVs retain
separate emitted-answer and teacher-forced boundary-logit accuracy, excluding exact
Bayesian ties from both choice metrics.

The two user selections reproduce counts 335
and 41. Among the latter,
7 have nonpositive saved
C1−C2 margins despite emitting C1. These must not be silently labeled logit-clean.

Exact follow-up match: 0 donor/recipient
pairs satisfy all requested matching fields. This dataset contains one result per
exact prompt. A change in question sets, reports, reliability, prompt or ordering
would be a different matching design and must be stated explicitly. Notebook 19
preserves this constraint and records the audit; an empty design is not evidence
that patching cannot work.

All-layer test curves and transfer plots are exploratory. The top-three layer lock
was saved before reading test activations and remains unchanged. Reliability-site
agreement probes are negative controls: those sites occur before the evidence.
See `reports/research_agenda/NOTEBOOK13_UPDATE.md`, experiment 2, for the intended
causal interchange and control requirements.
