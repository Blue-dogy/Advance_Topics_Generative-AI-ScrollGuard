# ScrollGuard Phase 2: Questions and Answers

Each answer starts with a one-line version that can be said aloud, followed by the supporting numbers. All figures come from the repository (`docs/results/`, `docs/annotation/results/`) and the Phase 2 report.

---

## 1. What agent techniques does this use?

**Short answer.** The benchmark evaluates agents, and the agent we ran on it uses tool calling, a ReAct-style loop, agentic retrieval and structured output with reasoning first.

| Technique | Where | Lecture |
|---|---|---|
| Tool calling | Track B: three tools (touch log, statistics, user context), called with JSON | Ch11 |
| Agent loop (ReAct-style) | Track B: decide, call a tool, observe the result, decide again; at most 4 calls | Ch13 |
| Agentic retrieval | The goal and observation log are not given; the agent must fetch them | Ch05 |
| Structured output, reason first | `{"reason": ..., "action": ...}`; format errors are recovered or retried | Ch04 |
| Workflow vs. agent | Track A (everything given) vs. track B (agent searches) | Ch13 |
| Prompt and tool design | Column descriptions, tool descriptions, tool order | Ch04, Ch11 |

---

## 2. How do you evaluate? What are the input, the output and the expected behavior?

**Short answer.** The agent reads one usage situation and returns one of three actions; we compare the action with a human-validated label, automatically.

- **Input:** the situation (time, app, how the app was opened), the user (stated goal, state), a five-part observation log, and usage and gesture numbers.
- **Output:** one line of JSON, `{"reason": "...", "action": "..."}`.
- **Expected behavior:** each case has one correct action, following written labeling rules:
  - `no_intervention`: intentional or normal use
  - `gentle_prompt`: autopilot use, meaning two or more independent autopilot signals and no counter-evidence
  - `ask_context`: signals conflict or key information is missing
- **Scoring:** close-ended. Only the action is scored. The reason is kept for error analysis.
- **Metrics:** we report several metrics together, not accuracy alone:
  - accuracy with a 95% CI
  - false intervention (the core product metric)
  - detect rate
  - `ask_recall` and question burden
  - counterfactual pairs
  - cost score
  - context-check rate in track B
- **Example (S013):** the user wrote "just 5 minutes", but 520 short videos have played. The expected action is `gentle_prompt`.

---

## 3. What if the agent asks a weird question?

**Short answer.** We score the decision to ask, not the wording of the question; unneeded questions are penalized, and checking the wording is a next step.

- `ask_context` is a decision. The agent does not write the question text, so its wording is **not evaluated yet**. This is a limitation.
- What is already caught:
  - **Asking when it was not needed:** counted in question burden, with a cost of 0.8, close to the 1.0 of an unneeded intervention.
  - **An answer outside the three actions:** scored invalid (`invalid_rate`).
  - **A malformed reply:** `PARSE_ERROR`. In track B the agent gets a reminder and up to two more tries.
- Next step: have the agent write the question and score it with a rubric (relevant, neutral, short), using human or LLM judges. That is an open-ended evaluation, as described in the lecture.

---

## 4. Isn't the dataset too small?

**Short answer.** Yes, 81 cases is small, so we only claim large effects and do not interpret small gaps.

- With 81 cases the accuracy CI is about ±0.1. A gap like 0.62 vs. 0.63 is not interpreted.
- **Why it is small:** every case has a hand-written context and was labeled by two people. We chose diagnostic depth over size: adversarial, boundary, counterfactual and shortcut cases, each aimed at a specific failure. For scale, HumanEval in the lecture has 164 problems.
- **What we claim are large effects:**
  - tool order: 13% → 100% context checks
  - error direction: 60% vs. 15%
  - asking: people asked 15 and 9 times; the models asked 2 times in 486 decisions
- **Safeguards:** bootstrap 95% CIs, three runs per case in track A, and re-scoring on the 65 cases both annotators agreed on.
- **Scaling up:** 5,777 real rows are available, so more cases can be generated from templates, with people checking a sample. Template generation tends to bring shortcuts back, so it must run together with our shortcut checks.

---

## 5. How were the answers (labels) made?

**Short answer.** We wrote labeling rules first, labeled the cases with them, then had two independent people label all 81 cases blind and fixed five labels by majority vote.

1. Written labeling rules. For example, `gentle_prompt` needs two or more independent autopilot signals and no counter-evidence.
2. The original labels were drafted from those rules (AI-assisted).
3. Two people who did not design the benchmark labeled all 81 cases without seeing the answers.
4. Five cases where both annotators disagreed with the original label were relabeled by majority vote (2 of 3).

Final labels: 34 `no_intervention`, 32 `gentle_prompt`, 15 `ask_context`.

---

## 6. Can we trust these accuracy numbers? Would they hold if people took the test?

**Short answer.** People score about 0.8 and the 7B models about 0.6 on the same labels, so the labels hold up and the gap is real.

All rows below use the **original labels, before majority vote**, so the annotators are not compared with labels they helped set.

| Rater | Match with original labels | Cohen's kappa |
|---|---|---|
| Annotator A | **0.84** | 0.75 |
| Annotator B | **0.81** | 0.71 |
| Annotator A vs. B (each other) | 0.80 | 0.68 |
| All three raters | | Fleiss' kappa 0.71 |
| Qwen, track A (run 1) | 0.67 | 0.39-0.40 with each annotator |
| EXAONE, track A (run 1) | 0.59 | 0.28-0.29 with each annotator |
| Qwen, track B | 0.40 | about 0 with each annotator |

- **Which kappa:**
  - **Cohen's kappa** is for two raters: annotator vs. annotator, or annotator vs. the original labels.
  - **Fleiss' kappa** is for three or more: 0.71 across both annotators and the original labels.
- **The labels are reliable:** two independent people matched the original labels 81-84% of the time, a kappa of 0.71-0.75, which counts as substantial agreement.
- **The task is solvable:** people reach about 0.8, so the models' 0.6 is a real gap, not an impossible test.
- **Where people disagree:** agreement differs a lot by category.

  | Original label | Annotator A | Annotator B |
  |---|---|---|
  | `no_intervention` | 1.00 | 0.97 |
  | `gentle_prompt` | 0.84 | 0.94 |
  | `ask_context` | 0.56 | 0.33 |

  That is why we also report scores on the 65 agreed cases. The ranking stays the same: Qwen A 0.71, EXAONE A 0.68, rule B2 0.62.

---

## 7. Why split the evaluation into track A and track B?

**Short answer.** To separate two abilities: interpreting the context once it is given (A) and finding the context on your own (B).

- **What each track tests:**
  - Track A gives all information at once. It tests judgment: does the model read the context correctly?
  - Track B gives only the time, the app and the usage numbers, and the rest must be fetched with tools. It tests agency: does the model know what it needs and look it up?
- **The two tracks are otherwise identical:** same action definitions, same output format, same cases. A drop from A to B is caused by information seeking.
- **This matches the lecture:**
  - workflow vs. agent (Ch13)
  - agentic benchmarks score the trajectory, not just the answer (Ch03)
- **The split paid off:**
  - The same Qwen model fell from 0.63 (A) to 0.42 (B), because it checked the context in only 9% of cases.
  - Track B also made the tool-order experiment possible. With the context tool moved to the top of the list, Qwen's context checks went from 13% to 100%, while EXAONE's stayed at 0-3%. The same symptom had different causes, which we could not have seen with one track.
- **It is realistic:** on a phone, reading the user's context has a cost (privacy, battery), so a real agent has to decide when to fetch it.

---

## 8. `ask_recall` is very low. Isn't that a problem with the benchmark?

**Short answer.** No. People did ask on these cases, so the low score shows a real model weakness; the remaining weakness is that this category is small, which we report.

- **The task is possible:** with the same inputs, annotator A chose "ask first" 15 times and annotator B 9 times. Both models together asked 2 times in 486 track A decisions, and both times wrongly.
- **It is not caused by unclear labels:** on the 5 `ask_context` cases both annotators agreed on, the models still asked 0 times.
- **It is not caused by vague definitions:** in experiment 1 we defined the two actions more sharply. Asking rose from 0/15 to only 2/15, while unneeded interventions rose from 0/8 to 3/8, so we did not adopt it. The models write the conflict into their reasons ("could indicate mindless scrolling. However, ... some intentional use") and still pick a side.
- **It is not a lack of information:** with the context tool listed first, Qwen read the context in 100% of cases and still never asked.
- **The metric cannot be gamed:** an "always ask" baseline gets `ask_recall` 1.0 but scores only 0.19 accuracy and a cost score of 0.47, among the lowest, because unneeded questions are penalized.
- **The main goal is measured separately:** on the 66 cases whose label is `no_intervention` or `gentle_prompt`, track A models reach 0.76-0.77, against 0.65 for the best rule.
- **The real weakness:** the category has only 15 cases, 5 of them agreed by both annotators, and people also disagree most here. That gives low statistical power for comparing models on asking. The next version needs more "ask first" cases and annotators, with `ask_recall` reported on the agreed cases.
