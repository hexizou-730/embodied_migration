# Repair, Don't Regenerate: Measured-Evidence Repair Makes LLM Robot-Code Migration Reliable

Draft v0.1 — 2026-11-21. All numbers trace to archived evidence under
`evidence/` and are recomputable offline. Citation placeholders marked
`\cite{}`. The StackCube fresh from-scratch cell is pending the running n=5
batch and marked `PENDING`.

## Abstract

Robot task programs do not transfer across embodiments: code written for one
manipulator couples its motion calls, gripper conventions, and geometry to
that robot. The default remedy — regenerating the program from scratch with an
LLM — is unreliable in simulation even for simple manipulation. We study an
alternative that keeps the source program's task structure and repairs only
what measured execution proves broken. A deterministic analysis first turns a
successful source program into a stage program with per-action attribution; a
replay of a failed target candidate then localizes repair to the first stage
whose measured end state contradicts an official predicate that the source run
had made true; an LLM rewrites only that stage under a structural contract, and
the simulator's official evaluator decides acceptance. On ManiSkill
StackCube (Panda to xArm6), this loop reaches official success in 75 of 80
repair runs (median one cycle) while fresh from-scratch generation succeeds
once in five runs under the same provider, and a repaired adapter held fixed
succeeds on 9 of 10 unseen seeds. A pre-registered paired ablation over 40
repair runs shows the choice matters less than expected: restricting the
repair output to the diagnosed stage saves a CI-separated 38% of tokens per
run but does not change success or executed actions, so the large action
savings reported for stage-preserving pipelines come from stage preservation
and diagnosis rather than output scope. On the easier PickCube task the same
pipeline generates from scratch successfully in six of six runs at about
$0.002 per run. We release all harness code, raw evidence, and offline
verification scripts.

## 1 Introduction

A robot task program is quietly full of its robot. Which planner call moves
the end-effector, which gripper value closes the fingers, where the
tool-center point sits relative to the fingers — all of it is embodiment
state. Moving the program to a different robot therefore cannot be a
re-formatting exercise, and the dominant LLM remedy today is to regenerate the
whole program against the new robot's interface. In simulation this remedy
works sometimes: with a flash-scale model, fresh from-scratch generation on a
hard task succeeds once in five independent runs (four exhaust their repair
budget), with failures dominated by a stage that claims completion while the
official success predicate stays false. "Sometimes" is a poor foundation for
a research method, so we ask a repair question instead: given one failed
candidate that already carries most of the task's structure, what does it take
to make fixing it near-certain?

Our answer is a measured-evidence repair loop built on three commitments.
First, the source program's task structure is extracted deterministically —
no LLM in the analysis — into a stage program with per-action attribution to
source lines, so every claim about "which stage" is auditable. Second,
failure localization is measured, not asked of the model: a replay of the
failed candidate records stage-boundary states, and a stage that claims
completion while an official predicate that the source run had made true
stays false is reopened for repair — the align stage that "passed" while
`is_cubeA_on_cubeB` stayed false is the canonical case. Third, acceptance
belongs to the simulator alone: the official evaluator, the unchanged
controller, and hash-checked artifacts decide success; the LLM only ever
proposes code.

Three findings organize the paper.

**Finding 1 — Repair is nearly certain where generation is not.** Across 80
paired repair runs on StackCube (Panda to xArm6), the loop reaches official
success in 75 runs, median one repair cycle, whether the LLM rewrites one
diagnosed stage method or the whole module (Section 5.2). A repaired adapter
held fixed succeeds on 9 of 10 held-out seeds, against 5 of 10 for a
historical from-scratch fixed adapter, so the product of repair is not a patch
overfit to one failure scene (Section 5.4).

**Finding 2 — What matters is feedback and preservation, not output scope.**
A pre-registered paired ablation over 40 repair runs shows equal success
(37/40 vs 38/40, symmetric discordance), no separable difference in executed
actions, and a CI-separated token overhead of about 38% per run for
whole-module rewrites (Section 5.3). The large action savings that
stage-preserving pipelines report in from-scratch comparisons therefore
attribute to locking passed stages and diagnosing failures — not to making the
model output less code.

**Finding 3 — The pipeline covers the difficulty spectrum at trivial cost.**
On the easier PickCube task the same pipeline — same harness, same model,
three-stage program extracted by the same analyzer — succeeds from scratch in
six of six independent runs, one cycle each, about $0.002 per run
(Section 5.1). Difficulty decides when repair is needed; the pipeline does
not have to.

We state the boundaries plainly. The repair evidence is resume-based by
construction — every repair run starts from the same recorded failed
candidate — and is never merged with from-scratch statistics. The study spans
two tasks and one robot pair at a flash-scale model, all in one physical
scene per task; nothing here is a cross-task or cross-robot generalization
claim. And the loop's reliability presumes interface facts that human probing
established first (a gripper direction measured by finger-pad separation, a
stationary-close no-op found by an explicit probe); automation starts where
interface measurement ends, and we treat that division of labor as a finding,
not a footnote (Section 6).

Contributions: (i) a measured-evidence repair loop with source-postcondition
localization, evaluated at 40 paired repair runs with a pre-registered
confirmatory half; (ii) an ablation that cleanly attributes from-scratch
savings to protocol rather than output scope; (iii) a difficulty-stratified
account of when LLM code migration needs repair at all; (iv) a full evidence
discipline — raw cycle records, offline recomputation (242 checks, zero
failures), retained failures — released with the harness.

## 2 Related Work

**Cross-embodiment transfer.** Most work transfers learned parameters or
policies across robots \cite{octo,rtx,crossformer}. We transfer executable
task code, which keeps the program inspectable and the task structure explicit,
and our evaluation criterion is the simulator's own success predicate rather
than task-level proxies.

**LLM code generation for robots.** Code-as-policies lines of work
\cite{codeaspolicies,progprompt} generate robot programs from language;
open-ended explorers \cite{voyager} accumulate skills with self-verification.
These loops rarely close on a physical official evaluator with per-action
attribution; our failures are grounded in measured stage-boundary states and
official booleans, which is precisely what makes a bounded repair loop
trustworthy.

**Automated program repair and self-debugging.** LLM self-repair
\cite{selfdebug,sweagent} fixes code against tests. Our setting differs
in two ways: the "test" is a physical simulation whose failure layers
(interface assumption, false stage completion, contact, budget) need
diagnosis, and the repair scope is constrained and verified structurally
(AST hashes) so a repair cannot silently rewrite what already works.

## 3 Method

### 3.1 Stage programs from source code

A successful source program is analyzed deterministically (AST and
reaching-definition dataflow): action calls partition `run(env)` into ordered
stages — approach, grasp, lift, align, release on StackCube; approach, grasp,
align on PickCube — with entities resolved from the recorded replay, per-line
action attribution (103/103 and 70/70 actions mapped on the two tasks), and
stage-end official evaluations. The extractor is task-neutral: task completion
semantics are declared per study configuration (PickCube's `is_grasped` /
`is_obj_placed` flow from the study file into prompts), and a byte-parity
fixture proves the original StackCube prompts unchanged by the
generalization. The extractor's boundary is documented rather than hidden: a
pull-cube program that pre-closes its gripper as a stick produces a nonsense
stage order and was excluded.

### 3.2 Measured-evidence repair

Each repair run begins by replaying the same hash-pinned failed candidate
(101 actions, stopping at release). Localization is two-layer: the first
stage that fails at runtime, and — the mechanism we consider load-bearing — a
source-postcondition check that reopens any stage claiming completion while an
official predicate that the source run made true remains false. The LLM then
rewrites only the localized stage method (or, in the ablation's other arm, a
complete module) under a structural contract: the fixed stage order as
literal calls, measured-state reads, bounded loops, no simulator mutation, no
helpers outside stages. Candidates that change nothing executable are
rejected before execution. Acceptance is the official evaluator alone.

### 3.3 Evidence discipline

Every run archives prompts, raw LLM responses, spliced candidates, trials,
AST hashes, and usage; an offline analyzer recomputes all aggregates from raw
cycle records (242 checks across this study, zero failures) and reports
Wilson intervals, an exact paired sign test, and percentile bootstrap CIs.
The confirmatory ablation half (n=30, seeds 10–39) was declared after seeing
n=10 and reported separately, never pooled. Failed runs stay in every
denominator, including a task whose candidate source failed validation 0/10
and was dropped.

## 4 Experimental Setup

Tasks: ManiSkill `StackCube-v1` (five stages) and `PickCube-v1` (three
stages), source robot Panda with frozen, multi-seed-validated source programs
(10/10 official success each), target xArm6 with the robotiq gripper in
`pd_ee_delta_pose` (7-D), a robot not on either task's supported list — the
non-native-target situation is deliberate and reported. Model:
deepseek-flash (V4.1 Flash) via the official DeepSeek API, temperature 0,
thinking disabled, 6,000-token output cap; historical comparisons used
OpenRouter routing, and provider changes are stated wherever numbers from
both eras appear near each other. Environment seed is fixed per task (0);
independence comes from LLM sampling seeds (0–39 across the ablation).
Held-out seeds 100–109 are never used for repair feedback. Gripper endpoints
(0.0 open, 0.81 close) and the stationary-close behavior were measured by
explicit probes before any generation. Total study cost to date is about
$0.15.

## 5 Results

### 5.1 Difficulty-stratified transfer

Table 1 (main results).

- PickCube: from scratch 6/6 independent runs (sampling seeds 0–5), all
  one cycle, 22–28 actions, ~$0.002 per run; the generalized three-stage
  prompt passed byte-level checks and study-declared semantics appeared in
  every prompt.
- StackCube: fresh from-scratch n=5 under the current provider: 1/5 official
  success (the one success took two cycles; the other four exhausted the
  three-cycle budget), corroborating the historical 2/3-per-method record
  from the OpenRouter era. The failure mode is systematic rather than random:
  all five first-cycle candidates claimed completion of every stage while the
  align postcondition was violated, the source-postcondition check caught it
  5/5 times, and what the budget could not absorb was repairing align from a
  cold start. Diagnosis was not the bottleneck; re-reachability of the
  correct stage behavior was. Repair from a recorded failure, by contrast,
  succeeds in 75 of 80 arm-runs (Section 5.2).

### 5.2 Repair reliability

Forty paired repair runs from the same hash-pinned failed candidate: n=10
(seeds 0–9) and a pre-registered n=30 (seeds 10–39). Success 37/40
(method-repair) vs 38/40 (module-repair); discordant pairs symmetric (1/1 in
the n=30 half; sign p = 1.0); median one repair cycle in both arms.

### 5.3 Granularity is second-order

Paired differences (module − method): actions −2.8 [−17.3, 8.3] then +5.5
[−14.8, 30.0] across the two runs — indistinguishable from zero. Tokens
+2,469 [−42, 4,741] at n=10, +3,178 [2,003, 4,458] at the pre-registered
n=30: whole-module rewrites cost about 38% more tokens per run, and nothing
else differs. Combined with the historical from-scratch comparison — where
the stage-preserving method executed 366 actions against 975 and 1,329 for
whole-program methods — the attribution is direct: stage locking and measured
diagnosis produce the action savings; output scope produces only the token
overhead. The one repair format failure in 40 pairs (a nested helper rejected
by the structural check before execution) is retained and shown in the
taxonomy.

### 5.4 Repaired code generalizes

The first successful method-repair adapter, held fixed and replayed on
unseen seeds 100–109 with zero LLM calls: 9/10 official success (the single
failure is retained), against 5/10 for a historical from-scratch fixed
adapter. Repair did not overfit the failure scene.

### 5.5 Failure taxonomy

Table 3: interface-assumption errors found by human probes (gripper
direction; stationary-close no-op); false stage completion caught by
source-postcondition reopening; format brittleness of single-method output
(one nested-helper rejection in 40 pairs); physical failures of from-scratch
generation (contact never established, budget exhausted). The pattern worth
naming: humans fixed interfaces, the LLM fixed policy — automation's real
boundary in this pipeline today.

## 6 Discussion

When is repair needed? The difficulty table answers operationally: on an easy
task, generate — it is one cycle and two-thirds of a cent; when generation
fails on a hard task, repair from the failure with measured evidence — it is
near-certain. What makes repair work is not how much code the model writes
but that its claims are checked against the official evaluator and the source
run's own predicate flips; the ablation is what lets us say this with a
straight face, because the obvious confound — write less, fail less — is
measured out. The honest limit of "automatic" is the interface probes that
preceded it; a pipeline that measures its own interface assumptions the way
it measures stage postconditions is the natural next step. Scope limits:
resume-based repair statistics, one scene per task, one robot pair, one model
class; the numbers are effects and intervals, not universals.

## 7 Conclusion

Measured feedback and source-grounded localization — not output scope — make
LLM robot-code repair reliable across embodiments, at cents per task, with
evidence that survives recomputation.

## References

(draft entries; verify venues and years against the camera-ready target
before submission)

- \cite{codeaspolicies} J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman,
  B. Ichter, P. Florence, A. Zeng. "Code as Policies: Typed Procedure Call
  Representations for Language Model Programs." ICRA 2023.
- \cite{progprompt} J. Singh, V. Blukis, A. Mousavian, A. Garg, D. Xu,
  E. Toshev, S. Song. "ProgPrompt: Generating Situated Robot Task Plans using
  Environment-Aware Large Language Models." CoRL 2022.
- \cite{voyager} G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Fan, Y. Zhu,
  L. Fan, A. Anandkumar. "Voyager: An Open-Ended Embodied Agent with Large
  Language Models." arXiv:2305.16225, 2023.
- \cite{octo} Octo Model Team (D. Ghosh et al.). "Octo: An Open-Source
  Generalist Robot Policy." RSS 2024.
- \cite{rtx} Open X-Embodiment Collaboration. "RT-X: Open Robotics Datasets
  for Large-Scale Cross-Embodiment Robot Learning." arXiv:2310.08899, 2023.
- \cite{crossformer} N. Doshi et al. "Cross-Embodiment Transfer with
  CrossFormer." RSS 2024.
- \cite{selfdebug} X. Chen, M. Lin, N. Schärli, D. Zhou. "Teaching Large
  Language Models to Self-Debug." ICLR 2024.
- \cite{sweagent} J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao,
  K. Narasimhan, O. Press. "SWE-agent: Agent-Computer Interfaces Enable
  Software Engineering." NeurIPS 2024.
- \cite{maniskill3} S. Tao et al. "ManiSkill3: GPU Parallelized Robotics
  Simulation and Rendering for Large-Scale Embodied AI." NeurIPS Datasets and
  Benchmarks, 2024.
- \cite{sapien} F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu,
  H. Jiang, B. Yuan, H. S. Fang, D. Xu, C. Gan. "SAPIEN: A SimulAted
  Part-based Interactive ENvironment." CVPR 2020.
