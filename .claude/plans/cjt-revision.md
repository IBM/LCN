# Plan: revise `docs/cjt.tex` — NeurIPS-style exposition

Revision of the existing 19-page `docs/cjt.tex` (1395 lines). Five goals from the
brief: (1) simplify §3 and purge implementation references; (2) same treatment for
§4–§6; (3) keep the formal results and proofs, and verify their correctness;
(4) replace the code-style pseudocode in §3.7 with formal pseudocode; (5) fix the
junction-tree figure spacing.

## Guiding principle

Reframe the document from *engine documentation* to *algorithm description*. The
present text repeatedly says "the implementation does X"; the revision states what
the **algorithm** does, mathematically. An implementation detail survives only if it
changes a guarantee — that test is what keeps the soundness caveats while cutting
the API tour.

## 1. Purge implementation references (§3 especially)

There are 90 `\texttt{}` occurrences; the great majority are code identifiers.

**Delete outright** (pure implementation, no mathematical content):
- §3.1 "Input: structure only" — the whole `cn.factors` / `cn.node_atoms` /
  `cn.factorization.factors` / `cn.lcn` inventory, `lobo`/`upbo`,
  `extreme_points`, `merge_budget`, the `CredalNetworkVertices.from_lcn`
  listing, the 1 ms build time, "the constructor's assertion". **Keep one
  sentence** of real content: the algorithm needs only the chain graph's family
  *scopes* plus the LCN's sentences and assertions — never the per-family interval
  bounds or their extreme points — which is why it targets $\DLCN$ and not $\DSE$.
  Fold that sentence into the §3 preamble and drop the subsection.
- §3.6 "Interface" — **delete entirely.** The `run(...)` signature, the return
  2-vectors, `singleton_marginals`, `d5_exact`, `nlp_stats`, `degenerate`,
  timings, and the `build_and_solve_jt_nlp` default-solver trap are all API, not
  algorithm. The one substantive item — that the width is reported in *atoms*
  ($w{+}1$) rather than as treewidth $w$ — becomes a one-line notational
  convention where $w$ is first defined.
- `Rem rem:repro` (discarded SCIP gap; `PYTHONHASHSEED` restart seeds) — **delete.**
  Pure implementation trivia. The reproducibility note about the interpreter
  (`Rem rem:interp`, §8) goes too.
- `\lstset` and every `lstlisting` in §3. After this, `listings` is needed only by
  §8; see item 5 below on whether §8 survives.
- Solver product names as section headings: "SCIP (default, global)" /
  "ipopt (local, multi-restart)" become "Global solution" / "Local solution", with
  the solver named once in passing.
- The line-reference sentence at the top of §3 ("Line references are to
  `lcn/inference/marginal/cn/junction_nlp.py`").
- `max_cluster_atoms`, `gap_tol`, `host`, `induced_width`, `nlp_stats` as
  identifiers → plain prose ("the width budget", "the gap tolerance", "the hosting
  map").

**Keep, because each changes a guarantee** — restated in mathematical rather than
code terms:
- Local solution yields an **inner** bound (`rem:ipoptinner`). Restate as a property
  of local optimization under widening aggregation; drop the option names.
- The denominator floor $\varepsilon$ is a **soundness** hypothesis
  (`rem:denfloor`). Keep in full; state $\varepsilon$ as a parameter, not a
  constant name.
- The $2^n$ fallback is exact in formulation but not certified (`rem:oracle`).
  Keep, shortened, phrased as "a fallback to the direct $2^n$ formulation, solved
  by a local method and therefore uncertified".
- Vacuous clamping on solver failure, and the all-vacuous degeneracy signal — one
  sentence, no flag name.

Benchmark *instance* names (`asia`, `alarm`, `smokers`, `polytree`, `chain`) and
sibling `.tex` filenames stay: those are data and bibliography, not implementation.

## 2. Restructure §3 for concision

Target: from ~8 subsections + 9 `\paragraph`s down to **four subsections**, roughly
2.5 pages (currently ~4.5).

- **3.1 Overview** — the four-step pipeline in a short paragraph, plus the
  scopes-only input sentence rescued from the old §3.1. Sets up notation
  $\mathcal C$, $A(C)$, $S_{CD}$, $w$.
- **3.2 Constructing the cluster tree** — merge the four `\paragraph`s
  (augmentation / elimination order / atom clusters / edges) into flowing prose.
  Keep: augmentation by cross-family **sentences only** and *why* (assertions are
  structural; augmenting with them inflates width — a clean chain would collapse to
  one $2^n$ cluster); min-fill; RIP by construction = H1; compound nodes; leaf
  pruning with the non-minimality caveat. Drop the bit-order/global-atom-order
  detail (it is a packing convention, invisible mathematically).
- **3.3 Hosting and the assertion triage** — keep the three-way triage as an
  enumerated list; this is the algorithm's most consequential step and the direct
  source of H2/H3. Keep the note that case 3 is the only looseness source.
- **3.4 Solving** — merge old 3.4 (query-independence), 3.5 (solving), 3.8
  (complexity). Query-independence is two sentences. Complexity keeps the
  variable/constraint counts and `rem:counting`, but recast: part (a) currently
  reads as an erratum against the implementation's statistics — restate it as a
  direct classification of which rows are nonlinear (bilinear assertion rows, and
  the ratio row under evidence), which is the mathematically interesting statement,
  and keep part (b)'s $2^{w+1}$ bound.

## 3. Formal pseudocode (§3.7 → a float in §3.2/3.4)

Replace the `lstlisting` block — which is written in a Python-ish dialect with
`<-`, `#` comments, `cap`, and function names lifted from the code — with a proper
`algorithm` + `algpseudocode` float. Both packages are present in this TeX
installation (verified), and they are the NeurIPS-conventional choice.

- `\usepackage{algorithm,algpseudocode}` in the preamble.
- One float, `\caption{\textsc{CredalJT}}`, `\label{alg:cjt}`, with
  `\Require` / `\Ensure`, `\State`, `\For`/`\EndFor`, `\If`/`\ElsIf`/`\Else`,
  `\Return`, and `\Comment` for the H1–H4 annotations.
- Mathematical notation throughout: $\mathcal C$, $A(C)$, $S_{CD}$,
  $\mathrm{host}(\cdot)$, $q_C$, $\Delta(\cdot)$, $\readoff_a$ — no
  `ConstraintModel`, no `ExactInference`, no `max_cluster_atoms`.
- Structure: build phase (scopes → order → tree → clusters → separators → budget
  check), hosting phase (sentences; assertion triage), program assembly, then the
  per-atom loop with only the objective changing.
- Keep the `[H1]`…`[H4]` margin comments — they tie the algorithm to
  Theorem~\ref{thm:d5} and are the main reason the float earns its place.

## 4. Figure: spacing and rendering

Current defects in `fig:jt`, confirmed by reading the coordinates: cluster nodes
sit at $y=1.75,0.95,0.15,-0.65$ — a 0.8 cm pitch for nodes 7 mm tall, so they
nearly touch; and separator labels are placed `[sep,midway,right=2mm]` on a
2 mm gap, so a dashed box overlaps the edge and its neighbours. Fixes:

- Cluster pitch $0.8\to1.5$ cm; `minimum height` 7 mm; give `clus` a
  `minimum width` so the four boxes align; `node distance` via `positioning`
  rather than absolute coordinates where practical.
- Separator labels: place with an explicit anchor to the *right* of the edge with a
  larger offset (≈6–8 mm), or convert the separator to a small labelled node
  *on* the edge (the standard junction-tree rendering: cluster–separator–cluster).
  Prefer the on-edge separator box: it is conventional, self-documenting, and
  removes the overlap by construction.
- Widen the horizontal gap between panel (a) and panel (b) (currently the moral
  graph spans $x\in[-0.75,1.5]$ and the tree sits at $x=4.6$ with labels
  extending right) so the panel-(a) caption and the tree do not collide.
- Panel captions (a)/(b): place below each panel at a common $y$, with enough
  clearance from the lowest node.
- Verify by measuring the compiled result, not by eye: check the `.log` for
  Overfull boxes in the figure and confirm the figure's bounding box.

## 5. §7, §8, §9 disposition

- **§7 (liftable paper subsection)** — keep. It is already NeurIPS-register and is
  the most reusable part of the document. Light edit only: ensure it has no
  dependence on anything deleted above.
- **§8 (Reproducing the numbers)** — the brief says to purge implementation
  references, and this section is entirely Python snippets. **Convert** to a short
  "Experimental setup" paragraph: the instance, that bounds were computed by the
  junction-tree program under a certified global solver and cross-checked against
  the direct $2^n$ formulation, and that all endpoint statuses were confirmed at
  gap $0$. Drop the two `lstlisting` blocks and `rem:interp`. This also removes the
  last use of `listings`, so drop that package.
- **§9 (Provenance)** — keep, trimmed: it is scholarly apparatus, not
  implementation. Keep the "measured vs cited" split and the related-notes
  paragraph.

## 6. Verify the formal results (do not merely re-typeset)

The brief asks to make sure the results are correct. Checks to perform, and one
already done:

- **Already verified:** in §5's instance, $S=\{x_1,x_3\}$ genuinely separates
  $x_0$ from $x_2$ in $G^\ast$ (computed the chordal adjacency
  $x_0\!-\!x_1,x_0\!-\!x_3,x_1\!-\!x_2,x_1\!-\!x_3,x_2\!-\!x_3$ and confirmed
  $x_2$ is unreachable from $x_0$ once $S$ is removed; $x_0\!-\!x_2$ is not an
  edge). So the assertion satisfies Definition~\ref{def:d5}'s
  separator-entailment, not merely a tree-level test. **Fix the wording in §5.3**,
  which currently argues via "removing $S_{12}$ leaves $x_0$ on the $C_1$ side" —
  a subtree argument. State it as separation in $G^\ast$, matching the definition
  the theorem actually uses.
- Re-read `lem:rip`'s induction for the leaf/RIP step and the $q_S(s)=0$ kernel
  case; confirm the non-uniqueness clause is stated as *existence + conditional
  uniqueness*, which is what the proof delivers.
- Re-read `thm:d5`: confirm the projection half truly uses only H1 and H4 (it
  should not invoke hosting), and that the reconstruction half's appeal to
  factorization $\Rightarrow$ global Markov is stated in the no-positivity
  direction. Confirm the conditional-query case is covered by H4 plus
  `rem:denfloor`.
- `cor:ktree`: check each of H1–H4 is actually discharged by the stated k-tree
  property, and that the cost arithmetic $\sum_C 2^{|C|}$ over $O(n)$ clusters of
  size $k{+}1$ gives $O(n2^{k+1})$.
- Confirm every $\ref{}$ still resolves after the deletions (`sec:algo-api`,
  `sec:algo-pseudo`, `sec:algo-input`, `sec:nlp-obj` are referenced from
  elsewhere — retarget or remove those references).
- Re-check the arithmetic in §5 against the recorded measurements: $22$ variables
  $=8{+}8{+}4{+}2$; $22$ constraints $=4{+}10{+}8{+}0$; separator count
  $4{+}4{+}2=10$; and for Table~\ref{tab:contrast}, asia $2{+}4{+}4{+}4=14$ and
  alarm $2{+}4{+}2=8$.

## 7. §4, §5, §6 streamlining

- **§4 (the program).** Already close to NeurIPS register. Compress the seven
  `\paragraph`s into a lead-in plus the displayed program, keeping every equation
  (they are all referenced). Remove the implementation aside about indicators being
  built over "the host cluster's own truth table"; keep the two encoding points
  (only the $x{=}1$ slice; the product must range over whole $Y$-configurations,
  since pairwise decomposition is strictly weaker for $|Y|\ge2$) — those are
  mathematical, and the second is a real trap.
- **§5 (example).** Keep all six subsections and every transcribed constraint —
  the full 22/22 listing is the point of the section. Edits: fix §5.3 per item 6;
  drop the `induced_width` aside in §5.2 (subsumed by the $w$ convention); in §5.5
  replace "the engine reports `d5_exact` true, maximum cluster 3 atoms, and a total
  time of about 1.8 s" with a plain statement that all four hypotheses hold and the
  bounds coincide with the reference; keep the realized separator marginals (they
  illustrate calibration) but present them as a table rather than an inline display.
- **§6 (exactness).** Structurally sound; keep definitions, lemmas, theorem, proofs,
  corollary and remarks verbatim except: replace `\texttt{ExactInference}` and
  option strings in `tab:master`/`tab:topo` with prose engine names ("direct $2^n$,
  global" / "direct $2^n$, local"); drop `rem:ipoptinner`'s option names; and in
  the closing "practical summary" paragraph remove the implementation-flavoured
  clause about which engine is "most trustworthy", keeping the factual part.

## Deliverables & verification

1. Revise `docs/cjt.tex` in place.
2. Compile twice; require **zero** errors, zero unresolved references, and no
   Overfull box $>20$ pt. Confirm no `??` in the PDF text.
3. Inspect the rendered figure region for overlap (extract the figure page and
   check node separations), not just the absence of TeX warnings.
4. Confirm the page count drops (expect ~15–16 from 19) while §5's full constraint
   listing, all statements and all proofs remain intact.
5. Re-verify no reference to a deleted label survives: `grep` for the removed
   labels and for the identifier list purged in item 1.

## Out of scope

- No change to any source file; no re-running of the inference engines (the
  numbers are unchanged and already verified).
- No change to the mathematical content of any theorem, lemma or proof — only
  wording, unless item 6 uncovers an actual error.
- `cjt_exactness.tex` and `properties.tex` remain untouched.
