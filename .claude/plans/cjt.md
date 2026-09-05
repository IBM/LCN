# Plan: `docs/cjt.tex` — CredalJT, consolidated

## Goal

A new **standalone** `docs/cjt.tex`: the single reference for the CredalJT engine
(`lcn/inference/marginal/cn/junction_nlp.py`, scheme D5) — algorithm, worked example,
paper-ready subsection, and the consolidated exactness theory with full proofs.

Standalone means: own preamble, own notation section, compiles on its own. It supersedes
the existing `docs/cjt_exactness.tex`, which is *not* a general CredalJT document but a
narrow forensic note on one instance (`ktree_fr_k3_n5_1`) investigating a
non-reproducing discrepancy report. **Leave `cjt_exactness.tex` in place**; cite it once
as the forensic companion.

## Grounding already done (facts verified, not to be re-derived)

- **Running example chosen:** `examples/ktree_fr_k2_n4_example.lcn` — 4 atoms, 4
  sentences. This is `Example ex:loopyexact` of `properties.tex`. It is ideal: loopy
  (so the example is non-trivial), yet small enough to typeset the whole junction tree
  and the whole NLP by hand.
- **Verified by running the shipped engines** (`.venv/bin/python`, `solver="scip"`):
  - Junction tree: 4 clusters `{x1,x3,x0}`, `{x1,x2,x3}`, `{x2,x3}`, `{x3}`; 3 edges;
    max cluster 3 atoms; chain-shaped, rooted at `{x3}`.
  - Separators: `{x1,x3}`, `{x2,x3}`, `{x3}`.
  - Hosting: `s0,s3 → {x1,x3,x0}`; `s1,s2 → {x1,x2,x3}`.
  - Sole LMC `(x0 ⟂ x2 | x3,x1)` is **RIP-implied** by the separator `{x1,x3}` and so
    is *not* imposed → `n_lmc = 0`. This is the doc's best single illustration of
    separator-entailment.
  - NLP: **22 variables**, **22 constraints** (4 simplex, 10 separator, 8 sentence,
    0 LMC), **6 nonlinear** (three Type-2 sentences × 2 rows).
  - Bounds: `P(x0)=[0,1]`, `P(x1)=[0.084382,0.684382]`,
    `P(x2)=[0.134877,0.996844]`, `P(x3)=[0,1]`; `exact=True`, width 3, ~1.8 s.
  - **Cross-validated** against `ExactInference(solver="global")`: all four atoms match
    to every printed digit, all eight SCIP solves `confirmed` at gap 0.
- Verified families (from the factorization, in build order): `x1` (root, no parents);
  `x2 | x1`; `x3 | x1, x2`; `x0 | x1, x3`. Max cluster is 3 atoms `= k+1` with `k=2` —
  say so, so the "3" does not read as inconsistent with `k=2`.
- `induced_width` is reported as **3**, which is the max cluster size in *atoms* (`w+1`),
  not the treewidth `w=2`. Use one convention in the doc and state it once.

## Structure of `docs/cjt.tex`

Preamble copied from `properties.tex` (same class/packages/theorem env/macros:
`\lo`, `\hi`, `\indep`, `\DSE`, `\DLCN`, `\restr`, `\readoff`, `\ch`, `\pa`, `\atoms`) so
quoted theorem text transplants unchanged. `article`, 11pt, geometry, amsmath/amssymb/
amsthm, booktabs, hyperref, tikz, listings, `\setlength{\emergencystretch}{3em}`.
Do **not** `\input{macros.tex}` — that file is for `neurips_2026.tex` and its `\pa`/`\ext`
would collide; declare macros locally as `properties.tex` does.

**Citations:** `properties.tex` has *no* bibliography — zero `\cite`, no `\bibliography`,
no `thebibliography`; external results are cited inline in prose ("Lauritzen, *Graphical
Models*, Prop. 3.8"). `docs/ref.bib` exists but no exactness note uses it. Match that
convention: inline prose citations, no bib file. This keeps the doc compilable with a
single `pdflatex` pass-pair and no bibtex step.

No ICLR `.sty` is vendored in this repo, so match ICLR in **language and typographic
discipline**, not in document class — state that choice in a comment at the top of the file.

1. **Abstract + Introduction.** What CredalJT is in one paragraph: it does *not* compile
   to a credal network; it builds one junction tree over the chain-graph moral graph and
   solves a *single* constraint NLP whose feasible set is the cluster-feasible set,
   optimizing the query read-off. Contributions: (i) full algorithm spec; (ii) worked
   example; (iii) exactness theory with proofs; (iv) self-contained paper subsection.

2. **Preliminaries and notation.** LCN syntax (Type-1 `P(φ)`, Type-2 `P(φ|ψ)` with
   bounds), the LMC, the induced credal set `\DLCN`, the strong extension `\DSE` and the
   `\DLCN ⊆ \DSE` framework, Gap A / Gap B, and where CredalJT sits: it targets
   `\DLCN` **directly** (unlike the `\DSE` family: CVE, CCTE, ApproxLP, IBP, IJGP).
   Keep it tight — one page, enough to make the doc standalone.

3. **The CredalJT algorithm.** The detailed description, in the order the code runs:
   - *Input:* only the chain-graph **structure** — `cn.factors` (`child`, `parents`),
     `cn.node_atoms`, `cn.factorization.factors` (for `CouplingConstraints`), and
     `cn.lcn` (the real input: sentences + `independencies`). Emphasize CredalJT reads
     **neither** `lobo`/`upbo` (the per-family interval local credal sets) **nor**
     `extreme_points`, hence the `solve_families=False, enumerate_vertices=False` build
     (placeholders `lobo=0, upbo=1`, `extreme_points=None`), and hence build time is just
     the symbolic factorization (measured 0.001 s on the example). Draw the structural
     conclusion the reader needs: **this is exactly why CredalJT targets `\DLCN` rather than
     `\DSE`** — it never passes through the interval/vertex representation that creates
     Gap A and Gap B, which is what separates it from every other engine in `cn/`.
     Mention the chain-graph families and compound nodes (an undirected component becomes
     one `"-"`-joined node, e.g. `C-D` in `alarm`), and that `merge_budget=1` is assumed.
   - *Junction-tree construction:* `_node_scopes` (per-family node scopes) → augmentation
     by cross-family **sentence** node sets only (`CouplingConstraints`,
     `kinds=("type1","type2")`) — with the explicit rationale from the code that LMC
     assertions are deliberately *not* used to augment, since RIP separators already
     enforce them and augmenting would inflate a clean Markov chain to a single `2^n`
     cluster instead of width 2; `min_fill_order` over the interaction graph (greedy
     min-fill, deterministic lexicographic tie-break) → `_jt_from_scopes` (bucket-tree
     port; **RIP holds by construction**, which is H1); node clusters flattened to
     atom-level clusters via `node_atoms` (so a *compound* node contributes all its atoms
     and a cluster can be wider than the node count suggests); the subsumed-leaf pruning
     loop; edges/separators (`S = A(C) ∩ A(D)`, possibly empty for a disconnected
     component); one global MSB-first atom order (`atom_order`/`_rank`).
     Two precision notes: the child→parent edge direction and the rooting are
     **inference-irrelevant** (the NLP constraints are symmetric equalities; rooting only
     drives printing and `_rip_implied`'s subtree split); and pruning removes subsumed
     *leaves* only, so a non-minimal clique tree can survive — do not claim minimality.
     A forest is possible (`roots()` returns one per component).
   - *Constraint assignment:* `host()` = first cluster containing the scope; sentences
     must be hosted or the engine falls back; `_rip_implied` decides which LMC
     assertions are already structurally enforced (skip), which are hosted (impose), and
     which are unhostable-and-unimplied (dropped as inert) — flag this last case as
     exactly where H3 can fail and the bound goes loose-but-valid (smokers).
   - *The NLP*, written formally (this is the doc's centerpiece — see §4 below).
   - *Objective and evidence:* linear read-off `A_q^T q_{C_a}`; with evidence, the
     bilinear auxiliary `obj_var · E = AE` plus the SCIP-only denominator floor
     `E ≥ 10^{-6}`.
   - *Solving:* `_solve_sense` — SCIP (`make_scip`, `time_limit=None → 3600 s`,
     `limits/gap = gap_tol`, single solve, no restarts) as a certified **global** backend
     vs ipopt as a **local** one: uniform interior start `_init_q`, then up to
     `_N_RESTARTS = 4` random restarts triggered by `_is_suspicious` (a vacuous endpoint
     within `_VACUOUS_TOL = 1e-6`), aggregating by `max` over maxima / `min` over minima.
     State the consequence plainly: because that aggregation only ever *widens*, the
     ipopt path returns an **inner** bound of the exact D5 interval — which is why
     `run` defaults to `solver="scip"`. Why global matters at all: the LMC rows are
     genuinely bilinear (and, under evidence, so is the ratio row), so the program is
     nonconvex.
     Two honest caveats to record: `_read_gap(results)` is called but its value is
     **discarded**, so CredalJT reports no per-solve optimality gap (unlike
     `ExactInference(solver="global")`, which keeps `self.status`); and the ipopt restart
     seeds come from `hash((sense, k))`, hence are not reproducible across processes
     unless `PYTHONHASHSEED` is fixed (the first, uniform, attempt is deterministic).
   - *Query-independence:* one tree, one constraint model, built **once**; per atom only
     the objective is swapped (`_clear_atom_objective`) and two solves run. Contrast
     with the old per-atom `CredalVE(coupling="d5")` path.
   - *Fallbacks:* `max_cluster_atoms = 16` budget exceeded, or an unhostable
     sentence/query → per-atom `ExactInference.run_query` at `2^n`; `d5_exact` records
     which path ran, and `nlp_stats` is `None` there. Be precise rather than flattering:
     that call passes `solver="local"`, so the fallback is exact *in formulation* but only
     **locally** solved — cite `[[exact-not-a-reliable-oracle]]`, and note the per-atom
     variant fires when `host([atom] + evidence_atoms)` is `None`, i.e. H4 fails for that
     atom alone. A failed solve is clamped to the vacuous side
     (`lo→0, hi→1`), never to an unsound tight bound; `self.degenerate` flags all-vacuous
     output as probable inconsistency.
   - *API:* `CredalJT(cnv).run(evidence={}, solver="scip", max_cluster_atoms=16,
     time_limit=None, gap_tol=0.0, verbosity=1)`. Returns
     `{atom → (np.array([1−hi, lo]), np.array([1−lo, hi]))}` — a pair of
     `[P(=0), P(=1)]` 2-vectors — and sets `singleton_marginals` (`{atom → (lo,hi)}` for
     `P(atom=1|e)`), `marginals`, `d5_exact`, `induced_width` (`max_cluster_size()`, in
     **atoms**, i.e. `w+1` not `w` — spell this out), `nlp_stats`, `degenerate`, and
     `build_time`/`elimination_time`/`total_time`. Evidence atoms are **excluded** from the
     result dict (unlike `ExactInference`, which emits a point mass). Note also that the
     single-query entry `build_and_solve_jt_nlp` defaults to `solver="ipopt"`, *not*
     `"scip"` — an easy trap for a reader reproducing numbers.
   - *Pseudocode* float (algorithmic environment or a `lstlisting`), ~20 lines.
   - *Complexity:* variables `Σ_C 2^{|A(C)|} = O(n 2^{w+1})`; constraints per category
     exactly as `_model_stats` counts them — simplex `|𝒞|`; separator
     `Σ_{(C,D), S≠∅} 2^{|S|}`; sentence `2` rows per hosted sentence; LMC one row per
     constraint group, where a group count is `2^{|Y|}·2^{|S|}` (`S ≠ ∅`) or `2^{|Y|}`
     (`S = ∅`). Exponential in **treewidth**, not `n`; `2·(#atoms)` solves against one
     shared constraint block. Two accounting corrections to make in the doc rather than
     inherit: (i) `_model_stats` classes a Type-2 sentence's two rows as *nonlinear*, but
     with constant `ℓ,u` they are **linear in `q`** — the genuinely nonconvex rows are the
     LMC groups and the evidence ratio, so state that and note the code's count is
     conservative; (ii) a hosted LMC's row count is `2^{|Y|+|S|}`, but `Y ∪ S ⊆ A(C)` for
     a *hosted* assertion, so it is bounded by `2^{w+1}` and the width bound survives —
     say so explicitly, since a wide `Y` looks alarming otherwise.
     Also note `_model_stats` rebuilds the truth table to count, so it is `O(2^{|C|})` per
     hosted assertion, not free.

4. **The nonlinear program, formally.** Decision variables `q_C ∈ Δ(2^{|C|})` per
   cluster. Write out: simplex; separator consistency `M_{C→S} q_C = M_{D→S} q_D` with
   `_marginal_matrix` defined as the 0/1 aggregation matrix; Type-1 rows
   `lo ≤ A_φ^T q_{C_h} ≤ hi`; Type-2 rows `A_{φ∧ψ}^T q ≥ lo · A_ψ^T q` and
   `≤ hi · A_ψ^T q` (note this is the ratio cleared of denominators — no division, and
   it is vacuously satisfied when `P(ψ)=0`, worth a remark); LMC groups from
   `lmc_constraint_groups_vec` in the **joint** form
   `P(x,y,z)P(z) = P(x,z)P(y,z)` — one group per `(y,s) ∈ {0,1}^{|Y|}×{0,1}^{|S|}`, with
   the `S = ∅` variant `P(x,y) = P(x)P(y)`; record the two encoding facts: only the
   `x = 1` slice is needed (binary `X` has one d.o.f. per `(y,s)` cell), and iterating **all
   `Y` configurations jointly** is required because a pairwise/per-element decomposition
   strictly under-constrains when `|Y| ≥ 2` (cite `[[lmc-joint-encoding]]`; this is the
   same encoding as `ExactInference` and `verify_lmc_factorization`). Then the objective,
   then `min`/`max`.

   Define `_marginal_matrix` properly, since every separator row depends on it: with
   `π_{C→S}` the restriction map on cluster states, `M^{C→S}_{r,j} = 𝟙[π_{C→S}(j) = r]`,
   so `(M q_C)[r] = (q_C↾S)(r)`. Add the remark that a Type-2 row is **vacuously satisfied
   when `P(ψ) = 0`** — this is the cleared reading of `properties.tex` eq. `cleared`, and it
   is why the doc must state which reading it targets (`rem:cleared`).
   Point out that with no evidence and no hosted LMC groups the program degenerates to a
   plain **LP** — a useful sanity anchor for the reader.

5. **Running example** (`examples/ktree_fr_k2_n4_example.lcn`), fully worked, using the
   verified numbers above:
   - The four sentences and the derived families
     (`x1`; `x2|x1`; `x3|x1,x2`; `x0|x1,x3`), the moral graph, the sole LMC.
   - A **TikZ figure**: DAG/moral graph on the left, junction tree on the right with
     clusters and separators labelled (reuse the tikz style of `properties.tex` /
     `cjt_exactness.tex` Fig. `fig:instance`).
   - The **explicit NLP**: name the 22 variables, then write every constraint block —
     4 simplex, the 10 separator equalities (transcribe them from the verified
     `describe_detailed` output, which prints one equality per separator state), the 8
     sentence rows, and note `n_lmc = 0` **with the reason** (RIP-implied by `{x1,x3}`).
   - The **solution table**: CredalJT vs `ExactInference(global)` on all four atoms,
     showing agreement; plus the realized separator messages at one optimum to make the
     mechanism concrete.
   - Why two atoms are vacuous `[0,1]` (`x0` and `x3` are simply unconstrained in the
     relevant direction) — pre-empt the reader's suspicion of a bug. `properties.tex`
     handles the analogous case in `ex:fr5` by pointing to explicit feasible witnesses;
     do the same rather than asserting it.
   - **A second, contrasting structure table** (measured by the agents, to be re-verified
     before publication) showing the machinery on non-k-tree inputs:
     `examples/asia.lcn` — 5 clusters, max 3 atoms, 4 edges, 30 vars, 33 constraints
     (5 simplex / 14 separator / 10 sentence / **4 LMC**), 10 nonlinear; chain-shaped
     `{x}—{D,X}—{C,D,X}—{B,C,D}—{B,S,C}`. And `examples/alarm.lcn` — 4 clusters, max 3
     atoms, 22 vars, 24 constraints (4/8/10/**2**), 6 nonlinear, illustrating a **compound
     node** (`C-D` → cluster atoms `{A,C,D}`) and a hosted *marginal* LMC `(B ⟂ E)` with
     `S = ∅`. These two earn their place: unlike the running example they have `n_lmc > 0`,
     so they show the bilinear rows actually being imposed rather than RIP-discharged.

6. **Exactness theory** (consolidated from `properties.tex` §5 `sec:d5`, lines
   1065–1281, transplanted verbatim where possible and re-numbered locally).

   **Dependency closure (verified):** the D5 proof needs exactly `lem:locality` and
   `lem:rip` and *nothing else*. It does **not** need `def:realizable`, `prop:incl`,
   `prop:suff`, `lem:bauer`, `lem:fractional` or `thm:cve`. So this section is closed by:
   `\DLCN` (eqs. `cleared`/`lmc`/`dlcn`), `def:char`, `lem:locality`, `def:d5`, `lem:rip`,
   `thm:d5`, `rem:d5reading`, `rem:denfloor`, `cor:ktree`, `rem:ktreehonest`,
   `ex:d5verified`. `eq:factor`/`eq:se` and Gap A/B are needed *only* for the comparison
   subsection — this is what makes the document genuinely standalone, and it means §2 can
   stay short.

   Contents, in order:
   - `Definition` (cluster-feasible set `\mathcal G`, hosting, separator-entailment,
     read-off `\readoff`, projection `\pi`) — `def:d5`.
   - `Lemma` locality (`lem:locality`, `properties.tex` §1.2) — *referenced* by the D5
     proof but stated elsewhere, so the standalone doc must **restate it with its proof**
     (three cases: Type-1, Type-2, LMC; it is short and transplants verbatim).
   - `Lemma` RIP reconstruction **with zeros** (`lem:rip`) + full proof — including the
     two repairs the source flags: uniqueness fails with zero separator cells, and the
     quotient formula's division is handled by the kernel construction. Keep the whole
     induction.
   - `Theorem` CredalJT exactness under H1–H4 (`thm:d5`) + full proof, both halves:
     projection (needs only H1, H4 → **unconditional outer bound**) and reconstruction
     (needs H2, H3; uses factorization ⇒ global Markov, no positivity needed, Lauritzen
     Prop. 3.8, plus decomposition of CI).
   - `Remark` reading the proof (`rem:d5reading`): never mentions chains; never unsound;
     H3 sufficient-not-necessary (asia, polytree exact though H3 fails and does not
     bind); smokers is the loose case, `P(F1)=[0,1]`.
   - `Remark` the **denominator floor is a soundness hypothesis** (`rem:denfloor`) — for
     conditional queries the outer-bound guarantee holds only for joints with
     `P(e) ≥ ε`; if `min P(e) < ε` the interval can be **unsound**. Must not be softened.
   - `Corollary` exactness on `ktree-fr` at `O(n 2^{k+1})` (`cor:ktree`) + proof
     (k-tree chordal, moralization adds no edge, triangulation a no-op, `G* =` the
     k-tree, clusters `=` the `(k+1)`-cliques ⇒ H1; sentences root-marginal or
     child+parents ⇒ H2; LMC conditioning sets are clique separators ⇒ H3; H4 trivial).
   - `Remark` what the corollary does **not** claim (`rem:ktreehonest`): it is about
     **cost**, not accuracy — CVE is also exact on the loopy `k=2` instance, and at
     `k=3` CVE fails by *not finishing* (stalls in `Potential.prune`), not by looseness.
     The CredalJT-vs-CVE separation on this family is **tractability**.
     ⚠ This *retracts* the framing in the `[[credaljt-exact-on-ktree-fr]]` memory ("the one
     loopy topology where a treewidth-bounded engine beats CVE/IBP"). Follow
     `properties.tex`, not the memory. **Post-task:** update that memory file so the stale
     claim stops propagating.
   - Positioning table: CredalJT vs `ExactInference` (global/local), CVE, CCTE,
     ApproxLP, IBP, IJGP, ARIEL — targets / character / exact-when / cost, plus the
     by-topology table row set. Adapt from `tab:topo` and the master table, keeping the
     smokers row where CredalJT is **out** and honest about it.

7. **A self-contained paper subsection** (§ "For inclusion in a paper"). One page, no
   forward references, no dependence on the rest of the file: the algorithm in a
   paragraph, the NLP compactly, the theorem statement with H1–H4 (proof deferred to an
   appendix pointer), the complexity, one sentence on the running example's verification.
   Wrap it in a clearly delimited block (comment markers) so it can be lifted verbatim.

8. **Reproducing the numbers.** The exact commands used, so the doc is auditable — the
   `CredalNetworkVertices.from_lcn(..., solve_families=False, enumerate_vertices=False)`
   + `CredalJT(cnv).run(solver="scip", verbosity=2)` snippet, and the
   `ExactInference(l).run(solver="global")` cross-check. Note `uv run` currently fails
   to resolve (scipy 3.12 pin vs `requires-python >=3.10`), so `.venv/bin/python` is the
   working interpreter — a footnote, not a fix.

9. **Provenance** section, mirroring `properties.tex`'s discipline: every interval in
   the document was produced by running the shipped engines on the stated instance, and
   values attributed to `ExactInference(global)` are reported exact only where the
   per-atom status is `confirmed` at gap 0.

## Writing standards

- ICLR register: precise, impersonal, no hedging, no marketing adjectives. Define every
  symbol before use. Theorem/proof discipline; no proof sketches where a full proof fits.
- Every claim either proved, or attributed, or marked as measured (with the instance).
- Preserve the source's intellectual honesty — the smokers looseness, the denominator-floor
  soundness caveat, and `rem:ktreehonest` are load-bearing and must survive the rewrite.

## Deliverables & verification

1. Write `docs/cjt.tex`.
2. Compile twice (for refs): `pdflatex -interaction=nonstopmode -output-directory=docs
   docs/cjt.tex`; confirm zero unresolved `??` references and no layout-breaking overfull
   boxes. No bibtex step (no bibliography, by convention).
3. Re-run the verification commands and confirm **every** number in the document matches:
   the running example (CredalJT + `ExactInference(global)`), and the asia/alarm
   `nlp_stats` figures in the secondary table. Any number that cannot be reproduced gets
   removed, not rounded.

## Out of scope

- No changes to `junction_nlp.py` or any other source file.
- No deletion/edit of `cjt_exactness.tex` or `properties.tex`.
- No new experiments beyond re-running the two verification commands above.
