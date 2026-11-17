# Revision plan v4: fit the main paper into 10 pages, with the changes in blue

Baseline: the current `iclr_2027.tex` (HEAD 9d7f267). The submitted version is
`git show d83bce5:docs/research-chain-lcn/iclr2027/iclr_2027.tex`.

## 0. Where things stand

The main text currently runs to about **12.75 pages**: the Conclusion ends three quarters
of the way down p. 12, and the AI use, ethics and reproducibility statements follow.
To fit 10 pages, **about 2.8 pages have to go**. The aim is 9.8 pages, which leaves some slack.

| Part | Pages now (from the .aux) | Target | Saving |
|---|---|---|---|
| Abstract + Intro (§1) | p1 – mid p2 (~1.3) | ~1.0 | −0.3 |
| Background (§2) | mid p2 – top p5 (~2.6) | ~2.2 | −0.4 |
| CJT (§3) | p5 – p6 (~1.3) | ~1.1 | −0.2 |
| §4.1–4.2 compilation + family-realizability, Fig. 2 | p6 – p8 (~1.9) | ~1.5 | −0.4 |
| §4.3–4.5 CVE / IVBP / CDVE | p8 – p10 (~1.8) | ~0.5 | **−1.3** |
| Experiments (§5) | p10 – p12 (~2.1) | ~1.8 | −0.3 |
| Related work + Conclusion | p12 (~0.75) | ~0.55 | −0.2 |
| **Total** | **~12.75** | **~9.7** | **−3.1** |

Rules:
- Do not add new spacing hacks. The existing `\addtolength{\textfloatsep}{-16pt}` and
  `\tabcolsep` stay as they are, and the ICLR style is not touched.
- Nothing is deleted outright. Every fact removed from the main text either moves to the
  appendix (merged into the matching section, not pasted as a duplicate) or already
  appears there.

## 1. How the blue highlighting works

Preamble (after `xcolor`):
```latex
\newcommand{\rev}[1]{{\color{blue}#1}}            % inline: words, sentences, math
\newenvironment{revblock}{\color{blue}}{}          % paragraphs, theorems, lists
```
- Use `\rev{}` inside paragraphs and `revblock` for whole paragraphs, theorem
  environments and enumerations. Avoid `soul`'s `\hl`, which breaks on math and `\cite`.
- **Figures and tables:** colour the caption blue when the figure or table is new, or
  when its content changed. `main_results.pdf` and `results-summary.tex` are both new
  since the submission.
- **Section headings:** colour a heading only when the section itself is new.
- **Diff aid:** run
  `latexdiff --flatten /tmp/sub.tex iclr_2027.tex > /tmp/diff.tex`
  to find the changed regions. Then apply `\rev`/`revblock` by hand. The latexdiff
  output is not shipped, because its red strike-throughs would cost pages.
- **What counts as a change (proposed; see open question Q1):** text whose content is new
  or materially different from the submission. That covers new results, theorems and
  hypotheses, corrected claims, new numbers and new benchmarks. It does not cover pure
  rewording or text that was only moved (for example, the CN background that moved from
  §3 to §2.1). Moved-but-unchanged text stays black. Without this rule almost the whole
  main text would be blue, because the section order changed: the submission had the CN
  route in §3 and CJT in §4.
- Expected blue regions in the main text:
  - the abstract (CJT-first framing, 740 instances, certified numbers);
  - the contribution paragraphs;
  - the cleared-form sentence semantics (eq. `cleared`), the conditional-query definition,
    and the chain-graph validity check;
  - all of §3 that is new relative to the submitted CJT section: H1–H4 as checked
    hypotheses, the ordered tree, reduction, Charnes–Cooper, SCIP certification;
  - §4.1: exact row polytopes for compound children, the infeasible-row convention, and
    Prop. `inclusion` restricted to factorizing LCNs;
  - §4.2: the Type-2 half of family-realizability (R2), and Prop. `suff` with its
    hypotheses (i)–(iii);
  - the guarantee statements of the condensed §4.3 (CVE exactness under evidence, IVBP
    exact on compound polytrees, CDVE inner);
  - all of §5 (rerun);
  - the conclusion.

## 2. Changes by section

### Abstract (~330 words → ~170 words)
Proposed text (to be blue):

> Logical Credal Networks (LCNs) combine propositional formulas with imprecise probability
> bounds. Exact inference has so far meant a nonconvex program over all $2^n$ joint states,
> and the only approximate method, ARIEL, carries no guarantee. We introduce Credal
> Junction Tree (CJT), an exact algorithm for LCNs whose structure is a chain graph. CJT
> replaces the joint by the cluster marginals of a junction tree, imposes each sentence and
> independence assertion on one cluster, and solves a nonconvex program with
> $O(n\,2^{w+1})$ variables, where $w+1$ is the largest cluster size. Under conditions
> that it checks mechanically, the global optimum of this program gives the exact lower
> and upper probabilities. As a complementary route, we compile LCNs into credal
> networks, characterize the family-realizable LCNs on which this compilation preserves
> the exact bounds, and adapt three credal-network algorithms to the result. On $740$
> synthetic LCNs with up to $100$ propositions, CJT with a global solver certifies every
> endpoint on trees, polytrees and $3$-trees, and $82$–$99\%$ of them on DAGs. The joint
> formulation is certified only up to about six propositions. On family-realizable LCNs
> the compiled methods match the certified bounds within seconds. Outside that class they
> remain sound but loosen with every non-realizable sentence.

### §1 Introduction (−0.3)
- Contribution paragraph 2:
  - Cut the per-algorithm guarantee list to one clause: "adapt CVE (with its FPTAS),
    interval BP and coordinate descent, with exactness/inner guarantees
    (Appendices F–H)".
  - Shorten the experimental claims to one sentence.
- Keep the one-line pointer to the appendix.

### §2 Background (−0.4)
- **§2.1 CN:** keep eqs. `pse` and `se`. Fold the remark on the extreme points and
  Lemma `hull` into one sentence.
- **§2.2 LCN:**
  - Move the primal-graph descendant definition (`de_P`) to App. A.3 and keep only what
    Def. `lmc-lcn` needs.
  - Drop the closing "state of the art consists of two approaches" sentence, which
    repeats the introduction.
  - Shorten Example 1 to one sentence.
- **§2.3 Chain graphs:**
  - Turn Def. `lmc-cg` into an inline sentence.
  - Move the positivity and intersection-property discussion to App. A.4. Keep only
    "DAG ⇒ no positivity needed (Lemma `dag`)".
  - In the "Chain-graph LCNs" paragraph, move the implementation sentence about
    rejecting intra-component directed edges to App. A.4, and keep the definitions of
    *factorizing* and *DAG-structured*.

### §3 CJT (−0.2)
- Trim the paragraph after the example ("The hypotheses H1–H4 below are checked…") to
  three sentences. The details of checking and certification are already in App. B.3.
- Trim the discussion after Thm. `cjt-size` to two sentences: formulation size is not
  solving time, and no compilation is needed.
- Keep Thm. `cjt` and its proof sketch unchanged. They answer reviewer A2AT's soundness
  concern.

### §4.1–4.2 Compilation and family-realizability (−0.4)
- **Fig. 2:** keep (a) the credal network and (c) the local credal set at reduced
  width. Move (b), the local fractional LPs (`fig-flp.png`), to App. F's running example.
  Then shorten Example 2 to match.
- Merge the exact/outer/inner/unreliable definitions into one sentence before Prop.
  `inclusion`.
- Cut the paragraph after Prop. `suff` to two sentences:
  - the proof idea (Lemma `noGapA` + Lemma `marg`);
  - (iii) fails when the LMC constrains a family's table, and CJT is then the exact
    method; refer to App. E.4.
- Move the R1/R2 "sufficient, not necessary" remark into a footnote or to App. D.

### §4.3–4.5 → a single §4.3 "Credal-network algorithms" (−1.3)
This is the main cut. Replace the three subsections with about half a page: one short
paragraph per method plus a guarantees table. The full descriptions, complexity theorems
and proofs go to Appendices F (CVE), G (IVBP) and H (CDVE).

- **CVE:**
  - Variable elimination over sets of pairs of functions (Mauá et al.).
  - The query node is eliminated last.
  - Query-free potentials are reduced to their convex-hull vertices.
  - CVE is exact for $\DSE$ with or without evidence.
  - CVE$_\varepsilon$, the FPTAS of Mauá et al. (2012), returns inner bounds within a
    factor $1+\varepsilon$.
- **IVBP:**
  - Credal sum-product on the factor graph. Each message is the set of hull vertices of
    its normalized vectors, the factor-graph form of 2U.
  - Exact on polytree credal networks, including compound nodes and evidence.
  - A heuristic on loopy networks.
- **CDVE:**
  - Follows ApproxLP: coordinate descent over families.
  - Each step is an exact linear-fractional program over VE-computed coefficients,
    solved by Dinkelbach.
  - Inner for $\DSE$.
- **Table 2 (new, small):**

  | Method | Guarantee for $\DSE$ | For $\DLCN$ under Prop. `suff` | Cost |
  |---|---|---|---|
  | CVE | exact | exact | $O(m\,C^2 d^{w^*+1})$ |
  | CVE$_\varepsilon$ | inner within $1+\varepsilon$ | inner within $1+\varepsilon$ | as CVE, with $C$ polynomial |
  | IVBP | exact on polytrees | exact on polytrees | per-sweep, App. G |
  | CDVE | inner | inner | App. H |

- **Theorem handling:**
  - Thms. `cve`, `ibp` and Prop. `cve-fptas` move to the appendix as stated results. The
    labels are kept, so the existing `\ref`s in §5 and in the appendix proofs still
    resolve.
  - The complexity theorems `cve-cplx`, `ibp-cplx` and `cdve-cplx` move to the start of
    their appendix sections.
  - Keep one consolidated main-text statement, **Proposition (guarantees)**, that cites
    those three results.
- **Merge, don't duplicate:** before deleting each main-text paragraph, check that every
  fact in it is in the appendix. Facts to check:
  - the pair-potential definition (App. F.4 "Pair Potentials");
  - the zero-mass / empty-belief convention ($[0,1]$ when $\overline P(\be)=0$);
  - IVBP's sweep schedule ($\tau=10^{-6}$, $T=100$), the 2U equivalence and the
    $\le 2$ vertices on binary nodes;
  - IVBP's NP-hardness remark for ternary polytrees (Mauá 2014);
  - CDVE's vertex-average start with the tie move, $R=8$ restarts, barren-family pruning
    and $T=50$.

  Whatever is missing gets added to App. F/G/H, in blue there.

### §5 Experiments (−0.3)
- **Benchmarks paragraph:** keep one sentence per family of sets. The bound formula
  $[\max(0.01,v-0.3),\min(1,v+0.3)]$ and the description of the nr kinds go to App. I.1
  and I.6. Remove the leftover `% REAL:` placeholder comments.
- **Reference and metrics:** condense the exact-joint-baseline numbers to one sentence;
  they are already in App. I.3. Move the run settings ($\varepsilon=0.01$, sweep and
  restart counts) to App. I.1.
- **"CJT certifies…" and "Family-realizable…":** keep the headline numbers. Move the
  detail on the three CVE out-of-reference endpoints and the CVE memory counts to App. I.
- **Summary paragraph:** delete it and merge its message into the conclusion.

### §6–7 Related work and Conclusion (−0.2)
- Related work: one sentence less on general probabilistic logics.
- Conclusion: cut to about 60%. Drop the restated method list and keep the findings and
  next steps.
- The AI use, ethics and reproducibility statements stay after the conclusion. See open
  question Q3 on whether the AI use statement counts toward the page limit.

### Appendix
- Additions come only from the moved material above, in blue wherever the content is new
  relative to the submission.
- New appendix sections, compared with the submission, get a blue heading and a single
  blue line saying "(new in this revision)" instead of 30 blue pages (see Q2). These
  include:
  - B, the CJT detailed description;
  - C, the NLP and the worked example;
  - E.4, the counterexamples;
  - most of F–I.

## 3. Execution order and checks
1. Add the macros. Generate `/tmp/diff.tex` with latexdiff for reference.
2. Condense §4.3–4.5 and move the theorems; this is the biggest win. Then compile and
   check the page count.
3. Make the remaining section cuts in the order of the table above. Recompile after
   each one.
4. Apply the blue highlighting last, by walking through the latexdiff output.
5. Checks:
   - Add a `\label{end-of-main}` right after the Conclusion, and confirm in the `.aux`
     that it sits on **page ≤ 10**.
   - `latexmk` reports no undefined references.
   - Every `\ref` to a moved theorem still resolves.
   - Visually check pp. 1–11.

## 4. Open questions
- **Q1. What "changed" means.** Content-level (recommended) or mechanical latexdiff-level?
  The mechanical version turns nearly all of the main text blue, because the sections
  were reordered.
- **Q2. Appendix scope.** Highlight inside the appendix sentence by sentence, or mark
  whole new sections with a blue heading and a "(new)" tag (recommended)?
- **Q3. AI use statement.** Does it count toward the 10 pages under the ICLR 2027 CFP? If
  it does, it needs about 0.15 pages more of cuts, or a move after the references.
