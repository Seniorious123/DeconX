# DeconX

**Iterative residual-based discovery of missing cell types from bulk RNA-seq.**

DeconX is a deep-learning method that, given a single-cell RNA-seq reference and
a set of bulk RNA-seq samples, **discovers cell types that are present in the
bulk but absent (technically missing or biologically rare) from the reference**.
The core idea is iterative: subtract the contribution of known cell types,
analyse the residual, propose a new candidate cell type, retrain, and repeat
until two stopping criteria are satisfied.

## Why DeconX

Most deconvolution tools (CIBERSORT, MuSiC, DWLS, BayesPrism, ...) assume the
single-cell reference covers every cell type in the bulk. When this assumption
fails — single-nucleus references lose adipocytes, rare populations like
plasmablasts are underrepresented — the deconvolved proportions are biased and
the missing types are silently invisible.

DeconX explicitly models the residual signal that conventional methods discard,
identifies which gene programs persist after known-type subtraction, and
attributes each surviving program to a candidate cell type from a user-supplied
signature library.

## At a glance

|                                       | Conventional deconvolution | NMF / residual-NMF | DeconX |
|---------------------------------------|:---:|:---:|:---:|
| Estimates known-cell-type proportions | yes | partial | yes |
| Discovers missing cell types          | no  | yes (with #missing known a priori) | **yes (auto-stop)** |
| Needs ground-truth proportions        | no  | yes (Pearson match in simulation) | no |
| Works on real bulk (no ground truth)  | yes | partial (GO only) | **yes (signature similarity)** |

## Where to start

- New here? Read the [Quickstart](quickstart.md).
- Bringing your own data? Check [Input format](user_guide/input_format.md).
- Tuning runs? See [Hyperparameters](user_guide/hyperparameters.md) and
  [Stopping criteria](user_guide/stopping_criteria.md).
- Worked examples end to end? Try the
  [Simulation walkthrough](tutorials/simulation.md) or the
  [Real HGSOC walkthrough](tutorials/real_hgsoc.md).
- Common questions: the [FAQ](faq.md).

## Citing DeconX

If you use DeconX in your research, please cite:

> Fan Y. et al., *DeconX: discovering missing cell types from bulk RNA-seq via
> iterative residual distillation*. (Submitted, 2026.)

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

user_guide/input_format
user_guide/building_signatures
user_guide/hyperparameters
user_guide/stopping_criteria
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/simulation
tutorials/real_hgsoc
```

```{toctree}
:hidden:
:caption: Reference

faq
changelog
```
