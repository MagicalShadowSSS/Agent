"""Single-module structural refinement for candidate generation.

The caller supplies a frozen multimodal score matrix and receives a refined
correspondence.  Predicate matching, neighbor support, and global row/column
competition are implementation details of this one module; no labels,
learned parameters, or dataset-specific branches are used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from typed_relation_consistency import TypedRelationConsistency


def transport(score, temperature=0.05, mode="balanced"):
    """Turn an evidence matrix into a soft correspondence."""
    import torch

    if mode == "unbalanced":
        kernel = torch.exp((score - score.max()) / temperature)
        u = torch.ones(score.shape[0], device=score.device)
        v = torch.ones(score.shape[1], device=score.device)
        exponent = 1 / (1 + temperature)
        for _ in range(30):
            u = ((1 / score.shape[0]) / (kernel @ v).clamp_min(1e-30)).pow(exponent)
            v = ((1 / score.shape[1]) / (kernel.T @ u).clamp_min(1e-30)).pow(exponent)
        return kernel * u[:, None] * v[None, :]

    coupling = torch.exp((score - score.max(1).values[:, None]) / temperature)
    for _ in range(10):
        coupling /= coupling.sum(1, keepdim=True).clamp_min(1e-12)
        coupling /= coupling.sum(0, keepdim=True).clamp_min(1e-12)
    return coupling


def _normalized_graphs(graph_root: Path, dataset: str):
    import torch

    graphs = []
    manifest = json.loads(
        (graph_root / "manifests" / f"{dataset}.json").read_text()
    )
    for side, count in ((1, manifest["source_count"]), (2, manifest["target_count"])):
        graph = sp.load_npz(graph_root / "graphs" / f"{dataset}_side{side}.npz").tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        degree = np.asarray(graph.sum(1)).reshape(-1)
        norm = sp.diags(1 / np.sqrt(np.maximum(degree, 1e-12)))
        graph = (norm @ graph @ norm).tocsr()
        graphs.append(
            torch.sparse_csr_tensor(
                torch.tensor(graph.indptr, device="cuda"),
                torch.tensor(graph.indices, device="cuda"),
                torch.tensor(graph.data, device="cuda"),
                size=(count, count),
            )
        )
    return graphs, manifest


class StructureConsistencyRefiner:
    """One public refinement module for typed and graph consistency.

    ``refine`` performs one repeated update rule.  It first obtains relation
    support from the current soft correspondence, averages it with the same
    normalized one-hop support used by the graph, and immediately updates the
    correspondence using either the previous coupling or frozen multimodal
    evidence as its reference.  This is a single
    structural update rule rather than a typed stage followed by a separate
    GW stage. This interface does not make a performance claim.
    """

    def __init__(
        self,
        dataset,
        graph_root,
        typed_audit,
        model,
        out,
        *,
        infer_schema=True,
        static_schema=False,
        use_context=True,
        update="proximal",
        base_weight=None,
        proximal_weight=None,
        support_weight=1.0,
        support_calibration="none",
        audit_rounds=None,
        temperature=0.05,
        transport_mode="balanced",
    ):
        self.dataset = dataset
        self.graph_root = Path(graph_root)
        self.typed_audit = Path(typed_audit)
        self.out = Path(out)
        self.temperature = temperature
        self.transport_mode = transport_mode
        self.use_context = use_context
        self.update = update
        if base_weight is None:
            base_weight = 1.0 if update == "anchored" else 0.0
        if proximal_weight is None:
            proximal_weight = 0.0 if update == "anchored" else 1.0
        if min(base_weight, proximal_weight, support_weight) < 0:
            raise ValueError("objective weights must be non-negative")
        self.base_weight = base_weight
        self.proximal_weight = proximal_weight
        self.support_weight = support_weight
        if support_calibration not in ("none", "row_std"):
            raise ValueError("unknown support calibration")
        self.support_calibration = support_calibration
        self.audit_rounds = audit_rounds
        self.context_graphs = None
        if use_context:
            self.context_graphs, _ = _normalized_graphs(self.graph_root, dataset)
        self.typed = TypedRelationConsistency(
            dataset,
            self.graph_root,
            self.typed_audit,
            None,
            self.out,
            model,
            directional=True,
            infer_schema=infer_schema and not static_schema,
            audit_rounds=audit_rounds,
            normalize_neighbors=True,
        )

    @staticmethod
    def _normalize(signal):
        return signal.clamp_min(0) / signal.max(1).values[:, None].clamp_min(1e-12)

    def _support(self, coupling):
        import torch

        typed_support = self._normalize(self.typed.propagate(coupling))
        if not self.use_context:
            return typed_support
        left = torch.sparse.mm(self.context_graphs[0], coupling)
        context = torch.sparse.mm(self.context_graphs[1], left.T).T.contiguous()
        del left
        context = self._normalize(context)
        return (typed_support + context) * 0.5

    def iter_refinements(self, prior, observation_counts, *, steps=8, checkpoints=None):
        """Yield ``(step, coupling, structural_support, elapsed_seconds)``."""
        import torch

        checkpoints = set(checkpoints or {steps})
        coupling = transport(prior, self.temperature, self.transport_mode)
        torch.cuda.synchronize()
        elapsed = 0.0
        for step in range(1, steps + 1):
            started = time.perf_counter()
            structural = self._support(coupling) / observation_counts[:, None]
            if self.support_calibration == "row_std":
                prior_scale = prior.std(1, unbiased=False).clamp_min(1e-6)
                structural_scale = structural.std(1, unbiased=False).clamp_min(1e-6)
                structural = structural * (prior_scale / structural_scale)[:, None]
            score = (
                self.base_weight * prior
                + self.proximal_weight * self.temperature * coupling.clamp_min(1e-30).log()
                + self.support_weight * structural
            )
            coupling = transport(score, self.temperature, self.transport_mode)
            del score
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            if step in checkpoints:
                yield step, coupling, structural, elapsed

    def refine(self, prior, observation_counts, *, steps=8):
        """Return the final correspondence through one caller-facing method."""
        if steps < 1:
            raise ValueError("at least one structural update is required")
        for _, coupling, _, _ in self.iter_refinements(
            prior, observation_counts, steps=steps, checkpoints={steps}
        ):
            return coupling


class StagedStructureConsistencyRefiner:
    """Encapsulate the preserved algorithm without changing its mathematics.

    This has one caller-facing method but retains typed-relation updates and
    the following GW optimization internally. It is an interface refactor,
    not an algorithmic simplification or a new unified optimization objective.
    """

    def __init__(
        self, typed, graphs, *, temperature=0.1, transport_mode="balanced",
        relation_steps=8, gw_steps=100, relation_fraction=1.0,
        local_fraction=0.0,
    ):
        if min(relation_steps, gw_steps) < 1:
            raise ValueError("positive relation and GW iteration counts required")
        if not 0 <= relation_fraction <= 1 or not 0 <= local_fraction <= 1:
            raise ValueError("fractions must be between zero and one")
        self.typed, self.graphs = typed, graphs
        self.temperature, self.transport_mode = temperature, transport_mode
        self.relation_steps, self.gw_steps = relation_steps, gw_steps
        self.relation_fraction, self.local_fraction = relation_fraction, local_fraction

    def refine(self, prior, relation_view, observation_counts):
        """Return the preserved full-catalog candidate score matrix."""
        import torch
        from collective_candidate_control import gw_refine, gw_affinity, tail_evidence

        current = prior
        for _ in range(self.relation_steps):
            coupling = transport(current, self.temperature, self.transport_mode)
            support = self.typed.propagate(coupling)
            del coupling
            support = StructureConsistencyRefiner._normalize(support)
            current = prior + self.relation_fraction * (support - relation_view) / observation_counts[:, None]
            del support
        initial = transport(current, self.temperature, self.transport_mode)
        for _, coupling, alpha in gw_refine(initial, current, self.graphs, [self.gw_steps]):
            source_has = torch.ones(prior.shape[0], dtype=torch.bool, device=prior.device)
            target_has = torch.ones(prior.shape[1], dtype=torch.bool, device=prior.device)
            posterior = tail_evidence(coupling, source_has, target_has)
            if self.local_fraction == 0:
                return posterior
            affinity = gw_affinity(coupling, current, self.graphs, alpha)
            local = tail_evidence(affinity, source_has, target_has)
            return (1 - self.local_fraction) * posterior + self.local_fraction * local
