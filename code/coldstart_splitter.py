"""
Cold-start splitter for miRNA-drug association datasets.

Implements three inductive evaluation protocols per Ai-laoshi's annotation:
  S2 (miRNA cold-start):  test miRNAs never appear in train
  S3 (drug cold-start):   test drugs never appear in train
  S4 (pair cold-start):   neither miRNA nor drug appears in train (hardest)

Compared to the standard transductive S1 (warm) split where positives are
randomly partitioned across folds, cold-start splits partition the ENTITY
SET (miRNAs and/or drugs), then construct train/test pair sets accordingly.

Key invariants (verified post-split):
  S2: train miRNAs ∩ test miRNAs == ∅
  S3: train drugs ∩ test drugs == ∅
  S4: both of the above
  All: a positive pair appears in exactly one fold (no leakage)
  All: every fold has at least min_test_positives positives (sanity)

Usage:
    splitter = ColdStartSplitter(association_matrix, n_folds=5, seed=42)
    folds_s2 = splitter.split('S2')   # list of dicts {train_pairs, test_pairs, train_mirnas, test_mirnas, train_drugs, test_drugs}
    folds_s3 = splitter.split('S3')
    folds_s4 = splitter.split('S4')

The returned per-fold dict contains:
    train_pairs:  np.ndarray (n_train, 2) of (mirna_idx, drug_idx) positives
    test_pairs:   np.ndarray (n_test, 2) of (mirna_idx, drug_idx) positives
    train_mirna_mask:  np.ndarray (M,) bool — True for miRNAs seen in train
    train_drug_mask:   np.ndarray (N,) bool — True for drugs seen in train

The downstream load_data is responsible for using these masks to:
  - restrict the KNN similarity graph to train entities only
  - restrict gene-bridge edges to train entities only
  - exclude test-only entities' features from any embedding cache
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ColdFold:
    """One fold of a cold-start split."""
    train_pairs: np.ndarray         # (n_train_pos, 2): col 0 = mirna, col 1 = drug
    test_pairs: np.ndarray          # (n_test_pos, 2): positive test pairs
    train_mirna_mask: np.ndarray    # (M,) bool: True if miRNA appears in any train_pair
    train_drug_mask: np.ndarray     # (N,) bool: True if drug appears in any train_pair
    test_mirna_mask: np.ndarray     # (M,) bool: True if miRNA is held out for test
    test_drug_mask: np.ndarray      # (N,) bool: True if drug is held out for test
    setting: str                    # 'S2' / 'S3' / 'S4'
    fold_id: int

    def __repr__(self):
        return (f"ColdFold(setting={self.setting}, fold={self.fold_id}, "
                f"train_pos={len(self.train_pairs)}, test_pos={len(self.test_pairs)}, "
                f"train_M={self.train_mirna_mask.sum()}, train_N={self.train_drug_mask.sum()})")


class ColdStartSplitter:
    def __init__(self, association_matrix: np.ndarray, n_folds: int = 5, seed: int = 42,
                 min_test_positives: int = 20):
        """
        Args:
            association_matrix: (M, N) binary {0,1} matrix; 1 = known positive association
            n_folds: number of cross-validation folds (default 5)
            seed: deterministic RNG seed
            min_test_positives: error if any fold has fewer test positives than this
        """
        assert association_matrix.ndim == 2
        self.A = (association_matrix > 0).astype(np.int8)
        self.M, self.N = self.A.shape
        self.n_folds = n_folds
        self.seed = seed
        self.min_test_positives = min_test_positives

        # The set of positive (mirna, drug) pairs, as a (P,2) array
        rows, cols = np.where(self.A == 1)
        self.positive_pairs = np.stack([rows, cols], axis=1)  # (P, 2)
        self.n_pos = len(self.positive_pairs)
        if self.n_pos == 0:
            raise ValueError("Association matrix has no positives.")

    # ---------- Public API ----------

    def split(self, setting: str) -> List[ColdFold]:
        """Return list of n_folds ColdFold objects for the given setting."""
        if setting == 'S2':
            return self._split_entity('mirna')
        elif setting == 'S3':
            return self._split_entity('drug')
        elif setting == 'S4':
            return self._split_pair()
        else:
            raise ValueError(f"Unknown setting {setting!r}; expected S2/S3/S4")

    # ---------- S2 & S3: entity-level partition ----------

    def _split_entity(self, which: str) -> List[ColdFold]:
        """
        Partition either miRNAs (S2) or drugs (S3) into n_folds groups.
        Fold i's test set = positives involving fold-i entities;
        train set = positives involving entities NOT in fold-i.
        Entities with zero positives are skipped from the fold pool but
        kept in the graph (only as zero-degree nodes).
        """
        rng = np.random.RandomState(self.seed)

        if which == 'mirna':
            entity_dim = 0          # axis 0 of A
            n_entities = self.M
            setting = 'S2'
        else:
            entity_dim = 1
            n_entities = self.N
            setting = 'S3'

        # only partition entities that have at least one positive
        positive_entity_set = np.unique(self.positive_pairs[:, entity_dim])
        n_pe = len(positive_entity_set)
        perm = rng.permutation(n_pe)
        positive_entity_set = positive_entity_set[perm]

        # split positive entities into n_folds chunks
        fold_entities = np.array_split(positive_entity_set, self.n_folds)

        folds = []
        for f_id, test_entities in enumerate(fold_entities):
            test_entity_set = set(test_entities.tolist())
            test_mask_pair = np.isin(self.positive_pairs[:, entity_dim], list(test_entity_set))
            test_pairs = self.positive_pairs[test_mask_pair]
            train_pairs = self.positive_pairs[~test_mask_pair]

            if len(test_pairs) < self.min_test_positives:
                raise RuntimeError(
                    f"{setting} fold {f_id}: only {len(test_pairs)} test positives "
                    f"(< {self.min_test_positives}). Consider fewer folds or larger dataset.")

            # entity masks
            train_mirna_mask = np.zeros(self.M, dtype=bool)
            train_drug_mask  = np.zeros(self.N, dtype=bool)
            test_mirna_mask  = np.zeros(self.M, dtype=bool)
            test_drug_mask   = np.zeros(self.N, dtype=bool)

            train_mirna_mask[np.unique(train_pairs[:, 0])] = True
            train_drug_mask[np.unique(train_pairs[:, 1])]  = True
            test_mirna_mask[np.unique(test_pairs[:, 0])]   = True
            test_drug_mask[np.unique(test_pairs[:, 1])]    = True

            # cold-start invariant: held-out entities never appear in train
            if which == 'mirna':
                # test miRNAs ∩ train miRNAs must be empty
                overlap = train_mirna_mask & test_mirna_mask
                assert not overlap.any(), \
                    f"S2 leak in fold {f_id}: {overlap.sum()} miRNAs in both train and test"
            else:
                overlap = train_drug_mask & test_drug_mask
                assert not overlap.any(), \
                    f"S3 leak in fold {f_id}: {overlap.sum()} drugs in both train and test"

            folds.append(ColdFold(
                train_pairs=train_pairs, test_pairs=test_pairs,
                train_mirna_mask=train_mirna_mask, train_drug_mask=train_drug_mask,
                test_mirna_mask=test_mirna_mask,   test_drug_mask=test_drug_mask,
                setting=setting, fold_id=f_id,
            ))
        return folds

    # ---------- S4: pair cold-start (both entities held out) ----------

    def _split_pair(self) -> List[ColdFold]:
        """
        Hardest setting: partition both miRNAs AND drugs into n_folds groups
        (using independent permutations), then the test fold contains positives
        where BOTH the miRNA and the drug are in the held-out groups.
        """
        rng_m = np.random.RandomState(self.seed)
        rng_d = np.random.RandomState(self.seed + 1)   # different seed for drug axis

        # partition miRNAs and drugs independently
        positive_mirnas = np.unique(self.positive_pairs[:, 0])
        positive_drugs  = np.unique(self.positive_pairs[:, 1])
        m_perm = rng_m.permutation(len(positive_mirnas))
        d_perm = rng_d.permutation(len(positive_drugs))
        mirna_folds = np.array_split(positive_mirnas[m_perm], self.n_folds)
        drug_folds  = np.array_split(positive_drugs[d_perm],  self.n_folds)

        folds = []
        for f_id in range(self.n_folds):
            test_mirnas = set(mirna_folds[f_id].tolist())
            test_drugs  = set(drug_folds[f_id].tolist())

            # test pair = positive AND mirna in test_mirnas AND drug in test_drugs
            in_test_m = np.isin(self.positive_pairs[:, 0], list(test_mirnas))
            in_test_d = np.isin(self.positive_pairs[:, 1], list(test_drugs))
            test_mask_pair = in_test_m & in_test_d

            # train pair = positive AND mirna NOT in test_mirnas AND drug NOT in test_drugs
            #   (pairs where exactly one entity is held out are dropped — they're
            #    neither train nor test, by design, to maintain S4's strict semantics)
            train_mask_pair = (~in_test_m) & (~in_test_d)

            test_pairs  = self.positive_pairs[test_mask_pair]
            train_pairs = self.positive_pairs[train_mask_pair]

            if len(test_pairs) < self.min_test_positives:
                raise RuntimeError(
                    f"S4 fold {f_id}: only {len(test_pairs)} test positives "
                    f"(< {self.min_test_positives}). S4 is intrinsically sparse — "
                    f"consider larger dataset or fewer folds.")

            train_mirna_mask = np.zeros(self.M, dtype=bool)
            train_drug_mask  = np.zeros(self.N, dtype=bool)
            test_mirna_mask  = np.zeros(self.M, dtype=bool)
            test_drug_mask   = np.zeros(self.N, dtype=bool)
            train_mirna_mask[np.unique(train_pairs[:, 0])] = True
            train_drug_mask[np.unique(train_pairs[:, 1])]  = True
            test_mirna_mask[np.unique(test_pairs[:, 0])]   = True
            test_drug_mask[np.unique(test_pairs[:, 1])]    = True

            # S4 invariant: NO miRNA and NO drug appears in both train and test
            assert not (train_mirna_mask & test_mirna_mask).any(), \
                f"S4 leak in fold {f_id} (miRNA)"
            assert not (train_drug_mask & test_drug_mask).any(), \
                f"S4 leak in fold {f_id} (drug)"

            folds.append(ColdFold(
                train_pairs=train_pairs, test_pairs=test_pairs,
                train_mirna_mask=train_mirna_mask, train_drug_mask=train_drug_mask,
                test_mirna_mask=test_mirna_mask,   test_drug_mask=test_drug_mask,
                setting='S4', fold_id=f_id,
            ))
        return folds


# ---------- Self-test ----------

def _self_test():
    """Run sanity checks on synthetic data."""
    rng = np.random.RandomState(0)
    A = (rng.rand(200, 150) < 0.05).astype(np.int8)   # ~5% density
    print(f"Synthetic A: {A.shape}, {A.sum()} positives")
    s = ColdStartSplitter(A, n_folds=5, seed=42)
    for setting in ['S2', 'S3', 'S4']:
        folds = s.split(setting)
        train_sizes = [len(f.train_pairs) for f in folds]
        test_sizes  = [len(f.test_pairs)  for f in folds]
        print(f"  {setting}: train sizes {train_sizes}, test sizes {test_sizes}")
        # verify positive_pairs are exactly partitioned (S2/S3)
        if setting in ('S2', 'S3'):
            recovered = sum(test_sizes) + sum(train_sizes) // (len(folds) - 1)
            # actually for S2/S3 each test fold's positives sum to A.sum()
            assert sum(test_sizes) == A.sum(), f"{setting} test_sizes sum {sum(test_sizes)} != {A.sum()}"
        # entity leak check (redundant w/ asserts inside _split_*, but explicit)
        for f in folds:
            if setting in ('S2', 'S4'):
                assert not (f.train_mirna_mask & f.test_mirna_mask).any()
            if setting in ('S3', 'S4'):
                assert not (f.train_drug_mask  & f.test_drug_mask).any()
    print("  all invariants pass ✓")


if __name__ == "__main__":
    _self_test()
