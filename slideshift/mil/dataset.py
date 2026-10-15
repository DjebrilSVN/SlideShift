"""Patient-aware and site-stratified datasets for Multiple-Instance Learning."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from slideshift.core.types import SlideBag
from slideshift.features.extractor import SlideFeatureExtractor
from slideshift.core.logging import logger


class SlideBagDataset:
    """Dataset storing slide bags with patient grouping and site awareness."""

    def __init__(self, bags: List[SlideBag]):
        self.bags = bags

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> SlideBag:
        return self.bags[idx]

    @property
    def labels(self) -> np.ndarray:
        return np.array([b.label if b.label is not None else -1 for b in self.bags], dtype=np.int64)

    @property
    def patient_ids(self) -> List[str]:
        return [b.patient_id or b.slide_id for b in self.bags]

    @property
    def sites(self) -> List[str]:
        return [b.site or "unknown_site" for b in self.bags]

    @property
    def scanners(self) -> List[str]:
        return [b.scanner or "unknown_scanner" for b in self.bags]

    @classmethod
    def from_directory(
        cls,
        feature_dir: Union[str, Path],
        metadata_csv: Optional[Union[str, Path]] = None,
    ) -> "SlideBagDataset":
        """Load all feature bag files (.h5, .pt, .npz) in a directory."""
        dir_path = Path(feature_dir)
        files = list(dir_path.glob("*.h5")) + list(dir_path.glob("*.pt")) + list(dir_path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No feature bag files (.h5, .pt, .npz) found in {feature_dir}")

        meta_lookup: Dict[str, Dict[str, any]] = {}
        if metadata_csv:
            import pandas as pd
            df = pd.read_csv(metadata_csv)
            for _, row in df.iterrows():
                sid = str(row["slide_id"])
                meta_lookup[sid] = row.to_dict()

        bags: List[SlideBag] = []
        for f in files:
            bag = SlideFeatureExtractor.load_bag(f)
            # Update with metadata csv if provided
            if bag.slide_id in meta_lookup:
                m = meta_lookup[bag.slide_id]
                if "label" in m and bag.label is None:
                    bag.label = int(m["label"])
                if "patient_id" in m and bag.patient_id is None:
                    bag.patient_id = str(m["patient_id"])
                if "site" in m and bag.site is None:
                    bag.site = str(m["site"])
                if "scanner" in m and bag.scanner is None:
                    bag.scanner = str(m["scanner"])
            bags.append(bag)

        return cls(bags)

    def create_splits(
        self,
        strategy: str = "patient_stratified",
        n_splits: int = 5,
        test_size: float = 0.20,
        val_size: float = 0.15,
        held_out_site: Optional[str] = None,
        seed: int = 42,
    ) -> List[Tuple["SlideBagDataset", "SlideBagDataset", "SlideBagDataset"]]:
        """Create leakage-free train, validation, and test splits.

        Args:
            strategy:
                'patient_stratified': Groups by patient to ensure no patient appears in both
                                      train and test, while stratifying by label.
                'site_held_out': Holds out an entire hospital / site for zero-shot site transfer testing.
                'random_stratified': Stratified without patient grouping (only if 1 slide per patient).
            n_splits: Number of cross-validation folds.
            held_out_site: Specific site name to hold out if strategy == 'site_held_out'.

        Returns:
            List of (train_dataset, val_dataset, test_dataset) folds.
        """
        y = self.labels
        patients = self.patient_ids
        sites = self.sites

        if strategy == "site_held_out":
            if not held_out_site:
                unique_sites = sorted(list(set(sites)))
                held_out_site = unique_sites[0]
            logger.info(f"Creating site-held-out split: test site is '{held_out_site}'.")

            test_indices = [i for i, s in enumerate(sites) if s == held_out_site]
            non_test_indices = [i for i, s in enumerate(sites) if s != held_out_site]

            # Split non-test into train and val (grouped by patient)
            non_test_patients = [patients[i] for i in non_test_indices]
            non_test_y = y[non_test_indices]

            sgkf = StratifiedGroupKFold(n_splits=max(2, int(1.0 / val_size)), shuffle=True, random_state=seed)
            train_sub_idx, val_sub_idx = next(sgkf.split(non_test_indices, non_test_y, non_test_patients))

            train_idx = [non_test_indices[i] for i in train_sub_idx]
            val_idx = [non_test_indices[i] for i in val_sub_idx]

            return [(
                SlideBagDataset([self.bags[i] for i in train_idx]),
                SlideBagDataset([self.bags[i] for i in val_idx]),
                SlideBagDataset([self.bags[i] for i in test_indices]),
            )]

        elif strategy == "patient_stratified":
            class_counts = np.bincount(y[y >= 0])
            min_class = int(np.min(class_counts)) if len(class_counts) > 0 else 1
            k_outer = min(n_splits, max(2, min_class))
            try:
                sgkf = StratifiedGroupKFold(n_splits=k_outer, shuffle=True, random_state=seed)
                outer_splits = list(sgkf.split(self.bags, y, patients))
            except Exception:
                gkf = GroupKFold(n_splits=min(k_outer, len(set(patients))))
                outer_splits = list(gkf.split(self.bags, y, patients))

            folds = []
            for train_val_idx, test_idx in outer_splits:
                tv_y = y[train_val_idx]
                tv_patients = [patients[i] for i in train_val_idx]
                tv_class_counts = np.bincount(tv_y[tv_y >= 0]) if len(tv_y[tv_y >= 0]) > 0 else np.array([1])
                min_tv_class = int(np.min(tv_class_counts)) if len(tv_class_counts) > 0 else 1
                k_inner = min(3, max(2, min_tv_class))

                try:
                    inner_sgkf = StratifiedGroupKFold(n_splits=k_inner, shuffle=True, random_state=seed)
                    tr_sub, val_sub = next(inner_sgkf.split(train_val_idx, tv_y, tv_patients))
                except Exception:
                    # Deterministic split if stratification cannot satisfy class constraints
                    split_pt = max(1, int(len(train_val_idx) * 0.75))
                    tr_sub = list(range(split_pt))
                    val_sub = list(range(split_pt, len(train_val_idx)))

                final_train = [train_val_idx[i] for i in tr_sub]
                final_val = [train_val_idx[i] for i in val_sub]

                folds.append((
                    SlideBagDataset([self.bags[i] for i in final_train]),
                    SlideBagDataset([self.bags[i] for i in final_val]),
                    SlideBagDataset([self.bags[i] for i in test_idx]),
                ))
            return folds

        else:
            # Simple stratified
            k_outer = min(n_splits, max(2, len(self.bags)))
            skf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=seed)
            folds = []
            for tr_idx, te_idx in skf.split(self.bags, y):
                k_inner = min(5, max(2, len(tr_idx)))
                inner_skf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed)
                tr_sub, val_sub = next(inner_skf.split(tr_idx, y[tr_idx]))
                folds.append((
                    SlideBagDataset([self.bags[tr_idx[i]] for i in tr_sub]),
                    SlideBagDataset([self.bags[tr_idx[i]] for i in val_sub]),
                    SlideBagDataset([self.bags[i] for i in te_idx]),
                ))
            return folds
