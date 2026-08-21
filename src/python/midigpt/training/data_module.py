"""LightningDataModule wrapping MidiGPTDataset + MidiGPTCollator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch.utils.data import DataLoader

from midigpt.training.collator import MidiGPTCollator
from midigpt.training.dataset import MidiGPTDataset

if TYPE_CHECKING:
    from midigpt.augmentation.mask_bar import MaskBarConfig
    from midigpt.tokenizer.tokenizer import Tokenizer


class MidiGPTDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str | list[str],
        tokenizer: Tokenizer,
        # Infill training
        infill_probability: float = 0.75,
        infill_bar_fraction: float = 0.5,
        # Humanize training (independent of infill)
        humanize_probability: float = 0.0,
        humanize_bar_fraction: float = 0.5,
        # Bar masking (independent of infill)
        mask_bar_config: MaskBarConfig | None = None,
        # Sequence budget
        max_seq_len: int = 2048,
        # Window / track sampling
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
        # DataLoader params
        per_device_batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        # Mechanical context / structured targeting (independent of infill;
        # see MidiGPTDataset for the full explanation of each).
        context_mechanical_fraction: float = 0.0,
        structured_target_probability: float = 0.0,
        mechanical_coherent_targets: bool = False,
        mechanical_coherent_residual: float = 0.0,
        # Optional eval
        eval_path: str | None = None,
    ):
        super().__init__()
        self._train_path = train_path
        self._eval_path = eval_path
        self._tokenizer = tokenizer
        self._infill_prob = infill_probability
        self._infill_frac = infill_bar_fraction
        self._humanize_prob = humanize_probability
        self._humanize_frac = humanize_bar_fraction
        self._mask_cfg = mask_bar_config
        self._max_seq_len = max_seq_len
        self._max_tracks = max_tracks
        self._min_tracks = min_tracks
        self._fill_ratio = min_fill_ratio
        self._batch_size = per_device_batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._context_mech_frac = context_mechanical_fraction
        self._structured_target_prob = structured_target_probability
        self._mech_coherent = mechanical_coherent_targets
        self._mech_coherent_residual = mechanical_coherent_residual
        self._collator = MidiGPTCollator(max_seq_len=max_seq_len)
        self._train_ds: MidiGPTDataset | None = None
        self._eval_ds: MidiGPTDataset | None = None

    def _task_shaping_kwargs(self) -> dict:
        """Shared task-distribution kwargs for both train and eval datasets --
        val/loss must reflect the SAME task the model is actually trained on
        (previously it silently didn't: eval was built with
        humanize_probability defaulting to 0.0 regardless of what the run
        configured, so val/loss measured plain autoregressive reconstruction
        even for humanize_probability=1.0 runs -- a different task than
        training, making it meaningless for model selection/early stopping)."""
        return dict(
            infill_probability=self._infill_prob,
            infill_bar_fraction=self._infill_frac,
            humanize_probability=self._humanize_prob,
            humanize_bar_fraction=self._humanize_frac,
            mask_bar_config=self._mask_cfg,
            max_tracks=self._max_tracks,
            min_tracks=self._min_tracks,
            min_fill_ratio=self._fill_ratio,
            context_mechanical_fraction=self._context_mech_frac,
            structured_target_probability=self._structured_target_prob,
            mechanical_coherent_targets=self._mech_coherent,
            mechanical_coherent_residual=self._mech_coherent_residual,
        )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = MidiGPTDataset(
                self._train_path,
                self._tokenizer,
                max_seq_len=self._max_seq_len,
                **self._task_shaping_kwargs(),
            )
        if self._eval_path and self._eval_ds is None:
            self._eval_ds = MidiGPTDataset(
                self._eval_path,
                self._tokenizer,
                max_seq_len=self._max_seq_len,
                **self._task_shaping_kwargs(),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collator,
            persistent_workers=self._num_workers > 0,
            multiprocessing_context="spawn" if self._num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._eval_ds is None:
            return None
        return DataLoader(
            self._eval_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collator,
            persistent_workers=self._num_workers > 0,
            multiprocessing_context="spawn" if self._num_workers > 0 else None,
        )

    @property
    def train_dataset_size(self) -> int:
        """Number of training samples (available after setup())."""
        return len(self._train_ds) if self._train_ds else 0
