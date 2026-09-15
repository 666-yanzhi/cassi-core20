"""Scene-macro validation adapter for the externally supplied RunnerV3."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import SequentialSampler

import training


class _SceneErrorAccumulator:
    """Online uniform-overlap fusion of standardized prediction errors."""

    def __init__(self, scene: str, height: int, width: int) -> None:
        self.scene = scene
        self.error_mean = np.zeros((height, width, training.INDEX_CHANNELS), dtype=np.float32)
        self.validity = np.zeros((height, width, training.INDEX_CHANNELS), dtype=np.bool_)
        self.coverage_count = np.zeros((height, width), dtype=np.uint16)

    def add(
        self,
        sample: dict[str, Any],
        prediction_chw: torch.Tensor,
        supervision_chw: torch.Tensor,
    ) -> None:
        target, validity = training.split_supervision(supervision_chw.unsqueeze(0))
        prediction = prediction_chw.detach().cpu().numpy().transpose(1, 2, 0)
        target_hwc = target.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        validity_hwc = validity.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(
                f"validation prediction contains NaN or Inf: {sample['sample_id']}"
            )
        top, left = int(sample["top"]), int(sample["left"])
        height, width = int(sample["height"]), int(sample["width"])
        if height <= 0 or width <= 0:
            raise ValueError(f"nonpositive validation patch extent: {sample['sample_id']}")
        if prediction.shape[0] < height or prediction.shape[1] < width:
            raise ValueError(f"validation prediction is smaller than patch: {sample['sample_id']}")
        if top < 0 or left < 0:
            raise ValueError(f"negative validation patch coordinate: {sample['sample_id']}")
        bottom, right = top + height, left + width
        if bottom > self.error_mean.shape[0] or right > self.error_mean.shape[1]:
            raise ValueError(f"validation patch exceeds scene canvas: {sample['sample_id']}")
        patch_error = (prediction[:height, :width] - target_hwc[:height, :width]).astype(
            np.float32, copy=False
        )
        patch_validity = validity_hwc[:height, :width]
        region = self.error_mean[top:bottom, left:right]
        validity_region = self.validity[top:bottom, left:right]
        counts = self.coverage_count[top:bottom, left:right]
        unseen = counts == 0
        if np.any(~unseen) and not np.array_equal(
            validity_region[~unseen], patch_validity[~unseen]
        ):
            raise ValueError(
                f"overlapping patches disagree on fixed label validity: {sample['sample_id']}"
            )
        region[unseen] = patch_error[unseen]
        if np.any(~unseen):
            old_counts = counts[~unseen].astype(np.float32)[:, None]
            region[~unseen] = (
                region[~unseen] * old_counts + patch_error[~unseen]
            ) / (old_counts + np.float32(1.0))
        if np.any(counts == np.iinfo(np.uint16).max):
            raise OverflowError("validation overlap count exceeds uint16 capacity")
        counts += np.uint16(1)
        validity_region |= patch_validity

    def macro_zmae(self) -> float:
        channel_values: list[float] = []
        for channel in range(training.INDEX_CHANNELS):
            mask = self.validity[..., channel] & (self.coverage_count > 0)
            if np.any(mask):
                errors = np.asarray(self.error_mean[..., channel][mask], dtype=np.float64)
                if not np.isfinite(errors).all():
                    raise FloatingPointError(
                        f"stitched validation error is non-finite for {self.scene}"
                    )
                channel_values.append(float(np.mean(np.abs(errors), dtype=np.float64)))
        if not channel_values:
            raise ValueError(f"validation scene has no valid supervision: {self.scene}")
        return float(np.mean(np.asarray(channel_values, dtype=np.float64)))


class SceneMacroValidationMixin:
    """Override only validation aggregation; inherit the RunnerV3 lifecycle."""

    @torch.no_grad()
    def _eval(self, loader: Any) -> tuple[float, float]:
        dataset = getattr(loader, "dataset", None)
        samples = getattr(dataset, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise TypeError("scene-macro validation requires Core20Dataset.samples")
        if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
            raise ValueError("scene-macro validation loader must use sequential sampling")
        expected_order = sorted(
            samples,
            key=lambda item: (item["scene"], item["top"], item["left"], item["sample_id"]),
        )
        if samples != expected_order:
            raise ValueError("scene-macro validation samples must be grouped and coordinate-sorted")
        scene_shapes: dict[str, tuple[int, int]] = {}
        for sample in samples:
            scene = sample["scene"]
            height = int(sample["top"]) + int(sample["height"])
            width = int(sample["left"]) + int(sample["width"])
            old_height, old_width = scene_shapes.get(scene, (0, 0))
            scene_shapes[scene] = (max(old_height, height), max(old_width, width))

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        cursor = 0
        current_scene: str | None = None
        accumulator: _SceneErrorAccumulator | None = None
        scene_metrics: list[float] = []
        for batch in loader:
            *inputs, supervision = self._to_device(batch)
            prediction = self.model(*inputs)
            batch_size = inputs[0].size(0)
            total_loss += self.loss_fn(prediction, supervision).item() * batch_size
            total_samples += batch_size
            if cursor + batch_size > len(samples):
                raise ValueError("validation loader yielded more samples than its dataset")
            for batch_index in range(batch_size):
                sample = samples[cursor + batch_index]
                scene = sample["scene"]
                if scene != current_scene:
                    if accumulator is not None:
                        scene_metrics.append(accumulator.macro_zmae())
                    current_scene = scene
                    accumulator = _SceneErrorAccumulator(scene, *scene_shapes[scene])
                if accumulator is None:
                    raise AssertionError("scene accumulator was not initialized")
                accumulator.add(sample, prediction[batch_index], supervision[batch_index])
            cursor += batch_size
        if cursor != len(samples) or total_samples == 0:
            raise ValueError("validation loader did not yield every dataset sample exactly once")
        if accumulator is not None:
            scene_metrics.append(accumulator.macro_zmae())
        return total_loss / total_samples, float(
            np.mean(np.asarray(scene_metrics, dtype=np.float64))
        )


def create_scene_macro_runner_class(runner_v3_base: type[Any]) -> type[Any]:
    """Create ``SceneMacroRunnerV3(RunnerV3)`` without hard-coding its location."""

    if not isinstance(runner_v3_base, type):
        raise TypeError("RunnerV3 base must be a class")
    return type(
        "SceneMacroRunnerV3",
        (SceneMacroValidationMixin, runner_v3_base),
        {"__module__": __name__},
    )
