import gc
import os
from pathlib import Path

import torch

from differential_uncertainty.extraction import load_frozen_detector
from differential_uncertainty.persistence import Layer2Capture, batched_persistence
from src.misc.tue_utils import (
    get_captured_persistence_diagrams,
    hook_decoder_layers,
)
from src.scene_uncertainty.runtime import (
    load_frozen_detector as load_legacy_detector,
)


DEFAULT_CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
CHECKPOINT = Path(
    os.environ.get("RTDETRV2_R18_CHECKPOINT", DEFAULT_CHECKPOINT)
)
CONFIG = Path("configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml")


def test_explicit_model_and_layer_two_persistence_equal_the_legacy_path():
    assert CHECKPOINT.is_file(), f"detector checkpoint does not exist: {CHECKPOINT}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = torch.rand(
        (1, 3, 640, 640),
        generator=torch.Generator().manual_seed(19),
    )

    legacy = load_legacy_detector(CONFIG, CHECKPOINT, device)
    captures, handles, layers = hook_decoder_layers(
        legacy.decoder,
        [2],
        "score",
    )
    try:
        with torch.inference_mode():
            legacy_outputs = legacy(samples.to(device))
        selected = [torch.arange(300, device=device)]
        legacy_persistence = torch.stack(
            list(
                get_captured_persistence_diagrams(
                    captures,
                    selected,
                    layers,
                )[2][0].values()
            )
        ).unsqueeze(0)
        legacy_logits = legacy_outputs["pred_logits"].cpu()
        legacy_boxes = legacy_outputs["pred_boxes"].cpu()
    finally:
        for handle in handles:
            handle.remove()
    del captures, legacy_outputs, legacy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    explicit = load_frozen_detector(CHECKPOINT, device)
    with torch.inference_mode(), Layer2Capture(
        explicit.decoder,
        2,
    ) as explicit_capture:
        explicit_outputs = explicit(samples.to(device))
        explicit_features, explicit_weight = explicit_capture.take()
        explicit_persistence = batched_persistence(
            explicit_weight,
            explicit_features.reshape(-1, explicit_features.shape[-1]),
        ).reshape(1, 300, 335)

    torch.testing.assert_close(
        explicit_outputs["pred_logits"].cpu(),
        legacy_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit_outputs["pred_boxes"].cpu(),
        legacy_boxes,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit_persistence.cpu(),
        legacy_persistence,
        rtol=0,
        atol=0,
    )
