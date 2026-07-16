# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_num_threads

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)

# Modalities whose tokens use bidirectional (prefix-LM) attention when
# is_mm_prefix_lm is enabled on the model architecture.
_MM_PREFIX_VISION_MODALITIES = ("image", "video")


def maybe_disable_mm_prefix_lm_for_text_only(model_config: ModelConfig) -> None:
    """Clear ``is_mm_prefix_lm`` when vision inputs cannot appear in requests.

    Gemma 4 and other prefix-LM multimodal models set ``is_mm_prefix_lm`` from
    static HF config. That flag is used at startup to reject attention backends
    that do not implement multimodal prefix attention — even for text-only
    workloads that never exercise the bidirectional vision-token path.

    When the user disables vision modalities via ``--limit-mm-per-prompt`` or
    ``--language-model-only``, downgrade to text-only attention backend
    selection so fast backends (e.g. FlashInfer on Blackwell NVFP4) remain
    available.
    """
    arch = model_config.model_arch_config
    if not arch.is_mm_prefix_lm:
        return

    mm_config = model_config.get_multimodal_config()
    if mm_config is None:
        return

    if mm_config.language_model_only:
        model_config.model_arch_config = replace(arch, is_mm_prefix_lm=False)
        logger.info_once(
            "Disabled mm_prefix attention mode because --language-model-only "
            "is set. Fast attention backends remain available for text-only "
            "serving."
        )
        return

    from vllm.multimodal import MULTIMODAL_REGISTRY

    with set_default_torch_num_threads():
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        allowed_limits = processor.info.allowed_mm_limits
        supported_modalities = processor.info.supported_mm_limits

    vision_enabled = any(
        allowed_limits.get(modality, 0) > 0
        for modality in _MM_PREFIX_VISION_MODALITIES
        if modality in supported_modalities
    )
    if vision_enabled:
        return

    model_config.model_arch_config = replace(arch, is_mm_prefix_lm=False)
    logger.info_once(
        "Disabled mm_prefix attention mode because all supported vision "
        "modalities are limited to zero via --limit-mm-per-prompt. Fast "
        "attention backends remain available for text-only serving."
    )
