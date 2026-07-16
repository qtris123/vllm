# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.config.mm_prefix import maybe_disable_mm_prefix_lm_for_text_only
from vllm.config.model_arch import ModelArchitectureConfig


def _fake_arch(*, is_mm_prefix_lm: bool = True) -> ModelArchitectureConfig:
    return ModelArchitectureConfig(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4",
        text_model_type=None,
        hidden_size=1,
        total_num_hidden_layers=1,
        total_num_attention_heads=1,
        head_size=1,
        vocab_size=1,
        total_num_kv_heads=1,
        num_experts=0,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=is_mm_prefix_lm,
        rswa_window=None,
        derived_max_model_len_and_key=(8192.0, "max_position_embeddings"),
    )


def _fake_model_config(
    *,
    is_mm_prefix_lm: bool = True,
    language_model_only: bool = False,
    limit_per_prompt: dict | None = None,
    multimodal_config=None,
):
    model_config = MagicMock()
    model_config.model_arch_config = _fake_arch(is_mm_prefix_lm=is_mm_prefix_lm)
    model_config.get_multimodal_config.return_value = multimodal_config
    if multimodal_config is not None:
        multimodal_config.language_model_only = language_model_only
        multimodal_config.limit_per_prompt = limit_per_prompt or {}
        multimodal_config.get_limit_per_prompt = (
            lambda modality: multimodal_config.limit_per_prompt.get(modality, 999)
            if not language_model_only
            else 0
        )
    return model_config


def _mock_processor(*, supported: dict[str, int | None], allowed: dict[str, int]):
    processor = MagicMock()
    processor.info.supported_mm_limits = supported
    processor.info.allowed_mm_limits = allowed
    return processor


@pytest.mark.parametrize("language_model_only", [True, False])
def test_keeps_mm_prefix_when_vision_modalities_enabled(language_model_only: bool):
    if language_model_only:
        pytest.skip("language_model_only always disables mm_prefix")

    mm_config = MagicMock()
    mm_config.language_model_only = False
    mm_config.limit_per_prompt = {"image": 1, "video": 1}
    mm_config.get_limit_per_prompt = MagicMock(side_effect=lambda m: 1)

    model_config = _fake_model_config(multimodal_config=mm_config)
    processor = _mock_processor(
        supported={"image": None, "video": None},
        allowed={"image": 1, "video": 1},
    )

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.create_processor",
        return_value=processor,
    ):
        maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config.is_mm_prefix_lm is True


def test_disables_mm_prefix_for_language_model_only():
    mm_config = MagicMock()
    mm_config.language_model_only = True

    model_config = _fake_model_config(multimodal_config=mm_config)

    maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config.is_mm_prefix_lm is False


def test_disables_mm_prefix_when_all_supported_vision_limits_are_zero():
    mm_config = MagicMock()
    mm_config.language_model_only = False
    mm_config.limit_per_prompt = {"image": 0, "video": 0}
    mm_config.get_limit_per_prompt = MagicMock(side_effect=lambda m: 0)

    model_config = _fake_model_config(multimodal_config=mm_config)
    processor = _mock_processor(
        supported={"image": None, "video": None},
        allowed={"image": 0, "video": 0},
    )

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.create_processor",
        return_value=processor,
    ):
        maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config.is_mm_prefix_lm is False


def test_image_only_model_disables_with_image_limit_zero():
    # Gemma 3 style: video not in supported modalities.
    mm_config = MagicMock()
    mm_config.language_model_only = False
    mm_config.limit_per_prompt = {"image": 0}
    mm_config.get_limit_per_prompt = MagicMock(side_effect=lambda m: 0)

    model_config = _fake_model_config(multimodal_config=mm_config)
    processor = _mock_processor(
        supported={"image": None},
        allowed={"image": 0},
    )

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.create_processor",
        return_value=processor,
    ):
        maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config.is_mm_prefix_lm is False


def test_gemma4_keeps_mm_prefix_when_only_image_disabled():
    mm_config = MagicMock()
    mm_config.language_model_only = False
    mm_config.limit_per_prompt = {"image": 0}
    mm_config.get_limit_per_prompt = MagicMock(
        side_effect=lambda m: 0 if m == "image" else 999
    )

    model_config = _fake_model_config(multimodal_config=mm_config)
    processor = _mock_processor(
        supported={"image": None, "video": None, "audio": None},
        allowed={"image": 0, "video": 999, "audio": 999},
    )

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.create_processor",
        return_value=processor,
    ):
        maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config.is_mm_prefix_lm is True


def test_noop_when_not_mm_prefix_lm():
    arch = _fake_arch(is_mm_prefix_lm=False)
    model_config = MagicMock()
    model_config.model_arch_config = arch
    model_config.get_multimodal_config.return_value = MagicMock()

    maybe_disable_mm_prefix_lm_for_text_only(model_config)

    assert model_config.model_arch_config is arch
