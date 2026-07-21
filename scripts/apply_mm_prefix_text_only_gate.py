#!/usr/bin/env python3
"""Apply DarkLight-style mm_prefix gate onto an installed vLLM tree."""

from __future__ import annotations

from pathlib import Path

import vllm

CALL_NEEDLE = (
    '                    "disable the cache with --mm-processor-cache-gb 0."\n'
    "                )\n"
    "\n"
    "        if self.disable_sliding_window:"
)
CALL_REPL = (
    '                    "disable the cache with --mm-processor-cache-gb 0."\n'
    "                )\n"
    "\n"
    "            # Rebuild after multimodal_config exists so mm_prefix limits apply.\n"
    "            self.model_arch_config = self.get_model_arch_config()\n"
    "\n"
    "        if self.disable_sliding_window:"
)

METHOD = '''\
    def _apply_mm_prefix_lm_limits(
        self, arch: ModelArchitectureConfig
    ) -> ModelArchitectureConfig:
        """Disable multimodal prefix attention for text-only serving.

        Applied inside ``get_model_arch_config`` so every regenerated arch
        config preserves the derived flag (including ``with_hf_config``).
        No-op until ``multimodal_config`` is initialized.
        """
        if not arch.is_mm_prefix_lm:
            return arch

        mm_config = self.multimodal_config
        if mm_config is None:
            return arch

        if mm_config.language_model_only:
            reason = "--language-model-only is set"
        else:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.get_processing_info(self)
            vision_modalities = {"image", "video"} & info.supported_mm_limits.keys()
            if not vision_modalities or any(
                info.allowed_mm_limits[modality] > 0 for modality in vision_modalities
            ):
                return arch
            reason = (
                "all supported vision modalities are limited to zero via "
                "--limit-mm-per-prompt"
            )

        logger.info_once(
            "Disabled mm_prefix attention mode because %s. Attention backends without "
            "mm_prefix support may now be selected.",
            reason,
        )
        return replace(arch, is_mm_prefix_lm=False)

    def get_model_arch_config(
'''

OLD_GET = (
    "    def get_model_arch_config(\n"
    "        self,\n"
    "    ) -> ModelArchitectureConfig:\n"
    "        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(\n"
    "            self.hf_config.model_type, ModelArchConfigConvertorBase\n"
    "        )\n"
    "        convertor = convertor_cls(self.hf_config, self.hf_text_config)\n"
    "        return convertor.convert()\n"
)
NEW_GET = (
    "    def get_model_arch_config(\n"
    "        self,\n"
    "    ) -> ModelArchitectureConfig:\n"
    "        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(\n"
    "            self.hf_config.model_type, ModelArchConfigConvertorBase\n"
    "        )\n"
    "        convertor = convertor_cls(self.hf_config, self.hf_text_config)\n"
    "        return self._apply_mm_prefix_lm_limits(convertor.convert())\n"
)


def main() -> None:
    path = Path(vllm.__file__).resolve().parent / "config" / "model.py"
    text = path.read_text()

    if "_apply_mm_prefix_lm_limits" in text and "self._apply_mm_prefix_lm_limits(convertor.convert())" in text:
        print("PATCHED_MM_PREFIX_ARCH_ALREADY", path)
        return

    if "from dataclasses import InitVar, field, replace" not in text:
        if "from dataclasses import InitVar, field\n" not in text:
            raise SystemExit(f"dataclasses import needle missing in {path}")
        text = text.replace(
            "from dataclasses import InitVar, field\n",
            "from dataclasses import InitVar, field, replace\n",
            1,
        )

    if CALL_NEEDLE not in text:
        raise SystemExit(f"call needle missing in {path}")
    text = text.replace(CALL_NEEDLE, CALL_REPL, 1)

    if OLD_GET not in text:
        raise SystemExit(f"get_model_arch_config needle missing in {path}")
    # Insert helper immediately before get_model_arch_config, then rewrite return.
    text = text.replace(OLD_GET, METHOD + NEW_GET.removeprefix("    def get_model_arch_config(\n"), 1)

    path.write_text(text)
    print("PATCHED_MM_PREFIX_ARCH", path)


if __name__ == "__main__":
    main()
