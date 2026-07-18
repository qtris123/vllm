#!/usr/bin/env python3
"""Apply the mm_prefix text-only gate onto an installed vLLM tree."""

from __future__ import annotations

from pathlib import Path

import vllm

CALL_NEEDLE = (
    "                    \"disable the cache with --mm-processor-cache-gb 0.\"\n"
    "                )\n"
    "\n"
    "        if self.disable_sliding_window:"
)
CALL_REPL = (
    "                    \"disable the cache with --mm-processor-cache-gb 0.\"\n"
    "                )\n"
    "\n"
    "            self._apply_mm_prefix_lm_limits()\n"
    "\n"
    "        if self.disable_sliding_window:"
)

METHOD = '''
    def _apply_mm_prefix_lm_limits(self) -> None:
        """Disable multimodal prefix attention for text-only serving."""
        arch = self.model_arch_config
        if not arch.is_mm_prefix_lm:
            return

        mm_config = self.get_multimodal_config()
        if mm_config is None:
            return

        if mm_config.language_model_only:
            reason = "--language-model-only is set"
        else:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.get_processing_info(self)
            vision_modalities = {"image", "video"} & info.supported_mm_limits.keys()
            if not vision_modalities or any(
                info.allowed_mm_limits[modality] > 0
                for modality in vision_modalities
            ):
                return
            reason = (
                "all supported vision modalities are limited to zero via "
                "--limit-mm-per-prompt"
            )

        self.model_arch_config = replace(arch, is_mm_prefix_lm=False)
        logger.info_once(
            "Disabled mm_prefix attention mode because %s. Attention backends without "
            "mm_prefix support may now be selected.",
            reason,
        )

'''

INSERT_AT = "    def get_model_arch_config(\n"


def main() -> None:
    path = Path(vllm.__file__).parent / "config" / "model.py"
    text = path.read_text()
    if "def _apply_mm_prefix_lm_limits" in text:
        print("PATCHED_REFACTOR_MM_PREFIX_GATE already present", path)
        return

    if "from dataclasses import InitVar, field\n" in text:
        import_line = text.split("from dataclasses import", 1)[1].split("\n", 1)[0]
        if "replace" not in import_line:
            text = text.replace(
                "from dataclasses import InitVar, field\n",
                "from dataclasses import InitVar, field, replace\n",
                1,
            )

    if CALL_NEEDLE not in text:
        raise SystemExit("call insertion point missing")
    if INSERT_AT not in text:
        raise SystemExit("method insertion point missing")

    text = text.replace(CALL_NEEDLE, CALL_REPL, 1)
    text = text.replace(INSERT_AT, METHOD + INSERT_AT, 1)
    path.write_text(text)
    print("PATCHED_REFACTOR_MM_PREFIX_GATE", path)


if __name__ == "__main__":
    main()
