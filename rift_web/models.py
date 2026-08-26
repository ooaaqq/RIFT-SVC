"""Model choices shared by the web API and serial dispatcher."""

from __future__ import annotations

MODEL_CATALOG: dict[str, list[dict[str, object]]] = {
    "rift": [
        {
            "id": "rift25k",
            "label": "露早 RIFT 25k",
            "description": "露早专用的 RIFT 音色转换模型。",
            "sources": [
                {
                    "label": "露早 RIFT 25k",
                    "url": "https://huggingface.co/ooaaqq/rift-svc-luzao-25k",
                }
            ],
        }
    ],
    "background": [
        {
            "id": "big-beta7",
            "label": "Big Beta 7",
            "description": "单模型提取干声，适合作为自然、完整的独立候选。",
            "profile": "big-beta7",
            "sources": [
                {
                    "label": "Big Beta 7",
                    "url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big",
                }
            ],
        },
        {
            "id": "bs-roformer-mag",
            "label": "BS-RoFormer Mag",
            "description": "偏重幅度谱准确度，适合为音色转换准备干声。",
            "profile": "bs-roformer-mag",
            "sources": [
                {
                    "label": "BS-RoFormer Mag",
                    "url": "https://huggingface.co/anvuew/BS_RoFormer_mag",
                }
            ],
        },
        {
            "id": "big-beta7-bs-roformer-mag-max-spec",
            "label": "Big Beta 7 + BS-RoFormer Mag（Max Spec）",
            "description": "两模型使用 Max Spec 合成，兼顾干声完整度与残留控制。",
            "profile": "big-beta7-bs-roformer-mag-max-spec",
            "sources": [
                {
                    "label": "Big Beta 7",
                    "url": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big",
                },
                {
                    "label": "BS-RoFormer Mag",
                    "url": "https://huggingface.co/anvuew/BS_RoFormer_mag",
                },
            ],
        }
    ],
    "deharmony": [
        {
            "id": "anvuew-karaoke",
            "label": "Anvuew Karaoke",
            "description": "主唱与和声分离的常用默认候选。",
            "profile": "anvuew-karaoke",
            "sources": [
                {
                    "label": "Anvuew Karaoke BS-RoFormer",
                    "url": "https://huggingface.co/anvuew/karaoke_bs_roformer",
                }
            ],
        },
        {
            "id": "becruily-frazer-karaoke",
            "label": "Becruily Frazer Karaoke",
            "description": "用于对比主唱完整度与和声残留的另一种 Karaoke 候选。",
            "profile": "becruily-frazer-karaoke",
            "sources": [
                {
                    "label": "Becruily Frazer BS-RoFormer",
                    "url": "https://huggingface.co/becruily/bs-roformer-karaoke",
                }
            ],
        },
        {
            "id": "small-karaoke-gaboxaufr",
            "label": "Small Karaoke GaboxAUFR",
            "description": "较轻量的 Karaoke 模型候选。",
            "profile": "small-karaoke-gaboxaufr",
            "sources": [
                {
                    "label": "Small Karaoke GaboxAUFR",
                    "url": "https://huggingface.co/GaboxR67/MelBandRoformers",
                }
            ],
        },
    ],
    "dereverb": [
        {
            "id": "anvuew-dereverb-22.5050",
            "label": "Anvuew Dereverb 22.5050",
            "description": "从人声或其他 stem 中分离混响残留。",
            "profile": "anvuew-dereverb-22.5050",
            "sources": [
                {
                    "label": "Anvuew Dereverb BS-RoFormer",
                    "url": "https://huggingface.co/anvuew/dereverb_bs_roformer",
                }
            ],
        }
    ],
}

DEFAULT_MODEL = {
    "rift": "rift25k",
    "background": "big-beta7-bs-roformer-mag-max-spec",
    "deharmony": "anvuew-karaoke",
    "dereverb": "anvuew-dereverb-22.5050",
}


def get_model(kind: str, model_id: str | None = None) -> dict[str, object] | None:
    """Resolve one model, using the workflow default for legacy jobs."""
    selected = model_id or str(DEFAULT_MODEL.get(kind, ""))
    return next(
        (model for model in MODEL_CATALOG.get(kind, []) if model["id"] == selected),
        None,
    )


def public_catalog() -> dict[str, list[dict[str, object]]]:
    """Return browser-safe model metadata without internal profile names."""
    return {
        kind: [
            {
                "id": model["id"],
                "label": model["label"],
                "description": model["description"],
                "sources": model["sources"],
            }
            for model in sorted(
                models, key=lambda item: item["id"] != DEFAULT_MODEL[kind]
            )
        ]
        for kind, models in MODEL_CATALOG.items()
    }
