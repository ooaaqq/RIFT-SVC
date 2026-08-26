"""Model choices shared by the web API and serial dispatcher."""

from __future__ import annotations

MODEL_CATALOG: dict[str, list[dict[str, object]]] = {
    "rift": [
        {
            "id": "rift25k",
            "label": "RIFT 25K",
            "sources": [
                {
                    "label": "RIFT 25K",
                    "url": "https://huggingface.co/ooaaqq/rift-svc-luzao-25k",
                }
            ],
        }
    ],
    "background": [
        {
            "id": "big-beta7-bs-roformer-mag-max-spec",
            "label": "Big Beta 7 + BS-RoFormer Mag（Max Spec）",
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

DEFAULT_MODEL = {kind: models[0]["id"] for kind, models in MODEL_CATALOG.items()}


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
                "sources": model["sources"],
            }
            for model in models
        ]
        for kind, models in MODEL_CATALOG.items()
    }
