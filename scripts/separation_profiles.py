"""Explicit model profiles supported by the Kaggle separation workflow."""

from __future__ import annotations

MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training.git"
MSST_COMMIT = "e247dfe4abc1f17c69dff719207fe045dc04413a"

PROFILES = {
    "big-beta7-bs-roformer-mag-max-spec": {
        "label": "big-beta7-bs-roformer-mag-max-spec",
        "ensemble_algorithm": "max_fft",
        "models": [
            {
                "label": "unwa-big-beta7-mel-roformer",
                "model_repo": "pcunwa/Mel-Band-Roformer-big",
                "model_revision": "1508d1ed7c54cb0017b2cbfaabdaf3ca87d2cf74",
                "checkpoint": "big_beta7.ckpt",
                "config": "big_beta7.yaml",
                "architecture": "mel_band_roformer",
            },
            {
                "label": "anvuew-bs-roformer-mag",
                "model_repo": "anvuew/BS_RoFormer_mag",
                "model_revision": "d4dd2390ed97f80b0d0278b220fcff677483227b",
                "checkpoint": "bs_roformer_mag_anvuew.ckpt",
                "config": "config.yaml",
                "architecture": "bs_roformer",
            },
        ],
    },
    "anvuew-dereverb-22.5050": {
        "label": "anvuew-dereverb-bs-roformer-22.5050",
        "model_repo": "anvuew/dereverb_bs_roformer",
        "model_revision": "bd5c6b55a429b4b74ce85fe5dcb690cfe36d91ec",
        "checkpoint": "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        "config": "config.yaml",
        "architecture": "bs_roformer",
        "rename_instrumental": "reverb-residual",
    },
    "anvuew-karaoke": {
        "label": "anvuew-karaoke-bs-roformer",
        "model_repo": "anvuew/karaoke_bs_roformer",
        "model_revision": "0d4423d42e12cf2ba39ae09171028507b8a2a7be",
        "checkpoint": "karaoke_bs_roformer_anvuew.ckpt",
        "config": "karaoke_bs_roformer_anvuew.yaml",
        "architecture": "bs_roformer",
    },
    "becruily-frazer-karaoke": {
        "label": "becruily-frazer-bs-roformer-karaoke",
        "model_repo": "becruily/bs-roformer-karaoke",
        "model_revision": "f7849ae934209184dc288d1018cd8a76a7fc8b3c",
        "checkpoint": "bs_roformer_karaoke_frazer_becruily.ckpt",
        "config": "config_karaoke_frazer_becruily.yaml",
        "architecture": "bs_roformer",
    },
    "small-karaoke-gaboxaufr": {
        "label": "small-karaoke-gaboxaufr",
        "model_repo": "GaboxR67/MelBandRoformers",
        "model_revision": "7860d31d8cc1c73b46c1cbdc2b497210c97478ae",
        "checkpoint": "melbandroformers/karaoke/small_karaoke_gaboxaufr.ckpt",
        "config": "melbandroformers/karaoke/config_karaoke_small.yaml",
        "architecture": "mel_band_roformer",
        "normalize_small_config": True,
    },
}
