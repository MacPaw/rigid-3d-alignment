import logging

from etils import epath

import asset_vla.shared.normalize as _normalize


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str, split: str) -> dict[str, _normalize.NormStats] | None:
    base = epath.Path(assets_dir) / asset_id

    if split is not None:
        base = base / split

    norm_stats_dir = base
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats
