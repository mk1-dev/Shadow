from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_occ.config import BevConfig, OccConfig, ShadowConfig
from shadow_occ.data.occ3d_loader import Occ3DSample, Occ3DLoader
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.shadow.labels import build_shadow_geometry_labels
from shadow_occ.shadow.render import render_shadow_from_height
from shadow_occ.utils.io import ensure_dir, write_json


def run(output_root: str | Path = "outputs/smoke") -> None:
    output_root = ensure_dir(output_root)
    bev_cfg = BevConfig(x_min=-20.0, x_max=20.0, y_min=-20.0, y_max=20.0, resolution=0.4)
    occ_cfg = OccConfig(
        voxel_size=(0.4, 0.4, 0.4),
        pc_range=(-20.0, -20.0, -1.0, 20.0, 20.0, 5.4),
        ground_class_ids=[11],
    )
    shadow_cfg = ShadowConfig(min_component_area=5, min_length_m=1.0, min_elongation=1.2)
    grid = BEVGrid.from_config(bev_cfg)

    semantics = np.full((100, 100, 16), occ_cfg.free_label, dtype=np.uint8)
    semantics[:, :, 2] = 11
    semantics[56:60, 49:52, 3:10] = 1
    occ = Occ3DSample(path=Path("synthetic"), semantics=semantics)
    loader = Occ3DLoader(".", occ_cfg)
    height = loader.height_map(occ, grid)

    shadow = np.zeros(grid.shape, dtype=np.uint8)
    shadow[48:52, 54:70] = 1
    rendered_shadow_prior = render_shadow_from_height(height, grid, shadow_cfg)
    labels = build_shadow_geometry_labels(shadow, height, grid, shadow_cfg)

    npz_path = output_root / "synthetic_shadow_label.npz"
    np.savez_compressed(
        npz_path,
        **labels.as_npz_dict(),
        rendered_shadow_prior=rendered_shadow_prior.astype(np.uint8),
        components=np.asarray([c.to_dict() for c in labels.components], dtype=object),
    )
    write_json(
        output_root / "synthetic_shadow_label.json",
        {
            "num_components": len(labels.components),
            "components": [c.to_dict() for c in labels.components],
        },
    )
    if len(labels.components) != 1:
        raise AssertionError(f"Expected one component, got {len(labels.components)}")
    if labels.components[0].length_m <= 1.0:
        raise AssertionError("Shadow component length was not computed correctly.")
    print(f"Synthetic smoke test passed: {npz_path}")
