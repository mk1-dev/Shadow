import numpy as np
import pytest
import torch

from shadow_occ.annotation.illumination import (
    directional_distribution, direction_inputs, geographic_position, light_prior_bonus,
    prior_record, rotate_between, solar_ray, sunlight_context, weak_light_target,
)
from shadow_occ.annotation.light_learning import light_loss
from shadow_occ.annotation.physics_labels import light_bank
from shadow_occ.models.illumination import GeoTemporalLightNet


def test_solar_azimuth_and_light_propagation_have_opposite_horizontal_directions():
    np.testing.assert_allclose(solar_ray(90, 45), [-2**-.5, 0, -2**-.5], atol=1e-8)
    np.testing.assert_allclose(solar_ray(0, 45), [0, -2**-.5, -2**-.5], atol=1e-8)


def test_official_map_origin_and_projected_east_north_offsets():
    pytest.importorskip("pyproj")
    latitude, longitude = geographic_position("singapore-onenorth", [0, 0])
    assert latitude == pytest.approx(1.2882100868743724)
    assert longitude == pytest.approx(103.78475189208984)
    assert geographic_position("singapore-onenorth", [100, 0])[1] > longitude
    assert geographic_position("singapore-onenorth", [0, 100])[0] > latitude


def test_microsecond_utc_and_slow_solar_drift():
    pytest.importorskip("pvlib")
    pytest.importorskip("pyproj")
    from datetime import datetime, timezone
    stamp = int(datetime(2018, 7, 12, 4, tzinfo=timezone.utc).timestamp()*1e6)
    result = sunlight_context("singapore-onenorth", [100, 100], [stamp, stamp+500000, stamp+1000000])
    assert result["direct_sun_possible"] and result["timestamp_utc"][0].startswith("2018-07-12T04:")
    rays = np.asarray(result["rays_world"])
    assert (rays[:, 2] < 0).all()
    assert np.linalg.norm(rays[-1]-rays[0]) < .001


def test_unknown_location_does_not_invent_sunlight():
    assert not sunlight_context(None, [0, 0], [0, 1, 2])["available"]
    target = weak_light_target(light_bank(), [], {"available": False})
    assert target["weight"] == 0 and target["confidence"] == 0


def test_uniform_light_prior_is_neutral_and_confident_prior_only_adds_ranking_bonus():
    prior = {"probability": [1/96]*96, "confidence": .9}
    assert light_prior_bonus(prior, 0, .15) == pytest.approx(0)
    prior["probability"] = [.9]+[.1/95]*95
    assert 0 < light_prior_bonus(prior, 0, .15) <= .15
    assert light_prior_bonus(prior, 1, .15) < 0


def test_model_inputs_and_distribution_cover_image_geo_and_temporal_evidence():
    directions = light_bank()
    projection = np.c_[np.eye(3), np.zeros(3)]
    features, metadata = direction_inputs(directions, [{"projection_world_to_image": projection}]*3,
        {"available": False}, [True, False], [.5, .5])
    assert features.shape == (96, 12) and metadata.shape == (9,)
    model = GeoTemporalLightNet()
    output = model(torch.randn(2, 9, 96, 160), torch.randn(2, 7, 80, 80),
        torch.tensor(np.stack([metadata]*2)), torch.tensor(np.stack([features]*2)))
    batch = {"target": torch.full((2, 96), 1/96), "weight": torch.ones(2), "confidence": torch.full((2,), .2)}
    loss = light_loss(output, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.image[0].weight.grad is not None


def test_prior_keeps_all_directions_and_uses_identical_middle_frame_bank():
    directions = light_bank()
    sun = {"available": True, "direct_sun_possible": True,
           "rays_world": [solar_ray(100+offset, 45).tolist() for offset in (-.001, 0, .001)]}
    row = {"id": "w", "scene_name": "s", "sample_tokens": ["0", "1", "2"]}
    prior = prior_record(row, directional_distribution(directions, sun["rays_world"][1]), .5, sun, directions, {})
    assert len(prior["probability"]) == 96 and sum(prior["probability"]) == pytest.approx(1)
    np.testing.assert_allclose(prior["directions_by_frame_world"][1], directions, atol=1e-8)
    np.testing.assert_allclose(rotate_between(np.array([1., 0, 0]), np.array([-1., 0, 0])) @ [1, 0, 0], [-1, 0, 0], atol=1e-7)
