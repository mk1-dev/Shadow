import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from shadow_occ.annotation.segmentation import (
    SingletonSafeBatchNorm2d, build_segmenter, masked_loss, set_segmenter_train_mode,
)


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_scratch_backbone_keeps_batch_statistics_and_nonvanishing_features():
    torch.manual_seed(0)
    model = build_segmenter("none")
    set_segmenter_train_mode(model)
    assert all(module.training for module in model.backbone.modules()
               if isinstance(module, torch.nn.BatchNorm2d))
    images = torch.randn(2, 3, 64, 96)
    features = model.backbone(images)["out"]
    assert float(features.detach().std()) > .01
    assert not torch.equal(features[0], features[1])
    features.square().mean().backward()
    assert float(model.backbone["0"][0].weight.grad.abs().sum()) > 0


def test_pretrained_policy_freezes_only_backbone_statistics():
    model = build_segmenter("none")
    set_segmenter_train_mode(model, freeze_backbone_stats=True)
    backbone = [module for module in model.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    head = [module for module in model.classifier.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert backbone and head
    assert all(not module.training for module in backbone)
    assert all(module.training for module in head)
    assert all(module.weight.requires_grad for module in backbone)
    model.eval()
    set_segmenter_train_mode(model, freeze_backbone_stats=True)
    assert all(module.training for module in head)


def test_singleton_pooling_is_finite_and_trains_affine_without_changing_statistics():
    norm = SingletonSafeBatchNorm2d(4).train()
    values = torch.randn(1, 4, 1, 1, requires_grad=True)
    result = norm(values)
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(values.grad).all()
    assert norm.weight.grad.abs().sum() > 0
    assert norm.num_batches_tracked.item() == 0
    norm(torch.randn(2, 4, 1, 1))
    assert norm.num_batches_tracked.item() == 1


def test_actual_segmenter_supports_batch_one_backward_and_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = build_segmenter("none")
    set_segmenter_train_mode(model)
    image = torch.randn(1, 3, 64, 96)
    target = torch.zeros(1, 64, 96, dtype=torch.long)
    target[:, :, 48:] = 1
    logits = model(image)["out"]
    loss = masked_loss(logits, target, .5)
    loss.backward()
    gradient = model.backbone["0"][0].weight.grad
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert any(isinstance(module, SingletonSafeBatchNorm2d) for module in model.classifier.modules())
    restored = build_segmenter("none")
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        assert torch.equal(model(image)["out"], restored(image)["out"])
