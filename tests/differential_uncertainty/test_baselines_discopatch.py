import numpy as np
import pytest
import torch
from PIL import Image

from differential_uncertainty.baselines import discopatch as dp

pytestmark = pytest.mark.skipif(not dp.DEFAULT_ROOT.exists(), reason="DisCoPatch repository not available")


def test_crop_positions_are_seeded_by_image_and_stay_inside_the_256_square():
    first, again, other = dp.crop_positions("a.jpg", 64), dp.crop_positions("a.jpg", 64), dp.crop_positions("b.jpg", 64)
    assert first.shape == (64, 2) and np.array_equal(first, again) and not np.array_equal(first, other)
    assert first.min() >= 0 and first.max() < 192


def test_coco_patch_dataset_returns_normalised_64px_patches_from_grayscale_too(tmp_path):
    path = tmp_path / "x.jpg"
    Image.new("L", (320, 200), 255).save(path)
    patches, label = dp.CocoPatchDataset([path], patches=5)[0]
    assert patches.shape == (5, 3, 64, 64) and label == 0
    assert patches.max().item() == pytest.approx(1.0, abs=1e-2)


def test_scorer_normalises_each_image_separately(tmp_path):
    module = dp.import_discopatch(tmp_path, dp.DEFAULT_ROOT)
    torch.save(module.Discriminator(64, 3, [4, 8], 1e-4, 2).state_dict(), tmp_path / "disc.pt")
    scorer = dp.DisCoPatchScorer(tmp_path / "disc.pt", tmp_path, "cpu", patches=8, hidden_dims=[4, 8])
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(2))
    together, alone = scorer.score([a, b], "img.jpg"), scorer.score([a], "img.jpg")
    assert together.shape == (2,) and together[0] == pytest.approx(alone[0], abs=1e-6)
    assert np.all((together >= 0) & (together <= 1))
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in scorer.discriminator.modules())


def test_scores_keep_their_order_when_every_patch_output_is_tiny(tmp_path):
    module = dp.import_discopatch(tmp_path, dp.DEFAULT_ROOT)
    torch.save(module.Discriminator(64, 3, [4, 8], 1e-4, 2).state_dict(), tmp_path / "disc.pt")
    scorer = dp.DisCoPatchScorer(tmp_path / "disc.pt", tmp_path, "cpu", patches=8, hidden_dims=[4, 8])

    class Tiny(torch.nn.Module):  # float32 patch outputs of 1e-9 (first image) and 1e-10 (second)
        def forward(self, x):
            return torch.tensor([1e-9] * 8 + [1e-10] * 8, dtype=torch.float32).view(-1, 1)

    scorer.discriminator = Tiny()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    first, second = scorer.score([image, image], "img.jpg")
    assert first < second < 1.0


def test_training_wrapper_runs_one_tiny_epoch_and_writes_the_discriminator(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (300, 260), (index * 100, 50, 200)).save(path)
        paths.append(path)
    checkpoint = dp.train_discopatch(
        paths, tmp_path / "models", epochs=1, num_workers=0, seed=0,
        overrides={"hidden_dims": [4, 8], "latent_dim": 8, "batch_size": 2, "patches": 2},
    )
    assert checkpoint.exists() and checkpoint.name == "Discriminator_coco.pt"


def _tiny_model(tmp_path):
    from argparse import Namespace
    module = dp.import_discopatch(tmp_path, dp.DEFAULT_ROOT)
    args = Namespace(**{**dp.TRAIN_ARGS, "hidden_dims": [4, 8], "latent_dim": 8, "batch_size": 2, "patches": 2,
                        "n_epochs": 1})
    return module.DisCoPatch(input_shape=64, input_channels=3, args=args)


def _generator_gradients(model, x):
    model.zero_grad()
    torch.manual_seed(1)
    recon, mu, logvar = model.vae(x)
    gen = model.vae.decode(torch.randn(x.size(0), model.vae.latent_dim))
    loss = (model.discriminator(recon).mean() + model.discriminator(gen).mean()
            + model.vae.loss_function(recon, x, mu, logvar))
    loss.backward()
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_recomputing_activations_leaves_the_gradients_unchanged(tmp_path):
    import copy
    torch.manual_seed(0)
    model = _tiny_model(tmp_path)
    recomputed = copy.deepcopy(model)
    dp.recompute_activations(recomputed)
    x = torch.randn(6, 3, 64, 64)
    expected, actual = _generator_gradients(model, x), _generator_gradients(recomputed, x)
    assert expected.keys() == actual.keys() and len(expected) > 10
    for name in expected:
        assert torch.allclose(expected[name], actual[name], rtol=1e-5, atol=1e-7), name
    assert recomputed.state_dict().keys() == model.state_dict().keys()


def test_mixed_precision_keeps_the_discriminator_head_in_fp32_and_restores_the_loss(tmp_path):
    model = _tiny_model(tmp_path)
    original = torch.nn.BCELoss
    x = torch.randn(4, 3, 64, 64)
    with dp.mixed_precision(model):
        output = model.discriminator(x)
        loss = torch.nn.BCELoss()(output, torch.ones_like(output))
    assert output.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.nn.BCELoss is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the autocast BCELoss ban is CUDA-only")
def test_training_wrapper_runs_under_cuda_mixed_precision(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (300, 260), (index * 100, 50, 200)).save(path)
        paths.append(path)
    checkpoint = dp.train_discopatch(
        paths, tmp_path / "models", epochs=1, num_workers=0, seed=0,
        overrides={"hidden_dims": [4, 8], "latent_dim": 8, "batch_size": 2, "patches": 2},
    )
    state = torch.load(checkpoint, map_location="cpu")
    assert all(value.dtype == torch.float32 for value in state.values() if value.is_floating_point())
