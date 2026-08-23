import pytest
import torch
from torch import nn

from differential_uncertainty.persistence import Layer2Capture, batched_persistence
# Transitional oracle: the later legacy-prune task must freeze this behavior in
# a self-contained fixture before src.misc is removed.
from src.misc.tue_utils import get_persistence_diagrams_batched


def test_batched_persistence_matches_the_legacy_implementation_exactly():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(5, 8, generator=generator)
    inputs = torch.randn(13, 8, generator=generator)

    expected = get_persistence_diagrams_batched(weight, inputs, chunk_size=4)
    actual = batched_persistence(weight, inputs, chunk_size=4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (13, 12)
    assert torch.all(actual[:, :-1] >= actual[:, 1:])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_batched_persistence_uses_input_device_and_float32_computation(device):
    weight = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], dtype=torch.float64)
    inputs = torch.tensor(
        [[2.0, -1.0]],
        dtype=torch.float64,
        device=device,
    )

    diagrams = batched_persistence(weight, inputs)

    assert diagrams.device == inputs.device
    assert diagrams.dtype == torch.float32
    torch.testing.assert_close(
        diagrams,
        torch.tensor(
            [[6.0, 4.0, 2.0]],
            dtype=torch.float32,
            device=device,
        ),
        rtol=0,
        atol=0,
    )


def test_batched_persistence_preserves_empty_query_shape_dtype_and_device():
    weight = torch.ones(5, 8, dtype=torch.float64)
    inputs = torch.empty(0, 8, dtype=torch.float64)

    diagrams = batched_persistence(weight, inputs, chunk_size=4)

    assert diagrams.shape == (0, 12)
    assert diagrams.device == inputs.device
    assert diagrams.dtype == torch.float32


@pytest.mark.parametrize(
    ("weight", "inputs", "message"),
    [
        (torch.ones(2), torch.ones(1, 2), "two-dimensional"),
        (torch.ones(2, 2), torch.ones(2), "two-dimensional"),
        (torch.ones(2, 3), torch.ones(1, 2), "dimensions do not match"),
        (torch.ones(2, 3), torch.empty(0, 2), "dimensions do not match"),
    ],
)
def test_batched_persistence_rejects_invalid_tensor_shapes(weight, inputs, message):
    with pytest.raises(ValueError, match=message):
        batched_persistence(weight, inputs)


@pytest.mark.parametrize(
    ("weight", "inputs", "message"),
    [
        (torch.empty(2, 0), torch.empty(1, 0), "at least one input vertex"),
        (torch.empty(0, 2), torch.empty(1, 2), "at least one output vertex"),
        (torch.empty(0, 0), torch.empty(0, 0), "at least one input vertex"),
        (torch.empty(0, 1), torch.empty(1, 1), "at least one output vertex"),
    ],
)
def test_batched_persistence_rejects_empty_graph_partitions(
    weight,
    inputs,
    message,
):
    with pytest.raises(ValueError, match=message):
        batched_persistence(weight, inputs)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_batched_persistence_requires_a_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        batched_persistence(torch.ones(2, 3), torch.ones(1, 3), chunk_size)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(3)]
        )
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(4, 3) for _ in range(3)]
        )

    def forward(self, values):
        for layer in self.decoder.layers:
            values = layer(values)
        return values


def test_layer_2_capture_records_detached_features_and_score_weights_once():
    transformer = _Transformer()
    values = torch.ones(2, 5, 4, requires_grad=True)

    with Layer2Capture(transformer, layer=2) as capture:
        output = transformer(values)
        features, weight = capture.take()
        assert capture.handles
        with pytest.raises(RuntimeError, match="did not execute decoder layer 2"):
            capture.take()

    torch.testing.assert_close(features, output)
    torch.testing.assert_close(weight, transformer.dec_score_head[2].weight)
    assert not features.requires_grad
    assert not weight.requires_grad
    assert not capture.handles
    assert not transformer.decoder.layers[2]._forward_hooks


def test_layer_2_capture_take_requires_a_completed_capture():
    transformer = _Transformer()

    with Layer2Capture(transformer) as capture:
        with pytest.raises(RuntimeError, match="did not execute decoder layer 2"):
            capture.take()


@pytest.mark.parametrize("layer", [-1, 3])
def test_layer_2_capture_rejects_layers_outside_the_decoder(layer):
    with pytest.raises(ValueError, match=f"decoder layer {layer} does not exist"):
        Layer2Capture(_Transformer(), layer=layer)


def test_layer_2_capture_requires_a_linear_score_head():
    transformer = _Transformer()
    transformer.dec_score_head[2] = nn.ReLU()

    with pytest.raises(TypeError, match="score head must be nn.Linear"):
        Layer2Capture(transformer)


def test_layer_2_capture_close_is_idempotent_and_removes_the_hook():
    transformer = _Transformer()
    capture = Layer2Capture(transformer)
    transformer(torch.ones(1, 4))
    assert capture.captured is not None
    assert transformer.decoder.layers[2]._forward_hooks

    capture.close()
    capture.close()

    assert capture.captured is None
    assert not capture.handles
    assert not transformer.decoder.layers[2]._forward_hooks


def test_layer_2_capture_context_exit_discards_an_unconsumed_activation():
    transformer = _Transformer()

    with Layer2Capture(transformer) as capture:
        transformer(torch.ones(1, 4))
        assert capture.captured is not None

    assert capture.captured is None
    assert not capture.handles
    assert not transformer.decoder.layers[2]._forward_hooks
