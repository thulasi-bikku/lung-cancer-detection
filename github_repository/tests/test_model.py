import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.model import HybridCNNTransformer, count_parameters  # noqa: E402


@pytest.mark.parametrize("kwargs", [
    {}, {"n_layers": 1}, {"n_layers": 0}, {"use_pos_embed": False},
    {"use_projection": False},
])
def test_forward_shape(kwargs):
    m = HybridCNNTransformer(pretrained=False, **kwargs).eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_token_grid():
    m = HybridCNNTransformer(pretrained=False).eval()
    assert m.backbone(torch.randn(1, 3, 224, 224)).shape[-2:] == (7, 7)
    assert m.pos_embed.shape == (1, 49, 256)
    assert count_parameters(m) > 0
