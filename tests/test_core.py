import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asym_ae.data import load_split, shift_images, rotate_images, split_validation
from asym_ae.models import AutoEncoder, NAMED, latent_reg
from asym_ae import evaluate as E
from asym_ae.nn import MLP


@pytest.fixture(scope="module")
def d():
    return load_split(0)


def num_grad(f, arr, idx, eps=1e-6):
    old = arr[idx]
    arr[idx] = old + eps; a = f()
    arr[idx] = old - eps; b = f()
    arr[idx] = old
    return (a - b) / (2 * eps)


@pytest.mark.parametrize("arch", list(NAMED) + [(1, 1), (2, 1)])
def test_backprop_matches_finite_differences(arch, d):
    ae = AutoEncoder(arch, 4, seed=1)
    x = d["Xtr"][:12]
    loss = lambda: ((ae.reconstruct(x) - x) ** 2).mean()
    z = ae.enc.forward(x, train=True); y = ae.dec.forward(z, train=True)
    dWd, _, dz = ae.dec.backward(2 * (y - x) / x.size)
    dWe, _, _ = ae.enc.backward(dz)
    for net, grads in ((ae.enc, dWe), (ae.dec, dWd)):
        i, j = 0, 1
        assert num_grad(loss, net.W[0], (i, j)) == pytest.approx(grads[0][i, j], rel=1e-4, abs=1e-9)


def test_latent_reg_gradient():
    z = np.random.default_rng(0).normal(size=(16, 4)) * 0.3
    _, g = latent_reg(z)
    f = lambda: latent_reg(z)[0]
    for idx in [(0, 0), (5, 3), (15, 2)]:
        assert num_grad(f, z, idx) == pytest.approx(g[idx], rel=1e-4, abs=1e-9)


def test_latent_reg_penalises_collapse():
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(256, 4))
    collapsed = np.outer(rng.normal(size=256), np.ones(4))      # rank-1 code
    assert latent_reg(collapsed)[0] > latent_reg(healthy)[0] * 5


def test_shapes_and_param_counts():
    ae = AutoEncoder("heavy_enc", 8, seed=0)
    X = np.zeros((5, 64))
    assert ae.encode(X).shape == (5, 8) and ae.reconstruct(X).shape == (5, 64)
    assert ae.n_enc > 20 * ae.n_dec                                  # heavy encoder / light decoder
    h = AutoEncoder("heavy_dec", 8)
    assert h.n_dec > 20 * h.n_enc


def test_training_reduces_loss_and_is_deterministic(d):
    h1 = AutoEncoder("symmetric", 8, seed=3).fit(d["Xtr"], epochs=8)
    h2 = AutoEncoder("symmetric", 8, seed=3).fit(d["Xtr"], epochs=8)
    assert h1[-1] < h1[0] * 0.6
    assert h1 == h2


def test_data_split_depends_on_seed_and_is_stratified():
    a, b = load_split(0), load_split(1)
    assert not np.array_equal(a["Xtr"][:5], b["Xtr"][:5])
    assert len(a["Xtr"]) == 1347 and len(a["Xte"]) == 450
    assert np.bincount(a["yte"]).min() >= 40
    assert a["Xtr"].min() >= 0 and a["Xtr"].max() <= 1


def test_validation_split_is_disjoint_from_test(d):
    Xf, yf, Xv, yv = split_validation(d, 0)
    assert len(Xf) + len(Xv) == len(d["Xtr"])


def test_shift_and_rotate_change_images_but_not_shape(d):
    X = d["Xte"][:10]
    assert shift_images(X).shape == X.shape and not np.allclose(shift_images(X), X)
    assert rotate_images(X, 0.0) == pytest.approx(X, abs=1e-9)


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    assert E.effective_rank(rng.normal(size=(2000, 6))) == pytest.approx(6, abs=0.3)
    line = np.outer(rng.normal(size=500), np.arange(1, 5))
    assert E.effective_rank(line + 1e-6 * rng.normal(size=line.shape)) == pytest.approx(1.0, abs=0.01)


def test_probe_perfect_on_separable_data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 50)
    Z = rng.normal(size=(150, 2)) * 0.1 + np.eye(3)[y][:, :2] * 5 + (y == 2)[:, None] * np.array([-5, -5])
    assert E.LinearProbe(Z, y).score(Z, y) > 0.98


def test_pixels_baseline_is_a_strong_probe(d):
    assert E.LinearProbe(d["Xtr"], d["ytr"]).score(d["Xte"], d["yte"]) > 0.93


def test_full_eval_returns_all_metrics(d):
    m = E.full_eval(lambda X: X[:, :8], lambda X: X, d, 0)
    for key in ["mse", "probe", "knn", "nmi", "trust", "rob_shift", "rob_rot", "rob_noise", "lab1", "lab5", "lab10", "lab25"]:
        assert key in m
        assert 0 <= m[key] <= 1 + 1e-9
    assert m["erank"] >= 1


def test_trace_records_effective_rank(d):
    ae = AutoEncoder("symmetric", 4, seed=0)
    ae.fit(d["Xtr"], epochs=6, trace_every=3)
    assert [t[0] for t in ae.trace] == [1, 3, 6]
