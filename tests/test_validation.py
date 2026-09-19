import numpy as np

from sigdiscover.validation.metrics import (
    cosine_sim,
    exposure_correlation,
    frobenius_error,
    reconstruction_r2,
    signature_precision_recall,
)


def test_cosine_sim():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    assert np.isclose(cosine_sim(a, b), 1.0)
    c = np.array([0, 1, 0])
    assert np.isclose(cosine_sim(a, c), 0.0)

def test_frobenius_error():
    M = np.array([[1, 2], [3, 4]])
    M_hat = np.array([[1, 2], [3, 4]])
    assert np.isclose(frobenius_error(M, M_hat), 0.0)

def test_reconstruction_r2():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    M_hat = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(reconstruction_r2(M, M_hat), 1.0)

def test_exposure_correlation():
    A_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    A_pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.isclose(exposure_correlation(A_true, A_pred), 1.0)

def test_signature_precision_recall():
    S_true = np.array([[1, 0, 0], [0, 1, 0]])
    S_pred = np.array([[1, 0, 0], [0, 1, 0]])
    p, r, f1 = signature_precision_recall(S_true, S_pred)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0
