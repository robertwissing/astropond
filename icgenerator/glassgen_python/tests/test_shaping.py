"""glassgen.shaping: the curvature step and its err_P calibration."""
import numpy as np
import pytest

from glassgen.params import density_from_param
from glassgen.shaping import CAL_A, CAL_C, CAL_P, ERRP_MARGIN, beta2_for_errp, curvature_step, curvature_table, \
    curvature_target

M, NS = 3.1e-5, 64


def slab(contrast=10.0, **kw):
    p = dict(dICdensprofile='3', dICdensdir='2', dICdensR='0.25', dICdensRsmooth='0.01', dICdensinner=str(contrast),
             dICdensouter='1.0', dxPeriod='0.125', dyPeriod='2.0', dzPeriod='0.153')
    p.update({k: str(v) for k, v in kw.items()})
    return p


@pytest.mark.parametrize('geometry', ['slab', 'cyl', 'sph'])
def test_step_keeps_mass_plateaus_and_curvature_limit(geometry):
    c = (np.arange(40000) + 0.5) * 1.0 / 40000
    rho = np.where(c < 0.25, 10.0, 1.0)
    beta2 = 0.05
    T, _, merr = curvature_step(c, rho, M, NS, beta2, geometry=geometry)
    assert abs(merr) < 1e-6
    assert T[0] == pytest.approx(10.0) and T[-1] == pytest.approx(1.0)
    assert np.all(np.diff(T) <= 1e-12)
    h = 0.5 * (3 * NS * M / (4 * np.pi * T)) ** (1 / 3)
    lt, dc = np.log(T), c[1] - c[0]
    g2 = h[1:-1] ** 2 * np.abs(lt[2:] - 2 * lt[1:-1] + lt[:-2]) / dc ** 2
    assert g2.max() == pytest.approx(beta2, rel=0.05)


@pytest.mark.parametrize('contrast,a', zip(CAL_C, CAL_A))
def test_beta2_inverts_the_calibration(contrast, a):
    beta2 = beta2_for_errp(0.02, contrast)
    assert a * beta2 ** CAL_P == pytest.approx(ERRP_MARGIN * 0.02)


def test_smaller_err_p_means_smaller_b2_and_higher_contrast_too():
    assert beta2_for_errp(0.01, 10.0) < beta2_for_errp(0.05, 10.0)
    assert beta2_for_errp(0.02, 100.0) < beta2_for_errp(0.02, 2.0)


@pytest.mark.parametrize('errp,contrast', [(0.001, 10.0), (0.5, 10.0), (0.02, 360.0)])
def test_outside_the_calibration_is_refused(errp, contrast):
    with pytest.raises(ValueError):
        beta2_for_errp(errp, contrast)


@pytest.mark.parametrize('kw', [dict(dICdensprofile=1), dict(dICdensdir=6), dict(dICdensdir=0),
                                dict(dICdensinner=1.0, dICdensouter=2.0)])
def test_a_target_that_is_not_a_dense_inside_1d_contact_is_refused(kw):
    with pytest.raises(ValueError):
        curvature_table(slab(**kw), M, NS, 0.05)


def test_a_feature_too_narrow_for_the_step_is_refused():
    with pytest.raises(ValueError, match='too narrow'):
        curvature_table(slab(dICdensR=0.02), M, NS, 0.005)


def test_target_reads_like_the_table_and_conserves_mass():
    table, beta2, merr = curvature_target(slab(), M, NS, 0.02)
    assert beta2 == beta2_for_errp(0.02, 10.0)
    assert abs(merr) < 1e-6
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, -0.9, 0.0]])
    rho = table.rho0(pos)
    assert rho[0] == pytest.approx(10.0)
    assert rho[1] == pytest.approx(1.0) and rho[2] == pytest.approx(1.0)


def test_shape_off_leaves_the_raw_param_target():
    raw = density_from_param(slab())
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.9, 0.0]])
    assert raw.rho0(pos) == pytest.approx([10.0, 5.5, 1.0])
