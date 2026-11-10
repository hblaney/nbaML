"""Smoke tests — package import only; no live APIs."""


def test_import_package() -> None:
    import nba_betting

    assert nba_betting.__version__


def test_import_subpackages() -> None:
    from nba_betting import backtest, data, features, models, pricing, slate

    assert data.__doc__
    assert features.__doc__
    assert models.__doc__
    assert pricing.__doc__
    assert backtest.__doc__
    assert slate.__doc__
