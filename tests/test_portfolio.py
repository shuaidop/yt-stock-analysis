from __future__ import annotations

from datetime import UTC, datetime

from ytstock.analysis import GroundedAnalyzer, compute_ticker_stats
from ytstock.portfolio import (
    Quote,
    build_context,
    fetch_quotes,
    load_portfolio_file,
    portfolio_context,
    portfolio_from_ibkr_connector,
    quotes_context,
    save_portfolio_file,
)
from ytstock.schemas import VideoAnalysis

from .conftest import ANALYSIS_DICT

SUMMARY = {
    "currency": "USD",
    "net_liquidation": 100000.0,
    "total_cash_value": 10000.0,
    "buying_power": 10000.0,
}
POSITIONS = {
    "positions": [
        {
            "contract_description": "CRWV",
            "position": 745,
            "market_price": 84.78,
            "market_value": 63161.1,
            "average_price": 94.26,
            "unrealized_pnl": -7063.48,
            "asset_class": "STK",
        },
        {
            "contract_description": "CRWV Oct16'26 120 CALL",
            "position": -5,
            "market_price": 1.07,
            "market_value": -537.19,
            "average_price": 0.87,
            "unrealized_pnl": -100.7,
            "asset_class": "OPT",
        },
    ]
}
ORDERS = {
    "orders": [
        {
            "order_status": "REPLACED",
            "order_type": "LIMIT",
            "side": "SELL",
            "limit_price": "129",
            "remaining_shares_qty": "50",
            "primary_description": "Sell 50 CRWV",
            "secondary_description": "Limit 129.00, GTC",
        },
        {
            "order_status": "NEW",
            "order_type": "LIMIT",
            "side": "SELL",
            "limit_price": "462.3",
            "remaining_shares_qty": "50",
            "primary_description": "Sell 50 AVGO",
            "secondary_description": "Limit 462.30, GTC",
        },
    ]
}


def test_connector_conversion_and_roundtrip(tmp_path):
    p = portfolio_from_ibkr_connector(SUMMARY, POSITIONS, ORDERS)
    assert p.symbols == ["CRWV"]
    assert [o.symbol for o in p.orders] == ["AVGO"]  # REPLACED orders dropped
    assert p.orders[0].tif == "GTC" and p.orders[0].limit_price == 462.3
    assert round(p.weight(p.holdings[0]) * 100, 1) == 63.2
    path = tmp_path / "pf.json"
    save_portfolio_file(path, p)
    back = load_portfolio_file(path)
    assert back is not None and back.net_liquidation == 100000.0 and len(back.holdings) == 2
    assert load_portfolio_file(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert load_portfolio_file(tmp_path / "bad.json") is None


def test_context_rendering():
    p = portfolio_from_ibkr_connector(SUMMARY, POSITIONS, ORDERS)
    ctx = portfolio_context(p)
    assert (
        "<portfolio>" in ctx
        and "CRWV | STK | 745" in ctx
        and "SELL 50 AVGO LIMIT @ 462.30 GTC" in ctx
    )
    q = {
        "SPY": Quote(
            symbol="SPY",
            last=773.2,
            prev_close=764.5,
            day_low=767.4,
            day_high=774.0,
            as_of=datetime(2026, 9, 3, 20, 0, tzinfo=UTC),
            source="yfinance",
        )
    }
    qc = quotes_context(q)
    assert "SPY | 773.20 | 764.50 | +1.14%" in qc
    assert portfolio_context(None) == "" and quotes_context({}) == ""


def test_fetch_quotes_uses_provider(monkeypatch):
    seen = []

    def fake(symbol):
        seen.append(symbol)
        if symbol == "BAD":
            return None
        return Quote(symbol=symbol, last=1.0, as_of=datetime.now(UTC), source="test")

    monkeypatch.setattr("ytstock.portfolio._yf_quote", fake)
    out = fetch_quotes(["spy", "SPY", "bad", ""])
    assert sorted(seen) == ["BAD", "SPY"] and list(out) == ["SPY"]


def test_build_context_merges_symbols(settings, tmp_path, monkeypatch):
    p = portfolio_from_ibkr_connector(SUMMARY, POSITIONS, ORDERS)
    save_portfolio_file(tmp_path / "pf.json", p)
    settings = settings.model_copy(update={"portfolio_file": tmp_path / "pf.json"})
    requested = {}
    monkeypatch.setattr(
        "ytstock.portfolio.fetch_quotes",
        lambda syms, **kw: requested.setdefault("syms", syms) and {},
    )
    portfolio, quotes = build_context(settings, ["NVDA", "SPY"])
    assert portfolio is not None and quotes == {}
    assert (
        requested["syms"][:2] == ["SPY", "QQQ"]
        and "CRWV" in requested["syms"]
        and "NVDA" in requested["syms"]
    )
    assert requested["syms"].count("SPY") == 1


def test_brief_prompt_includes_context(video):
    a = VideoAnalysis.model_validate(ANALYSIS_DICT)
    stats = compute_ticker_stats({video.video_id: a})
    prompt = GroundedAnalyzer.build_brief_prompt(
        "2026-09-03",
        [video],
        {video.video_id: a},
        {},
        stats,
        extra_context="<portfolio>x</portfolio>",
    )
    assert prompt.index("<portfolio>") < prompt.index("<ticker_stats>")
