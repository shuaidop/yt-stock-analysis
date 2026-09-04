"""Portfolio and live-quote context for the trading brief.

Sources, in order of preference:
- IBKR TWS / IB Gateway API via ``ib_async`` (optional extra ``ibkr``; needs TWS running
  with API enabled). Enable with ``IBKR_ENABLED=true``.
- A JSON snapshot at ``PORTFOLIO_FILE`` (``data/portfolio.json``), e.g. exported from the
  IBKR connector in Claude Code or written by ``ytstock portfolio-import``.

Quotes come from IBKR when connected, else from Yahoo Finance via ``yfinance`` (keyless,
possibly delayed). Everything here degrades gracefully: any failure yields ``None`` /
empty quotes and the brief proceeds without that context.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ytstock.config import Settings
from ytstock.log import get_logger

log = get_logger(__name__)

DEFAULT_QUOTE_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "^VIX", "^TNX"]


class Holding(BaseModel):
    symbol: str
    asset_class: Literal["STK", "OPT", "FUT", "CASH", "OTHER"] = "STK"
    description: str = ""
    quantity: float
    avg_price: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def root(self) -> str:
        """Underlying symbol for options like 'CRWV Oct16'26 120 CALL'."""
        return self.symbol.split()[0].upper()


class OpenOrder(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: str = "LIMIT"
    quantity: float
    limit_price: float | None = None
    status: str = ""
    tif: str = ""
    description: str = ""


class Portfolio(BaseModel):
    as_of: datetime
    source: str
    currency: str = "USD"
    net_liquidation: float
    cash: float
    buying_power: float = 0.0
    holdings: list[Holding]
    orders: list[OpenOrder] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return list(dict.fromkeys(h.root for h in self.holdings))

    def weight(self, holding: Holding) -> float:
        return holding.market_value / self.net_liquidation if self.net_liquidation else 0.0


class Quote(BaseModel):
    symbol: str
    last: float
    prev_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    as_of: datetime
    source: str

    @property
    def change_pct(self) -> float | None:
        if self.prev_close:
            return (self.last / self.prev_close - 1.0) * 100.0
        return None


# --------------------------------------------------------------------------- #
# Portfolio providers
# --------------------------------------------------------------------------- #
def load_portfolio_file(path: Path) -> Portfolio | None:
    if not path.exists():
        return None
    try:
        return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("portfolio.file_invalid", path=str(path), error=str(exc)[:200])
        return None


def save_portfolio_file(path: Path, portfolio: Portfolio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")


AssetClass = Literal["STK", "OPT", "FUT", "CASH", "OTHER"]


def _asset_class(value: Any) -> AssetClass:
    v = str(value or "").upper()
    if v == "STK":
        return "STK"
    if v == "OPT":
        return "OPT"
    if v == "FUT":
        return "FUT"
    if v == "CASH":
        return "CASH"
    return "OTHER"


def portfolio_from_ibkr_connector(
    summary: dict[str, Any], positions: dict[str, Any], orders: dict[str, Any] | None = None
) -> Portfolio:
    """Build a Portfolio from the raw JSON returned by the IBKR connector tools
    (get_account_summary / get_account_positions / get_account_orders)."""
    holdings = []
    for p in positions.get("positions", []):
        holdings.append(
            Holding(
                symbol=p.get("contract_description", ""),
                asset_class=_asset_class(p.get("asset_class", "STK")),
                quantity=float(p.get("position", 0)),
                avg_price=float(p.get("average_price", 0) or 0),
                market_price=float(p.get("market_price", 0) or 0),
                market_value=float(p.get("market_value", 0) or 0),
                unrealized_pnl=float(p.get("unrealized_pnl", 0) or 0),
            )
        )
    open_orders = []
    for o in (orders or {}).get("orders", []):
        status = str(o.get("order_status", ""))
        if status.upper() in {"FILLED", "CANCELLED", "CANCELED", "REPLACED", "INACTIVE"}:
            continue
        desc = str(o.get("primary_description", ""))
        symbol = desc.split()[-1] if desc else ""
        open_orders.append(
            OpenOrder(
                symbol=symbol,
                side="SELL" if str(o.get("side", "")).upper() == "SELL" else "BUY",
                order_type=str(o.get("order_type", "LIMIT")),
                quantity=float(o.get("remaining_shares_qty") or o.get("total_shares_qty") or 0),
                limit_price=float(o["limit_price"]) if o.get("limit_price") else None,
                status=status,
                tif="GTC" if "GTC" in str(o.get("secondary_description", "")) else "",
                description=f"{desc} — {o.get('secondary_description', '')}".strip(" —"),
            )
        )
    return Portfolio(
        as_of=datetime.now(UTC),
        source="ibkr-connector",
        currency=str(summary.get("currency", "USD")),
        net_liquidation=float(summary.get("net_liquidation", 0)),
        cash=float(summary.get("total_cash_value", 0)),
        buying_power=float(summary.get("buying_power", 0)),
        holdings=holdings,
        orders=open_orders,
    )


def portfolio_from_ib_async(
    host: str, port: int, client_id: int, timeout: float = 10.0
) -> Portfolio:
    """Live pull from TWS / IB Gateway. Requires the `ibkr` extra and a running TWS."""
    from ib_async import IB  # lazy: optional extra

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=True)
    try:
        summary = {row.tag: row.value for row in ib.accountSummary()}
        holdings = []
        for item in ib.portfolio():
            c = item.contract
            holdings.append(
                Holding(
                    symbol=c.localSymbol or c.symbol,
                    asset_class=_asset_class(c.secType),
                    description=c.symbol,
                    quantity=float(item.position),
                    avg_price=float(item.averageCost)
                    / (float(c.multiplier) if c.secType == "OPT" and c.multiplier else 1.0),
                    market_price=float(item.marketPrice),
                    market_value=float(item.marketValue),
                    unrealized_pnl=float(item.unrealizedPNL),
                )
            )
        orders = []
        for tr in ib.openTrades():
            o, c = tr.order, tr.contract
            orders.append(
                OpenOrder(
                    symbol=c.localSymbol or c.symbol,
                    side="SELL" if o.action == "SELL" else "BUY",
                    order_type=o.orderType,
                    quantity=float(o.totalQuantity),
                    limit_price=float(o.lmtPrice) if o.orderType in ("LMT", "LIMIT") else None,
                    status=tr.orderStatus.status,
                    tif=o.tif,
                )
            )
        return Portfolio(
            as_of=datetime.now(UTC),
            source="ibkr-tws",
            currency=str(summary.get("Currency", "USD")),
            net_liquidation=float(summary.get("NetLiquidation", 0) or 0),
            cash=float(summary.get("TotalCashValue", 0) or 0),
            buying_power=float(summary.get("BuyingPower", 0) or 0),
            holdings=holdings,
            orders=orders,
        )
    finally:
        ib.disconnect()


def load_portfolio(settings: Settings) -> Portfolio | None:
    if settings.ibkr_enabled:
        try:
            p = portfolio_from_ib_async(
                settings.ibkr_host, settings.ibkr_port, settings.ibkr_client_id
            )
            save_portfolio_file(settings.portfolio_file, p)  # keep the snapshot fresh
            return p
        except Exception as exc:
            log.warning("portfolio.ibkr_failed", error=str(exc)[:200])
    return load_portfolio_file(settings.portfolio_file)


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #
def _yf_quote(symbol: str) -> Quote | None:
    import yfinance as yf  # lazy

    try:
        fi = yf.Ticker(symbol).fast_info
        last = fi.last_price
        if last is None:
            return None
        return Quote(
            symbol=symbol,
            last=float(last),
            prev_close=float(fi.previous_close) if fi.previous_close else None,
            day_high=float(fi.day_high) if getattr(fi, "day_high", None) else None,
            day_low=float(fi.day_low) if getattr(fi, "day_low", None) else None,
            as_of=datetime.now(UTC),
            source="yfinance",
        )
    except Exception as exc:
        log.debug("quotes.yf_failed", symbol=symbol, error=str(exc)[:120])
        return None


def fetch_quotes(symbols: list[str], *, workers: int = 8) -> dict[str, Quote]:
    syms = [s for s in dict.fromkeys(x.strip().upper() for x in symbols if x) if s]
    if not syms:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_yf_quote, syms))
    quotes = {q.symbol: q for q in results if q is not None}
    log.info("quotes.fetched", requested=len(syms), ok=len(quotes))
    return quotes


# --------------------------------------------------------------------------- #
# Prompt context
# --------------------------------------------------------------------------- #
def portfolio_context(portfolio: Portfolio | None) -> str:
    if portfolio is None:
        return ""
    L = [
        "<portfolio>",
        f"as_of: {portfolio.as_of.isoformat(timespec='minutes')}  source: {portfolio.source}",
        f"net_liquidation: {portfolio.net_liquidation:,.0f} {portfolio.currency}  "
        f"cash: {portfolio.cash:,.0f}  buying_power: {portfolio.buying_power:,.0f}",
        "holdings (symbol | class | qty | avg | last | value | weight | unrealized):",
    ]
    for h in sorted(portfolio.holdings, key=lambda x: -abs(x.market_value)):
        L.append(
            f"  {h.symbol} | {h.asset_class} | {h.quantity:g} | {h.avg_price:.2f} | "
            f"{h.market_price:.2f} | {h.market_value:,.0f} | {portfolio.weight(h) * 100:.1f}% | "
            f"{h.unrealized_pnl:+,.0f}"
        )
    if portfolio.orders:
        L.append("open_orders:")
        for o in portfolio.orders:
            price = f" @ {o.limit_price:.2f}" if o.limit_price else ""
            L.append(
                f"  {o.side} {o.quantity:g} {o.symbol} {o.order_type}{price} {o.tif} [{o.status}]"
            )
    L.append("</portfolio>")
    return "\n".join(L)


def quotes_context(quotes: dict[str, Quote]) -> str:
    if not quotes:
        return ""
    L = ["<quotes>  (symbol | last | prev_close | change% | day_low-day_high | as_of)"]
    for q in quotes.values():
        chg = f"{q.change_pct:+.2f}%" if q.change_pct is not None else "n/a"
        rng = f"{q.day_low:.2f}-{q.day_high:.2f}" if q.day_low and q.day_high else "n/a"
        prev = f"{q.prev_close:.2f}" if q.prev_close else "n/a"
        L.append(
            f"  {q.symbol} | {q.last:.2f} | {prev} | {chg} | {rng} | "
            f"{q.as_of.strftime('%Y-%m-%d %H:%MZ')} ({q.source})"
        )
    L.append("</quotes>")
    return "\n".join(L)


def build_context(
    settings: Settings, mentioned_symbols: list[str]
) -> tuple[Portfolio | None, dict[str, Quote]]:
    portfolio = load_portfolio(settings) if settings.brief_include_portfolio else None
    if not settings.quotes_enabled:
        return portfolio, {}
    symbols = list(DEFAULT_QUOTE_SYMBOLS)
    if portfolio:
        symbols += portfolio.symbols
    symbols += mentioned_symbols
    symbols = [s for s in dict.fromkeys(symbols) if s and not s.startswith("$")]
    return portfolio, fetch_quotes(symbols[: settings.max_quote_symbols])


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
