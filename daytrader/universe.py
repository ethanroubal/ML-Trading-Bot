"""Tradable universes.

Day trading needs names with tight spreads and deep volume, and it helps to
have high-beta names whose intraday moves are large relative to trading
costs. Symbols that fail to return data are skipped automatically, so a
stale ticker here is harmless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

# ~70 of the most liquid US names: the default when using Yahoo (one request per symbol).
CORE = """
SPY QQQ IWM
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE INTC MU QCOM CSCO
PLTR COIN HOOD SOFI MSTR UBER SHOP SNOW CRWD PANW ARM SMCI MRVL DELL
JPM BAC WFC C GS MS V MA PYPL SCHW
XOM CVX OXY SLB
UNH LLY JNJ PFE MRK ABBV
WMT COST HD TGT NKE SBUX MCD DIS
BA CAT GE F GM DAL UAL CCL
BABA TSM NIO RIVN
""".split()

# ~200 names: the default with Alpaca (multi-symbol requests make this cheap).
LIQUID = CORE + """
DIA SMH XLF XLE XLK
TXN AMAT LRCX KLAC ADI NXPI ON MCHP ASML ANET NOW INTU IBM HPQ
NET DDOG ZS OKTA MDB TEAM WDAY TTD APP SPOT RBLX ROKU SNAP PINS ZM DOCU TWLO
ABNB DASH LYFT BKNG MAR RCL AAL
AFRM UPST MARA RIOT CLSK CVNA CHWY DKNG RDDT HIMS
AXP COF USB PNC TFC BX KKR BLK SPGI
COP EOG MPC VLO PSX HAL DVN FCX NEM AA CLF NUE
AMGN GILD BMY TMO ABT DHR ISRG MDT SYK VRTX REGN MRNA CVS CI ELV HUM
PG KO PEP PM MO MDLZ CL LOW TJX LULU CMG YUM
HON RTX LMT NOC GD DE UPS FDX UNP CSX NSC ETN EMR WM
T VZ TMUS CMCSA
NEE DUK SO VST CEG GEV VRT
PDD JD BIDU LI XPEV
ENPH FSLR PLUG LCID
OKLO IONQ RKLB ASTS CRWV
""".split()


def _dedupe(symbols: List[str]) -> List[str]:
    seen, out = set(), []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_universe(spec: Any, provider_name: str, root: str = ".") -> List[str]:
    """Turn the `data.universe` setting into a list of tickers."""
    if isinstance(spec, (list, tuple)):
        return _dedupe([str(s) for s in spec])
    name = str(spec).strip()
    low = name.lower()
    if low == "auto":
        low = "liquid" if provider_name == "alpaca" else "core"
    if low == "core":
        return _dedupe(CORE)
    if low == "liquid":
        return _dedupe(LIQUID)
    path = Path(name)
    if not path.is_absolute():
        path = Path(root) / path
    if path.exists():
        text = path.read_text().replace(",", " ")
        syms = [tok for line in text.splitlines() for tok in line.split("#")[0].split()]
        return _dedupe(syms)
    raise ValueError(
        "data.universe must be 'auto', 'core', 'liquid', a list of tickers, "
        "or a path to a text file of tickers (got %r)." % (spec,)
    )


MIN_SYMBOLS = 8


def check_universe(symbols: List[str]) -> List[str]:
    if len(symbols) < MIN_SYMBOLS:
        raise ValueError(
            "The universe needs at least %d symbols (got %d): the model ranks stocks against "
            "each other, so it cannot work on one or two tickers." % (MIN_SYMBOLS, len(symbols))
        )
    return symbols
