"""Async email syntax and MX validation for large CSV files."""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    import aiodns  # type: ignore
except ImportError:  # pragma: no cover - fallback is used only when aiodns is unavailable
    aiodns = None

try:
    import dns.asyncresolver  # type: ignore
except ImportError:  # pragma: no cover - fallback is used only when dnspython is unavailable
    dns = None  # type: ignore
else:  # pragma: no cover - imported only when dnspython is available
    dns = dns  # type: ignore

EMAIL_REGEX = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)[A-Z0-9._%+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ProcessingProgress:
    """Progress payload emitted while chunk processing runs."""

    chunk_index: int
    chunks_total: int
    rows_processed: int
    total_rows: int
    working_count: int
    invalid_count: int
    progress_ratio: float


@dataclass(slots=True)
class VerificationSummary:
    """Final output summary for a processed CSV file."""

    input_path: str
    output_working: str
    output_invalid: str
    email_column: str
    total_rows: int
    working_count: int
    invalid_count: int
    processed_chunks: int


class _AioDnsMxResolver:
    """aiodns-backed MX resolver."""

    def __init__(self) -> None:
        if aiodns is None:
            raise ImportError("aiodns is not installed. Install dependencies from requirements.txt.")
        self._resolver = aiodns.DNSResolver(timeout=2.5, tries=2)

    async def query_mx(self, domain: str) -> bool:
        try:
            response = await self._resolver.query(domain, "MX")
            return bool(response)
        except Exception:
            return False


class _DnsPythonMxResolver:
    """dnspython async resolver used as a stable fallback."""

    def __init__(self) -> None:
        if dns is None:
            raise ImportError("dnspython is not installed. Install dependencies from requirements.txt.")
        self._resolver = dns.asyncresolver.Resolver()
        self._resolver.timeout = 2.5
        self._resolver.lifetime = 3.0

    async def query_mx(self, domain: str) -> bool:
        try:
            response = await self._resolver.resolve(domain, "MX")
            return bool(response)
        except Exception:
            return False


class _AsyncMxResolver:
    def __init__(self) -> None:
        if aiodns is not None:
            self._backend: Any = _AioDnsMxResolver()
        else:
            self._backend = _DnsPythonMxResolver()

    async def query_mx(self, domain: str) -> bool:
        return await self._backend.query_mx(domain)


def normalize_email(value: object) -> str:
    """Convert a raw CSV cell into a trimmed email candidate string."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_syntax_valid(email: str) -> bool:
    """Fast regex gate for basic email syntax checks."""
    if not email:
        return False
    return bool(EMAIL_REGEX.fullmatch(email))


def extract_domain(email: str) -> str:
    """Return the lowercase domain component from a validated email string."""
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower()


def count_csv_rows(csv_path: str | Path) -> int:
    """Count data rows in a CSV file without loading it into memory."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        total_lines = sum(1 for _ in handle)
    return max(0, total_lines - 1)


async def _resolve_domains(
    domains: set[str],
    resolver: _AsyncMxResolver,
    concurrency: int,
) -> dict[str, bool]:
    """Resolve many domains concurrently and return an MX cache."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _resolve_one(domain: str) -> tuple[str, bool]:
        async with semaphore:
            has_mx = await resolver.query_mx(domain)
            return domain, has_mx

    tasks = [_resolve_one(domain) for domain in domains]
    resolved = await asyncio.gather(*tasks)
    return {domain: has_mx for domain, has_mx in resolved}


async def validate_emails(
    emails: list[object],
    *,
    domain_cache: dict[str, bool] | None = None,
    concurrency: int = 250,
    resolver: _AsyncMxResolver | None = None,
) -> list[dict[str, object]]:
    """Validate a list of email candidates using regex + MX verification."""
    cache = domain_cache if domain_cache is not None else {}
    mx_resolver = resolver or _AsyncMxResolver()

    normalized = [normalize_email(email) for email in emails]
    syntax_rows: list[tuple[str, str, bool, str]] = []
    domains_to_lookup: set[str] = set()

    for email in normalized:
        if not email:
            syntax_rows.append(("", "", False, "empty_value"))
            continue

        if not is_syntax_valid(email):
            syntax_rows.append((email, extract_domain(email), False, "syntax_invalid"))
            continue

        domain = extract_domain(email)
        syntax_rows.append((email, domain, True, ""))
        if domain and domain not in cache:
            domains_to_lookup.add(domain)

    if domains_to_lookup:
        cache.update(await _resolve_domains(domains_to_lookup, mx_resolver, concurrency))

    results: list[dict[str, object]] = []
    for email, domain, syntax_ok, reason in syntax_rows:
        if not syntax_ok:
            results.append(
                {
                    "email": email,
                    "domain": domain,
                    "status": "invalid",
                    "reason": reason,
                }
            )
            continue

        has_mx = cache.get(domain, False)
        results.append(
            {
                "email": email,
                "domain": domain,
                "status": "working" if has_mx else "invalid",
                "reason": "mx_verified" if has_mx else "mx_missing",
            }
        )

    return results


def _append_frame(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    frame.to_csv(output_path, mode="a", index=False, header=not output_path.exists())


async def _process_csv_async(
    input_path: Path,
    output_working: Path,
    output_invalid: Path,
    email_column: str,
    chunksize: int,
    concurrency: int,
    progress_callback: Callable[[ProcessingProgress], None] | None,
) -> VerificationSummary:
    header = pd.read_csv(input_path, nrows=0)
    if email_column not in header.columns:
        available = ", ".join(map(str, header.columns.tolist()))
        raise ValueError(f"Column '{email_column}' was not found. Available columns: {available}")

    total_rows = count_csv_rows(input_path)
    total_chunks = max(1, math.ceil(total_rows / chunksize))
    domain_cache: dict[str, bool] = {}
    resolver = _AsyncMxResolver()

    output_working.parent.mkdir(parents=True, exist_ok=True)
    output_invalid.parent.mkdir(parents=True, exist_ok=True)
    output_working.unlink(missing_ok=True)
    output_invalid.unlink(missing_ok=True)

    processed_rows = 0
    working_count = 0
    invalid_count = 0
    processed_chunks = 0

    iterator = pd.read_csv(
        input_path,
        usecols=[email_column],
        chunksize=chunksize,
        dtype={email_column: "string"},
        keep_default_na=False,
        na_filter=False,
    )

    for chunk_index, chunk in enumerate(iterator, start=1):
        processed_chunks = chunk_index
        emails = chunk[email_column].tolist()
        validation_rows = await validate_emails(
            emails,
            domain_cache=domain_cache,
            concurrency=concurrency,
            resolver=resolver,
        )
        result_frame = pd.DataFrame(validation_rows, columns=["email", "domain", "status", "reason"])

        working_frame = result_frame[result_frame["status"] == "working"][["email", "domain", "reason"]]
        invalid_frame = result_frame[result_frame["status"] == "invalid"][["email", "domain", "reason"]]

        _append_frame(working_frame, output_working)
        _append_frame(invalid_frame, output_invalid)

        chunk_total = len(result_frame)
        processed_rows += chunk_total
        working_count += len(working_frame)
        invalid_count += len(invalid_frame)

        if progress_callback is not None:
            progress_callback(
                ProcessingProgress(
                    chunk_index=chunk_index,
                    chunks_total=total_chunks,
                    rows_processed=processed_rows,
                    total_rows=total_rows,
                    working_count=working_count,
                    invalid_count=invalid_count,
                    progress_ratio=min(1.0, processed_rows / max(1, total_rows)),
                )
            )

    return VerificationSummary(
        input_path=str(input_path),
        output_working=str(output_working),
        output_invalid=str(output_invalid),
        email_column=email_column,
        total_rows=processed_rows,
        working_count=working_count,
        invalid_count=invalid_count,
        processed_chunks=processed_chunks,
    )


def process_csv(
    input_path: str | Path,
    output_working: str | Path,
    output_invalid: str | Path,
    email_column: str = "email",
    chunksize: int = 50_000,
    concurrency: int = 250,
    progress_callback: Callable[[ProcessingProgress], None] | None = None,
) -> VerificationSummary:
    """Process a CSV file in chunks and split working and invalid emails into separate outputs."""
    input_file = Path(input_path)
    working_file = Path(output_working)
    invalid_file = Path(output_invalid)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    return asyncio.run(
        _process_csv_async(
            input_file,
            working_file,
            invalid_file,
            email_column,
            chunksize,
            concurrency,
            progress_callback,
        )
    )
