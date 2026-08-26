"""Machine-code normalization shared by MySQL and dashboard callers."""

from __future__ import annotations

import re


MACHINE_RE = re.compile(r"^(?:MX)?(?P<number>[0-9]{1,20})$", re.IGNORECASE)


def normalize_machine(value: str) -> str:
    """Return the canonical display key, for example ``MX057``."""
    match = MACHINE_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError("machine_id must look like MX012")
    return f"MX{match.group('number').zfill(3)}"


def database_machine_candidates(machine_id: str) -> list[str]:
    """Return common database representations without guessing a new machine."""
    canonical = normalize_machine(machine_id)
    number = str(int(canonical[2:]))
    padded = canonical[2:]
    return list(dict.fromkeys((canonical, padded, number)))
