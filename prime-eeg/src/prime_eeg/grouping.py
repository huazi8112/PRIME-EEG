"""Independent subjects are the resampling unit, not recording stages."""
from .evaluation import independent_subject


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(independent_subject(row["subject"]), []).append(row)
    return {key: groups[key] for key in sorted(groups)}
