# AI Generated macros to make documentation in docs/dlog appear on index.md automatically
from pathlib import Path
from datetime import datetime
import frontmatter

DATE_FMT = "%Y-%m-%d"


def _to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _parse_minutes(sessions):
    total = 0
    for s in sessions or []:
        if "in" in s and "out" in s:
            total += _to_minutes(s["out"]) - _to_minutes(s["in"])
    return max(total, 0)


def _load_entries(docs_dir: Path):
    dlog_dir = docs_dir / "dlog"

    entries = []
    if not dlog_dir.exists():
        return entries

    for p in sorted(dlog_dir.glob("*.md")):
        fm = frontmatter.load(p)
        meta = fm.metadata or {}
        date = meta.get("date")
        sessions = meta.get("sessions", [])
        minutes = _parse_minutes(sessions)
        entries.append({
            "date": date,
            "date_str": date.strftime(DATE_FMT),
            "goal": meta.get("goal", ""),
            "summary": meta.get("summary", ""),
            "minutes": minutes,
            "path": str(p.relative_to(docs_dir)).replace("\\", "/"),
            "tags": meta.get("tags", []),
        })

    # newest first
    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def define_env(env):
    """
    mkdocs-macros entrypoint. Register macros here.
    """
    project_dir = Path(env.project_dir)         # repo root
    docs_dir = project_dir / env.conf['docs_dir']  # usually "docs"

    @env.macro
    def dlog_total_time():
        total_minutes = sum(e["minutes"] for e in _load_entries(docs_dir))
        return {
            "hrs": total_minutes // 60,
            "mins": total_minutes % 60,
            "raw_mins": total_minutes
        }

    @env.macro
    def dlog_num_days():
        return len(_load_entries(docs_dir))

    @env.macro
    def dlog_cards(limit=7):
        items = _load_entries(docs_dir)[:int(limit)]
        if not items:
            return "_No entries yet._"
        out = []
        for e in items:
            tags_str = " · ".join(e["tags"]) if e["tags"] else ""
            line = (
                f'- **[{e["date_str"]}]({e["path"]})** | {e["minutes"]//60} hr {e["minutes"] % 60} min | '
                f'**Goal:** {e["goal"]}  \n'
                f'  _{e["summary"]}_  \n'
            )
            if tags_str:
                line += f'  \n     <sub>{tags_str}</sub>'
            out.append(line)
        return "\n\n".join(out)

    @env.macro
    def dlog_consecutive_days():
        """
        Returns the number of consecutive days (streak) from the most recent date backward.
        """
        entries = _load_entries(docs_dir)
        if not entries:
            return 0

        # Extract unique sorted dates (ascending)
        dates = sorted(set(e["date"] for e in entries))

        # Walk backwards from the latest day
        streak = 1
        for i in range(len(dates) - 1, 0, -1):
            diff = (dates[i] - dates[i - 1]).days
            if diff == 1:
                streak += 1
            else:
                break
        return streak
