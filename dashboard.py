"""Week 6 lab: analytics dashboard.

Reads the two background logs already written by this codebase —
`evaluation.log_decision` (`logs/evaluations.jsonl`, one entry per JD scored)
and `form_fill.log_application` (`logs/applications.jsonl`, one entry per
form-fill run) — and renders a single self-contained `dashboard.html`
summarizing application volume, field fill/skip rates, how often a run
reaches a Submit control, and the most common skip reasons. Both logs are
append-only JSONL, so this is safe to run at any point, repeatedly, across
however many `run_evaluation_lab.py` / `run_form_fill_lab.py` runs have
accumulated:

    python dashboard.py

No web server needed — `dashboard.html` is a plain static file, open it
directly in a browser.
"""

import html
import json
from collections import Counter
from pathlib import Path

EVALUATIONS_LOG = Path(__file__).parent / "logs" / "evaluations.jsonl"
APPLICATIONS_LOG = Path(__file__).parent / "logs" / "applications.jsonl"
OUTPUT_PATH = Path(__file__).parent / "dashboard.html"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_stats() -> dict:
    evaluations = _read_jsonl(EVALUATIONS_LOG)
    applications = _read_jsonl(APPLICATIONS_LOG)

    apply_count = sum(1 for e in evaluations if e["recommendation"] == "APPLY")
    avg_score = round(sum(e["match_score"] for e in evaluations) / len(evaluations), 1) if evaluations else 0.0

    total_filled = sum(a["filled"] for a in applications)
    total_skipped = sum(a["skipped"] for a in applications)
    total_fields = total_filled + total_skipped
    fill_rate = round(100 * total_filled / total_fields, 1) if total_fields else 0.0
    reached_submit = sum(1 for a in applications if a["reached_submit"])
    submit_reach_rate = round(100 * reached_submit / len(applications), 1) if applications else 0.0

    skip_reasons = Counter(reason for a in applications for reason in a["skip_reasons"])

    return {
        "evaluations": evaluations,
        "applications": applications,
        "eval_count": len(evaluations),
        "apply_count": apply_count,
        "skip_count": len(evaluations) - apply_count,
        "avg_score": avg_score,
        "application_count": len(applications),
        "total_filled": total_filled,
        "total_skipped": total_skipped,
        "fill_rate": fill_rate,
        "reached_submit": reached_submit,
        "submit_reach_rate": submit_reach_rate,
        "top_skip_reasons": skip_reasons.most_common(8),
    }


def _stat_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="value">{html.escape(value)}</div><div class="label">{html.escape(label)}</div></div>'


def _bar_row(label: str, count: int, max_count: int) -> str:
    pct = round(100 * count / max_count) if max_count else 0
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>'
        f'<div class="bar-count">{count}</div>'
        "</div>"
    )


def _application_row(a: dict) -> str:
    status = "reached submit" if a["reached_submit"] else "incomplete"
    url = html.escape(a["job_url"])
    return (
        "<tr>"
        f'<td>{html.escape(a["timestamp"][:19].replace("T", " "))}</td>'
        f'<td class="url"><a href="{url}">{url}</a></td>'
        f'<td>{a["steps"]}</td><td>{a["filled"]}</td><td>{a["skipped"]}</td>'
        f'<td>{status}</td>'
        "</tr>"
    )


def render_html(stats: dict) -> str:
    cards = "".join(
        [
            _stat_card("Job listings evaluated", str(stats["eval_count"])),
            _stat_card("Recommended APPLY", str(stats["apply_count"])),
            _stat_card("Avg match score", f'{stats["avg_score"]}/100'),
            _stat_card("Forms attempted", str(stats["application_count"])),
            _stat_card("Field fill rate", f'{stats["fill_rate"]}%'),
            _stat_card("Reached Submit", f'{stats["submit_reach_rate"]}%'),
        ]
    )

    max_reason = max((c for _, c in stats["top_skip_reasons"]), default=0)
    skip_bars = (
        "".join(_bar_row(reason, count, max_reason) for reason, count in stats["top_skip_reasons"])
        or '<p class="empty">No skipped fields recorded yet.</p>'
    )

    rows = (
        "".join(_application_row(a) for a in reversed(stats["applications"][-25:]))
        or '<tr><td colspan="6" class="empty">No form-fill runs recorded yet.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Auto-Apply Agent — Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f6f7f9; color: #1c1e21; margin: 0; padding: 32px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: #6b7280; font-size: 13px; margin: 0 0 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px; margin-bottom: 32px; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; }}
  .card .value {{ font-size: 24px; font-weight: 600; }}
  .card .label {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
  section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
             padding: 20px; margin-bottom: 24px; }}
  section h2 {{ font-size: 14px; margin: 0 0 16px; color: #374151; }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 13px; }}
  .bar-label {{ width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ flex: 1; background: #f0f1f3; border-radius: 6px; height: 10px; overflow: hidden; }}
  .bar-fill {{ background: #6366f1; height: 100%; }}
  .bar-count {{ width: 24px; text-align: right; color: #6b7280; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #f0f1f3; }}
  th {{ color: #6b7280; font-weight: 500; }}
  td.url {{ max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  td.url a {{ color: #4338ca; text-decoration: none; }}
  .empty {{ color: #9ca3af; font-style: italic; }}
</style>
</head>
<body>
  <h1>Auto-Apply Agent — Dashboard</h1>
  <p class="subtitle">Generated from logs/evaluations.jsonl and logs/applications.jsonl. Re-run <code>python dashboard.py</code> after more runs to refresh.</p>
  <div class="cards">{cards}</div>
  <section>
    <h2>Most common skip reasons</h2>
    {skip_bars}
  </section>
  <section>
    <h2>Recent form-fill runs (latest 25)</h2>
    <table>
      <thead><tr><th>Time (UTC)</th><th>Job URL</th><th>Steps</th><th>Filled</th><th>Skipped</th><th>Outcome</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def print_summary(stats: dict) -> None:
    print(f"Job listings evaluated: {stats['eval_count']} (APPLY {stats['apply_count']} / SKIP {stats['skip_count']})")
    print(f"Average match score: {stats['avg_score']}/100")
    print(f"Form-fill runs: {stats['application_count']}")
    print(f"Field fill rate: {stats['fill_rate']}% ({stats['total_filled']} filled / {stats['total_skipped']} skipped)")
    print(f"Reached Submit: {stats['submit_reach_rate']}% ({stats['reached_submit']}/{stats['application_count']})")
    if stats["top_skip_reasons"]:
        print("Top skip reasons:")
        for reason, count in stats["top_skip_reasons"]:
            print(f"  {count:>3}  {reason}")


def main() -> None:
    stats = build_stats()
    OUTPUT_PATH.write_text(render_html(stats))
    print_summary(stats)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
