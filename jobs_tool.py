"""Job search tool backed by JobSpy.

The Week 2 roadmap's lab task was originally "open LinkedIn, log in, search
Java Engineer, print the top 5 results." We do the search step through
JobSpy instead of an authenticated browser session: LinkedIn's User
Agreement prohibits automated login/scraping and actively bans accounts for
it, whereas JobSpy queries the same public job-search results LinkedIn
serves to anyone without an account. See week2/README.md for the full
reasoning. `browser_tools.browse_page` still gives the agent a real,
Playwright-driven browser for the AOM-parsing / visual-debugging half of the
lab.
"""

from langchain_core.tools import tool
from jobspy import scrape_jobs

DEFAULT_SITES = ["linkedin", "indeed"]


@tool
def search_jobs(search_term: str, location: str = "Remote", results_wanted: int = 5) -> str:
    """Search job boards (LinkedIn, Indeed) for a role and return the title,
    company, and link for each result. No login required."""
    jobs = scrape_jobs(
        site_name=DEFAULT_SITES,
        search_term=search_term,
        location=location,
        results_wanted=results_wanted,
        country_indeed="USA",
        linkedin_fetch_description=False,
    )

    if jobs.empty:
        return f"No results for '{search_term}' in '{location}'."

    jobs = jobs.head(results_wanted)
    lines = [
        f"{i}. {row.title} — {row.company} — {row.job_url}"
        for i, row in enumerate(jobs.itertuples(), start=1)
    ]
    return "\n".join(lines)
