"""Week 2 lab: open LinkedIn in a real, visible browser, log in, search
"Java Engineer", and print the top 5 results' titles and links.

Run directly so you can watch agent-browser operate the window:

    python run_linkedin_lab.py

Needs LINKEDIN_EMAIL / LINKEDIN_PASSWORD set in .env. If LinkedIn shows a
security checkpoint mid-login, solve it in the visible window when prompted.
"""

from dotenv import load_dotenv

from linkedin_tool import search_linkedin_jobs

load_dotenv()

if __name__ == "__main__":
    print(search_linkedin_jobs("Java Engineer", results_wanted=5))
