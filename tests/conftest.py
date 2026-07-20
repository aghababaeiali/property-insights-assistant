"""
Shared pytest config.

Forces LLM_PROVIDER=offline and JUDGE_SAMPLE_RATE=0 before any `agent.*`
module gets imported — pytest always imports conftest.py before collecting
test modules, so this runs first. python-dotenv's load_dotenv() (called
from agent/__init__.py on first import) defaults to override=False, so it
won't clobber these once they're already set here — the whole suite runs
deterministically with zero external API calls, no GROQ_API_KEY needed.

Still needs Postgres reachable (`docker compose up -d --wait`): these are
integration tests against the real query/training pipeline, not mocks.
"""
import os

os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("JUDGE_SAMPLE_RATE", "0")
