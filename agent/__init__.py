"""Property Insights Assistant — LangGraph agent package.

Loads .env from the project root as soon as anything imports from this
package, before any submodule reads os.environ at import time (db.py's
DATABASE_URL, llm.py's PROVIDER, graph.py's JUDGE_SAMPLE_RATE) — Python
always runs a package's __init__.py before its submodules, so this
ordering is guaranteed regardless of which entry point (agent.run,
agent.eval, ml.model, ...) started the process.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
