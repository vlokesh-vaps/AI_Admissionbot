import sys
import os
from pathlib import Path

# Add services/admission_chatbot to sys.path so tests can find `src`
service_root = Path(__file__).resolve().parents[1]
if str(service_root) not in sys.path:
    sys.path.insert(0, str(service_root))

# Tests deliberately use ephemeral Qdrant; production defaults to fail-fast persistence.
os.environ.setdefault("QDRANT_ALLOW_IN_MEMORY", "true")
