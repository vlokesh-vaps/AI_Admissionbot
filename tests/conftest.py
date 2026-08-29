import sys
from pathlib import Path

# Add services/admission_chatbot to sys.path so tests can find `src`
service_root = Path(__file__).resolve().parents[1]
if str(service_root) not in sys.path:
    sys.path.insert(0, str(service_root))
