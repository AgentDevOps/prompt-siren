import sys
from pathlib import Path

# The CLI scripts are imported as the `scripts` package from the latent_risk root.
LATENT_RISK_ROOT = Path(__file__).resolve().parents[1]
if str(LATENT_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_RISK_ROOT))
