"""Benchmark adapter, part 1: how the host benchmark represents trajectories and injections.

The analysis modules (trajectory labeling, attack-chain judge, attribution) import these names from
here. It is kept apart from ``latent_risk.benchmark`` because prompt-siren's job storage imports
the analysis modules, so importing prompt-siren's job code from here would be circular.

To port to another benchmark, provide:
  - ``InjectableModelMessagesTypeAdapter``: validates a trajectory (a list of pydantic-ai messages,
    possibly with injection placeholders) from its JSON form;
  - ``InjectionAttack`` and ``InjectionAttacksDictTypeAdapter``: an injected payload and the
    mapping from injection vector ids to payloads stored with each run.
"""

from prompt_siren.types import (
    InjectableModelMessagesTypeAdapter,
    InjectionAttack,
    InjectionAttacksDictTypeAdapter,
)

__all__ = [
    "InjectableModelMessagesTypeAdapter",
    "InjectionAttack",
    "InjectionAttacksDictTypeAdapter",
]
