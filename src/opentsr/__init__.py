"""OpenTSR: open-source, open-data traffic sign recognition.

Layout: shared code (evaluation metrics, split manifests) lives at the package root,
``opentsr.signs`` is the main track and ``opentsr.traffic_lights`` is the secondary
traffic light track. Nothing in the shared modules or in ``opentsr.signs`` may import from
``opentsr.traffic_lights``.
"""

__version__ = "0.0.1"
