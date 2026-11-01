"""Switchable production-style TSR pipeline (docs/system_design.md)."""

from opentsr.signs.pipeline.config import ablation_variants, load_config, normalize, stages_on
from opentsr.signs.pipeline.runner import Pipeline, read_log, run_sequence, write_log
from opentsr.signs.pipeline.schema import Candidate, FrameMeta, FrameRecord, Message, SignState
from opentsr.signs.pipeline.stages import STAGE_ORDER, FrameContext, PassThrough, Stage, available, build_stage, register

__all__ = [
    "STAGE_ORDER", "Candidate", "FrameContext", "FrameMeta", "FrameRecord", "Message", "PassThrough", "Pipeline",
    "SignState", "Stage", "ablation_variants", "available", "build_stage", "load_config", "normalize", "read_log",
    "register", "run_sequence", "stages_on", "write_log",
]
