"""Versioned business waiting policy; never part of semantic intent or retry identity."""

from typing import Literal

BusinessWaitPolicy = Literal["v11-business-wait.2"]
BUSINESS_WAIT_POLICY: BusinessWaitPolicy = "v11-business-wait.2"
BUSINESS_MODEL_WAIT_SECONDS = 60
REVIEW_COMMAND_SECONDS = 120
