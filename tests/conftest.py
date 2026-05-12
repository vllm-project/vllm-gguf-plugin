# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def example_prompts() -> list[str]:
    return [
        (
            "vLLM is a high-throughput and memory-efficient inference and "
            "serving engine for LLMs."
        ),
        "Briefly describe the major phases of the moon.",
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main differences between Python and C++?",
    ]
