"""Synthetic Data Generator for Logic & Math."""
from __future__ import annotations
import random
from typing import Iterator, Tuple

class SyntheticDataPipeline:
    """Streams synthetic logic tasks."""

    def __init__(self, curriculum_level: str = "easy"):
        self.level = curriculum_level

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        while True:
            yield self.generate_sample()

    def generate_sample(self) -> Tuple[str, str]:
        """Generate a (prompt, completion) pair."""
        if self.level == "easy":
            return self._generate_math_easy()
        elif self.level == "medium":
            return self._generate_math_medium()
        else:
            return self._generate_sequence()

    def _generate_math_easy(self) -> Tuple[str, str]:
        """Simple addition/subtraction."""
        op = random.choice(["+", "-"])
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        if op == "+":
            ans = a + b
        else:
            ans = a - b
        return f"{a} {op} {b} =", str(ans)

    def _generate_math_medium(self) -> Tuple[str, str]:
        """Mixed operators with multiplication."""
        op = random.choice(["+", "-", "*"])
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b
        return f"{a} {op} {b} =", str(ans)

    def _generate_sequence(self) -> Tuple[str, str]:
        """Number sequences."""
        start = random.randint(0, 10)
        step = random.randint(1, 5)
        length = 5
        seq = [start + i * step for i in range(length)]
        prompt = " ".join(map(str, seq[:-1])) + " next is"
        completion = str(seq[-1])
        return prompt, completion
