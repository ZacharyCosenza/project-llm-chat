"""core/reward.py
--------------
Reward functions for online RL training.

  PythonExecutor    Executes Python code extracted from model responses and scores
                    the output. Handles two task modes:
                    - Calculator: extract ```python block, run it, compare stdout.
                    - Code tasks: extract function body, assemble with test harness, run.

                    __call__ returns (reward: float, loss_mask: Tensor) where loss_mask
                    is 1.0 for tokens the executor considers trainable. Subclasses can
                    override to mask specific token spans (e.g. only assistant turns in
                    multi-turn RL).

Add new reward classes here as needed.
"""

import re
import subprocess
import sys
import torch


def _extract_code_block(text: str) -> str | None:
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None


def _run_code(code: str, timeout: float) -> tuple[str, bool]:
    try:
        r = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip(), r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return '', False


class PythonExecutor:
    """Reward via Python code execution.

    Two task modes selected by which kwargs are passed:

    Calculator  (expected='42.0', test_suffix=None):
        Extract ```python block from response, execute, compare stdout to expected.
        Partial credit: +0.3 executes, +0.2 numeric output, +0.5 correct value.
        Falls back to last-number parse if no code block found.

    Code task   (test_suffix='<assertions>', expected=None):
        Extract model's function body, assemble with optional test_prefix + test_suffix,
        execute. Partial credit: +0.3 compiles, +0.7 all assertions pass.
    """

    def __init__(self, timeout: float = 5.0, max_new_tokens: int = 384, pad_token_id: int = 0):
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.pad_token_id = pad_token_id

    def __call__(self, pid: torch.Tensor, gen_ids: torch.Tensor, response: str,
                 expected: str | None = None, test_prefix: str | None = None,
                 test_suffix: str | None = None) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Returns (reward, gen_ids_padded, loss_mask).

        gen_ids_padded: gen_ids padded to max_new_tokens with pad_token_id.
        loss_mask: float [T_prompt + max_new_tokens - 1] — 0 for prompt and padding, 1 for
                   real generated tokens. Subclasses may return finer-grained masks.
        """
        reward = self._reward(response, expected, test_prefix, test_suffix)

        T_prompt, T_gen = pid.shape[0], gen_ids.shape[0]
        pad_len = self.max_new_tokens - T_gen
        gen_ids_padded = torch.cat([
            gen_ids,
            torch.full((pad_len,), self.pad_token_id, dtype=gen_ids.dtype, device=gen_ids.device),
        ]) if pad_len > 0 else gen_ids[:self.max_new_tokens]

        loss_mask = torch.zeros(T_prompt + self.max_new_tokens - 1, dtype=torch.float, device=gen_ids.device)
        loss_mask[T_prompt - 1 : T_prompt - 1 + min(T_gen, self.max_new_tokens)] = 1.0

        return reward, gen_ids_padded, loss_mask

    def _reward(self, response: str, expected: str | None = None,
                test_prefix: str | None = None, test_suffix: str | None = None) -> float:
        code = _extract_code_block(response)

        if test_suffix is not None:
            # HumanEval / MBPP: assemble test harness and run
            if code is None:
                return 0.0
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError:
                return 0.0
            harness = (test_prefix or '') + code + '\n' + test_suffix
            _, ok = _run_code(harness, self.timeout)
            return 0.3 + (0.7 if ok else 0.0)

        # Calculator: execute code block and score stdout
        if code:
            stdout, ok = _run_code(code, self.timeout)
            return self._score_stdout(stdout, expected, ok)

        # No code block: try last number in plain text
        nums = re.findall(r'-?\d+\.?\d*', response)
        if not nums:
            return 0.0
        try:
            got = float(nums[-1])
            exp = float(str(expected))
            return 1.0 if abs(got - exp) <= 1e-3 * (abs(exp) + 1e-8) else 0.1
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _score_stdout(stdout: str, expected: str | None, success: bool) -> float:
        reward = 0.3 if success else 0.0
        try:
            got = float(stdout.replace(',', ''))
            reward += 0.2
            exp = float(str(expected).replace(',', ''))
            if abs(got - exp) <= 1e-3 * (abs(exp) + 1e-8):
                reward += 0.5
        except (ValueError, TypeError):
            if success and stdout.strip().lower() == str(expected).strip().lower():
                reward += 0.7
        return reward
