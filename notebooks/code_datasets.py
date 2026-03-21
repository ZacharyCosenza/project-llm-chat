"""
code_datasets.py
----------------
Dataset classes and reward utilities for --mode code in run_13_code.py.

  CalculatorDataset      Programmatic arithmetic / statistics / conversion problems.
                         Each sample is a full (question, response) pair with step-by-step
                         reasoning. 20% of samples contain a deliberate arithmetic error
                         followed by self-correction via a Python code block.

  CodeProblemsDataset    HumanEval / MBPP from HuggingFace for function-completion tasks.

Run directly to execute all validation checks:
  python notebooks/code_datasets.py
"""

import random
import re
import subprocess
import collections
from torch.utils.data import IterableDataset

# =============================================================================
# Code Execution Utilities
# =============================================================================

def extract_code_block(text: str) -> str | None:
    """Extract first ```python ... ``` block from model output."""
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None


def execute_code(code: str, timeout: float = 5.0) -> tuple[str, bool]:
    """Run code in a subprocess. Returns (stdout, success)."""
    try:
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return '', False
    except Exception:
        return '', False


def compute_reward(stdout: str, expected: str, success: bool) -> float:
    """Partial credit reward for calculator-style tasks.
    +0.3  code executed without error
    +0.2  stdout is parseable as a number
    +0.5  number matches expected within relative tolerance 1e-3
    """
    reward = 0.0
    if success:
        reward += 0.3
    try:
        got = float(stdout.replace(',', '').strip())
        reward += 0.2
        exp = float(expected.replace(',', '').strip())
        if abs(got - exp) <= 1e-3 * (abs(exp) + 1e-8):
            reward += 0.5
    except ValueError:
        if success and stdout.strip().lower() == expected.strip().lower():
            reward += 0.7  # non-numeric exact match
    return reward


def compute_code_reward(test_code: str, model_code: str, timeout: float = 5.0) -> float:
    """Partial credit reward for HumanEval/MBPP tasks.

    test_code  : fully assembled Python to execute (stub + model completion + assertions).
    model_code : the model's completion only, used for syntax checking.

    +0.3  model_code compiles without SyntaxError
    +0.7  test_code runs and all assertions pass
    """
    try:
        compile(model_code, '<string>', 'exec')
    except SyntaxError:
        return 0.0
    _, success = execute_code(test_code, timeout=timeout)
    return 0.3 + (0.7 if success else 0.0)


def compute_calculator_reward(response: str, expected: str) -> float:
    """Reward for a model-generated calculator-style response.

    If the response contains a ```python block, execute it and score stdout.
    Otherwise extract the last number from the text and compare.
    Returns a float in [0.0, 1.0].
    """
    code = extract_code_block(response)
    if code:
        stdout, ok = execute_code(code)
        return compute_reward(stdout, expected, ok)

    numbers = re.findall(r'-?\d+\.?\d*', response)
    if not numbers:
        return 0.0
    try:
        got = float(numbers[-1])
        exp = float(expected)
        return 1.0 if abs(got - exp) <= 1e-3 * (abs(exp) + 1e-8) else 0.1
    except ValueError:
        return 0.0


# =============================================================================
# CalculatorDataset
# =============================================================================

_VERIFY_PHRASES = [
    "Hmm, let me double-check that with Python.",
    "Wait, let me verify that using a calculator.",
    "Actually, let me make sure by running this in Python.",
    "Let me confirm that calculation with code.",
    "I should double-check this — let me use Python to be sure.",
    "That doesn't feel right. Let me verify with Python.",
]


class CalculatorDataset(IterableDataset):
    """Infinite stream of step-by-step reasoning problems with optional error + correction.

    Each item yields:
      question        (str)   natural-language problem
      response        (str)   full assistant response with reasoning steps
      expected_output (str)   correct numerical answer
      has_error       (bool)  True for the 20% error+correction samples
      problem_type    (str)   'arithmetic' | 'statistics' | 'conversion'
      calculator_code (str)   reference Python code — always correct

    ERROR_RATE (default 0.20) of samples contain a deliberate arithmetic error
    in the reasoning chain. The error propagates forward through subsequent steps,
    and is followed by self-correction using a Python code block. The code block
    is always correct; only the natural-language reasoning is wrong.
    """

    ERROR_RATE = 0.20

    CONVERSIONS = [
        ('miles',     'kilometers',  1.60934,  'v * 1.60934'),
        ('kilograms', 'pounds',      2.20462,  'v * 2.20462'),
        ('feet',      'meters',      0.3048,   'v * 0.3048'),
        ('gallons',   'liters',      3.78541,  'v * 3.78541'),
        ('inches',    'centimeters', 2.54,     'v * 2.54'),
        ('ounces',    'grams',       28.3495,  'v * 28.3495'),
    ]

    def __init__(self, seed: int = 42, rank: int = 0, max_problems: int | None = None):
        self.seed = seed + rank
        self.max_problems = max_problems

    @staticmethod
    def _wiggle(rng: random.Random, value: float) -> float:
        """Return a plausibly wrong value: 5–15% offset, minimum ±2."""
        delta = max(2.0, abs(value) * rng.uniform(0.05, 0.15))
        return round(value + rng.choice([-1, 1]) * delta, 4)

    @staticmethod
    def _format_response(
        step_lines: list[str],
        correct_final: str,
        introduce_error: bool = False,
        wrong_final: str | None = None,
        calculator_code: str | None = None,
        rng: random.Random | None = None,
    ) -> str:
        body = "Let me work through this step by step.\n\n"
        body += "\n".join(step_lines)
        if not introduce_error:
            body += f"\n\nThe answer is {correct_final}."
        else:
            body += f"\n\nSo the answer would be {wrong_final}."
            phrase = rng.choice(_VERIFY_PHRASES)
            body += f"\n\n{phrase}\n\n```python\n{calculator_code}\n```\n\n"
            body += (
                f"The calculator gives {correct_final}, so I made an arithmetic error earlier. "
                f"The correct answer is {correct_final}."
            )
        return body

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _gen_arithmetic(self, rng: random.Random, introduce_error: bool) -> dict:
        ops = ['+', '-', '*']
        sym = {'+': '+', '-': '−', '*': '×'}

        n_terms    = rng.randint(3, 5)
        nums       = [float(rng.randint(2, 40)) for _ in range(n_terms)]
        chosen_ops = [rng.choice(ops) for _ in range(n_terms - 1)]

        # Correct intermediate values, left-to-right
        vals = [nums[0]]
        for op, n in zip(chosen_ops, nums[1:]):
            prev = vals[-1]
            vals.append(prev + n if op == '+' else prev - n if op == '-' else prev * n)

        correct_final = str(round(vals[-1], 4))

        # Display expression (no parens) for the question
        expr_display  = ' '.join([str(int(nums[0]))] + [f"{op} {int(n)}" for op, n in zip(chosen_ops, nums[1:])])
        question      = f"Calculate the following expression step by step:\n{expr_display}"

        # Fully parenthesized expression for calc_code so Python's operator
        # precedence matches the left-to-right evaluation shown in the steps.
        running_expr  = str(int(nums[0]))
        for op, n in zip(chosen_ops, nums[1:]):
            running_expr = f"({running_expr} {op} {int(n)})"
        calc_code     = f"print(round({running_expr}, 4))"

        def _make_steps(v: list[float]) -> list[str]:
            return [
                f"Step {i+1}: {round(v[i], 4)} {sym[op]} {int(n)} = {round(v[i+1], 4)}"
                for i, (op, n) in enumerate(zip(chosen_ops, nums[1:]))
            ]

        if not introduce_error:
            response = self._format_response(_make_steps(vals), correct_final)
        else:
            err_idx            = rng.randrange(len(chosen_ops))
            wvals              = list(vals)
            wvals[err_idx + 1] = self._wiggle(rng, vals[err_idx + 1])
            for i in range(err_idx + 1, len(chosen_ops)):
                p, op, n = wvals[i], chosen_ops[i], nums[i + 1]
                wvals[i + 1] = p + n if op == '+' else p - n if op == '-' else p * n
            wvals    = [round(v, 4) for v in wvals]
            response = self._format_response(
                _make_steps(wvals), correct_final,
                introduce_error=True, wrong_final=str(wvals[-1]),
                calculator_code=calc_code, rng=rng,
            )

        return dict(question=question, response=response, expected_output=correct_final,
                    has_error=introduce_error, problem_type='arithmetic', calculator_code=calc_code)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _gen_statistics(self, rng: random.Random, introduce_error: bool) -> dict:
        n    = rng.randint(4, 8)
        nums = [rng.randint(1, 50) for _ in range(n)]
        stat = rng.choice(['mean', 'median'])

        if stat == 'mean':
            correct_sum   = sum(nums)
            correct_mean  = round(correct_sum / n, 4)
            correct_final = str(correct_mean)
            question      = f"Find the mean of the following list:\n{nums}"
            calc_code     = f"nums = {nums}\nprint(round(sum(nums) / len(nums), 4))"

            if not introduce_error:
                response = self._format_response(
                    [f"Step 1: Sum = {' + '.join(map(str, nums))} = {correct_sum}",
                     f"Step 2: Divide by count ({n}): {correct_sum} / {n} = {correct_mean}"],
                    correct_final,
                )
            else:
                wrong_sum  = int(self._wiggle(rng, correct_sum))
                wrong_mean = round(wrong_sum / n, 4)
                response   = self._format_response(
                    [f"Step 1: Sum = {' + '.join(map(str, nums))} = {wrong_sum}",
                     f"Step 2: Divide by count ({n}): {wrong_sum} / {n} = {wrong_mean}"],
                    correct_final,
                    introduce_error=True, wrong_final=str(wrong_mean),
                    calculator_code=calc_code, rng=rng,
                )

        else:  # median
            sorted_nums = sorted(nums)
            mid         = n // 2
            if n % 2 == 1:
                correct_median = float(sorted_nums[mid])
                median_desc    = f"Middle element at index {mid}: {sorted_nums[mid]}"
            else:
                correct_median = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
                median_desc    = (f"Average the two middle values "
                                  f"({sorted_nums[mid-1]} + {sorted_nums[mid]}) / 2 = {correct_median}")
            correct_final = str(round(correct_median, 4))
            question      = f"Find the median of the following list:\n{nums}"
            calc_code     = f"import statistics\nprint(round(statistics.median({nums}), 4))"

            if not introduce_error:
                response = self._format_response(
                    [f"Step 1: Sort the list: {sorted_nums}", f"Step 2: {median_desc}"],
                    correct_final,
                )
            else:
                wrong_median = round(self._wiggle(rng, correct_median), 4)
                wrong_desc   = median_desc.rsplit('= ', 1)[0] + f"= {wrong_median}"
                response     = self._format_response(
                    [f"Step 1: Sort the list: {sorted_nums}", f"Step 2: {wrong_desc}"],
                    correct_final,
                    introduce_error=True, wrong_final=str(wrong_median),
                    calculator_code=calc_code, rng=rng,
                )

        return dict(question=question, response=response, expected_output=correct_final,
                    has_error=introduce_error, problem_type='statistics', calculator_code=calc_code)

    # ------------------------------------------------------------------
    # Unit conversions
    # ------------------------------------------------------------------

    def _gen_conversion(self, rng: random.Random, introduce_error: bool) -> dict:
        use_celsius = rng.random() < 0.25

        if use_celsius:
            value          = round(rng.uniform(-20, 100), 1)
            correct_result = round(value * 9 / 5 + 32, 4)
            name_from, name_to = 'Celsius', 'Fahrenheit'
            factor_desc    = "The formula is: °F = °C × (9/5) + 32"
            mult_desc      = f"Step 2: {value} × (9/5) + 32 = {correct_result}"
            calc_code      = f"c = {value}\nprint(round(c * 9/5 + 32, 4))"
        else:
            idx            = rng.randrange(len(self.CONVERSIONS))
            name_from, name_to, factor, code_expr = self.CONVERSIONS[idx]
            value          = round(rng.uniform(1, 100), 1)
            correct_result = round(value * factor, 4)
            factor_desc    = f"The conversion factor is 1 {name_from} = {factor} {name_to}"
            mult_desc      = f"Step 2: {value} × {factor} = {correct_result}"
            calc_code      = f"v = {value}\nprint(round({code_expr}, 4))"

        correct_final = str(correct_result)
        question      = f"Convert {value} {name_from} to {name_to}."

        if not introduce_error:
            response = self._format_response(
                [f"Step 1: {factor_desc}.", mult_desc],
                f"{correct_final} {name_to}",
            )
        else:
            wrong_result = round(self._wiggle(rng, correct_result), 4)
            response     = self._format_response(
                [f"Step 1: {factor_desc}.", mult_desc.rsplit('= ', 1)[0] + f"= {wrong_result}"],
                f"{correct_final} {name_to}",
                introduce_error=True, wrong_final=f"{wrong_result} {name_to}",
                calculator_code=calc_code, rng=rng,
            )

        return dict(question=question, response=response, expected_output=correct_final,
                    has_error=introduce_error, problem_type='conversion', calculator_code=calc_code)

    # ------------------------------------------------------------------
    # IterableDataset
    # ------------------------------------------------------------------

    def __iter__(self):
        rng = random.Random(self.seed)
        generators = {
            'arithmetic': self._gen_arithmetic,
            'statistics': self._gen_statistics,
            'conversion': self._gen_conversion,
        }
        types = list(generators.keys())
        count = 0
        while True:
            if self.max_problems and count >= self.max_problems:
                return
            ptype         = rng.choice(types)
            introduce_err = rng.random() < self.ERROR_RATE
            yield generators[ptype](rng, introduce_err)
            count += 1


# =============================================================================
# CodeProblemsDataset
# =============================================================================

from datasets import load_dataset


class CodeProblemsDataset(IterableDataset):
    """HumanEval or MBPP streaming dataset.

    Each item yields:
      question    (str)   natural-language prompt in chat format
      test_code   (str)   fully assembled Python to execute for grading.
                          Contains: stub + canonical_solution + test assertions.
                          At training time, replace canonical_solution with model output.
      stub        (str)   function stub only (HumanEval); None for MBPP
      source      (str)   'humaneval' | 'mbpp'
    """

    def __init__(self, source: str = 'humaneval', split: str = 'test',
                 rank: int = 0, world_size: int = 1, shuffle: bool = True):
        assert source in ('humaneval', 'mbpp'), f"Unknown source: {source}"
        self.source     = source
        self.split      = split
        self.rank       = rank
        self.world_size = world_size
        self.shuffle    = shuffle
        self._data      = self._load()

    def _load(self) -> list[dict]:
        if self.source == 'humaneval':
            return list(load_dataset('openai_humaneval', split='test'))
        split = 'train' if self.split == 'train' else 'test'
        return list(load_dataset('mbpp', split=split))

    def _format_humaneval(self, row: dict) -> dict:
        question = (
            f'Complete the following Python function. '
            f'Write only the function body (no imports unless needed).\n\n'
            f'```python\n{row["prompt"]}\n```'
        )
        test_code = (
            row['prompt'] + row['canonical_solution'] +
            '\n' + row['test'] + f'\ncheck({row["entry_point"]})'
        )
        return dict(question=question, test_code=test_code,
                    stub=row['prompt'], source='humaneval')

    def _format_mbpp(self, row: dict) -> dict:
        question  = f'Write a Python function to solve the following task:\n\n{row["text"]}'
        test_code = row['code'] + '\n' + '\n'.join(row['test_list'])
        return dict(question=question, test_code=test_code,
                    stub=None, source='mbpp')

    def __iter__(self):
        indices = list(range(self.rank, len(self._data), self.world_size))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            row = self._data[i]
            if self.source == 'humaneval':
                yield self._format_humaneval(row)
            else:
                yield self._format_mbpp(row)


# =============================================================================
# Validation
# =============================================================================

if __name__ == '__main__':

    # --- Utility smoke tests ---
    assert extract_code_block('```python\nprint(42)\n```') == 'print(42)'
    assert extract_code_block('no code here') is None
    stdout, ok = execute_code('print(1 + 1)'); assert stdout == '2' and ok
    stdout, ok = execute_code('1/0');          assert not ok
    assert compute_reward('2.0',   '2.0',  True)  == 1.0
    assert compute_reward('',      '2.0',  False) == 0.0
    assert compute_reward('2.001', '2.0',  True)  == 1.0   # within tolerance
    assert compute_reward('3.0',   '2.0',  True)  == 0.5   # wrong answer
    print('Utility smoke tests passed.')

    # --- Inspect: one correct + one error sample per problem type ---
    ds   = CalculatorDataset(seed=99)
    seen = {t: {'correct': False, 'error': False}
            for t in ('arithmetic', 'statistics', 'conversion')}
    for item in ds:
        t   = item['problem_type']
        key = 'error' if item['has_error'] else 'correct'
        if not seen[t][key]:
            seen[t][key] = True
            print(f"\n{'='*60}")
            print(f"TYPE: {t.upper()}  |  has_error={item['has_error']}")
            print(f"Q: {item['question']}")
            print(f"\nRESPONSE:\n{item['response']}")
            print(f"\nEXPECTED: {item['expected_output']}")
        if all(seen[t][k] for t in seen for k in ('correct', 'error')):
            break

    # --- Verify: calculator_code always scores reward=1.0 ---
    ds    = CalculatorDataset(seed=42, max_problems=600)
    stats = collections.defaultdict(lambda: {'n': 0, 'perfect': 0, 'mean_r': 0.0})
    for item in ds:
        stdout, ok = execute_code(item['calculator_code'])
        r = compute_reward(stdout, item['expected_output'], ok)
        t = item['problem_type']
        stats[t]['n']       += 1
        stats[t]['perfect'] += int(r == 1.0)
        stats[t]['mean_r']  += r
    print(f"\n{'type':12s}  {'n':>6}  {'perfect':>8}  {'mean_r':>8}")
    print('-' * 42)
    for t, s in stats.items():
        mean_r = s['mean_r'] / s['n']
        print(f"{t:12s}  {s['n']:6d}  {s['perfect']:8d}  {mean_r:8.4f}")
        assert s['perfect'] == s['n'], f"calculator_code not perfect for {t}"
    print("All calculator_code samples score reward=1.0.")

    # --- Verify: error rate ~20% ---
    ds      = CalculatorDataset(seed=42, max_problems=1000)
    n_error = sum(1 for item in ds if item['has_error'])
    print(f"\nError rate: {n_error}/1000 = {n_error/10:.1f}%  (target ~20%)")
    assert 150 <= n_error <= 250, f"Error rate out of expected range: {n_error}/1000"
    print("Error rate check passed.")

    # --- Reward simulation: correct / wrong-no-code / wrong-reasoning+code ---
    def _mock_correct(item):
        return item['response']
    def _mock_wrong_no_code(item):
        return "I think the answer is 999999."
    def _mock_correct_via_code(item):
        wrong = str(round(float(item['expected_output']) * 1.2, 4))
        return (f"I calculated {wrong} in my head, but let me verify.\n\n"
                f"```python\n{item['calculator_code']}\n```\n\n"
                f"The answer is {item['expected_output']}.")

    print(f"\n{'type':12s}  {'scenario':28s}  {'reward':>8}")
    print('-' * 55)
    ds = CalculatorDataset(seed=5, max_problems=9)
    for item in ds:
        for name, fn in [
            ('correct response',    _mock_correct),
            ('wrong, no code',      _mock_wrong_no_code),
            ('wrong reason + code', _mock_correct_via_code),
        ]:
            r = compute_calculator_reward(fn(item), item['expected_output'])
            print(f"{item['problem_type']:12s}  {name:28s}  {r:8.2f}")
        print()
