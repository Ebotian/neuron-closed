"""CPU-only tests for the terminal progress line (no torch, no model)."""
import io
import unittest

from progress import Progress, human_duration


class FakeClock:
    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


class ProgressTests(unittest.TestCase):
    def test_human_duration_bands(self):
        self.assertEqual(human_duration(0), '0s')
        self.assertEqual(human_duration(45.4), '45s')
        self.assertEqual(human_duration(89), '89s')
        self.assertEqual(human_duration(90), '1.5 min')
        self.assertEqual(human_duration(5399), '90.0 min')
        self.assertEqual(human_duration(5400), '1.5 h')
        self.assertEqual(human_duration(7200), '2.0 h')
        self.assertEqual(human_duration(-3), '0s')

    def test_counter_elapsed_and_eta(self):
        out = io.StringIO()
        p = Progress(4, stream=out, clock=FakeClock([0.0, 1.0, 3.0, 6.0, 10.0]))
        for i in range(4):
            p.step(f'item{i}')
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith('[1/4] 1s elapsed, eta 3s  item0'), lines[0])
        self.assertTrue(lines[1].startswith('[2/4] 3s elapsed, eta 3s'), lines[1])
        self.assertTrue(lines[2].startswith('[3/4] 6s elapsed, eta 2s'), lines[2])
        # The last item has nothing left: eta must read zero, never negative.
        self.assertTrue(lines[3].startswith('[4/4] 10s elapsed, eta 0s'), lines[3])

    def test_progress_is_monotone(self):
        out = io.StringIO()
        p = Progress(3, stream=out, clock=FakeClock([0.0, 2.0, 5.0, 9.0]))
        for _ in range(3):
            p.step()
        self.assertEqual([p.done], [3])
        self.assertEqual([line.split(']')[0] + ']' for line in out.getvalue().splitlines()],
                         ['[1/3]', '[2/3]', '[3/3]'])

    def test_total_must_be_positive(self):
        with self.assertRaises(ValueError):
            Progress(0)
        with self.assertRaises(ValueError):
            Progress(-2)

    def test_disabled_still_counts_without_output(self):
        out = io.StringIO()
        p = Progress(2, stream=out, enabled=False)
        p.step()
        p.step()
        self.assertEqual(out.getvalue(), '')
        self.assertEqual(p.done, 2)


if __name__ == '__main__':
    unittest.main()
