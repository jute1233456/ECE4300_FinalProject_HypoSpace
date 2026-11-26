import unittest
from server.lr_schedules import warmup_reciprocal


class ScheduleTests(unittest.TestCase):
    def test_endpoints_shape_and_warmup_for_all_groups(self):
        for total in (20, 40, 80):
            warm = total // 10
            curve = [warmup_reciprocal(s, total, warm) for s in range(total + 1)]
            self.assertEqual(curve[0], 0)
            self.assertEqual(curve[warm], 1)
            self.assertEqual(curve[-1], 0)
            self.assertTrue(all(a < b for a, b in zip(curve[:warm], curve[1:warm + 1])))
            drops = [curve[s] - curve[s + 1] for s in range(warm, total)]
            self.assertTrue(all(x > 0 for x in drops))
            self.assertTrue(all(a > b for a, b in zip(drops, drops[1:])))

    def test_outside_range_and_bad_config(self):
        self.assertEqual(warmup_reciprocal(100, 40, 4), 0)
        self.assertEqual(warmup_reciprocal(-1, 40, 4), 0)
        with self.assertRaises(ValueError):
            warmup_reciprocal(0, 4, 4)


if __name__ == "__main__":
    unittest.main()
