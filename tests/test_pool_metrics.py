import unittest
from server.pool_metrics import engine_jobs,choose_backend


class PoolMetricsTests(unittest.TestCase):
    def test_load_includes_running_and_waiting(self):
        text='# HELP ignored\nvllm:num_requests_running{model_name="Qwen3-8B"} 1.0\nvllm:num_requests_waiting{model_name="Qwen3-8B"} 2\n'
        self.assertEqual(engine_jobs(text),3)
        with self.assertRaises(ValueError):engine_jobs('not a metrics response')

    def test_old_gateway_load_is_visible(self):
        self.assertEqual(choose_backend([1,1,0],[0,0,0],[0,0,0]),2)

    def test_local_reservation_prevents_burst_on_same_idle_engine(self):
        self.assertEqual(choose_backend([0,0],[1,0],[1024,0]),1)

    def test_healthy_engine_preferred_to_unknown(self):
        self.assertEqual(choose_backend([None,1],[0,0],[0,0]),1)
