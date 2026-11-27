import hashlib
import unittest
from unittest.mock import Mock, patch, AsyncMock

from server.gpu_node_agent import started_command_hash


class StartupIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_real_argv_instead_of_recording_empty_hash(self):
        process=Mock(pid=123)
        process.poll.return_value=None
        command=['python','-m','vllm.entrypoints.openai.api_server']
        raw=b'python\0-m\0vllm.entrypoints.openai.api_server\0'
        with patch('pathlib.Path.read_bytes',side_effect=[b'',raw]),patch('server.gpu_node_agent.asyncio.sleep',new_callable=AsyncMock) as pause:
            result=await started_command_hash(process,command)
        self.assertEqual(result,hashlib.sha256(raw).hexdigest())
        pause.assert_awaited_once()

    async def test_exited_service_is_not_adopted(self):
        process=Mock(pid=123)
        process.poll.return_value=1
        with self.assertRaises(RuntimeError):
            await started_command_hash(process,['python'])


if __name__=='__main__':unittest.main()
