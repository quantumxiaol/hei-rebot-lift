"""Network failures must not replay robot tools or leak credentials."""

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

import httpx

from sim_agent.__main__ import main, run_deepseek
from sim_agent.api import DeepSeekAPI

HTTP_CLIENT = httpx.Client
TEST_KEY = "test-key-not-real"
OK = {"choices": [{"message": {"content": "OK"}}]}


@contextmanager
def mock_transport(handler):
    def create_client(**kwargs):
        return HTTP_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    with patch("sim_agent.api.httpx.Client", side_effect=create_client) as factory:
        yield factory


class APITest(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": TEST_KEY}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        sleeper = patch("sim_agent.api.time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        self.api = DeepSeekAPI()

    def test_connection_reset_then_success_uses_direct_connection_and_refreshes(self) -> None:
        replies = Mock(
            side_effect=[
                httpx.ConnectError("[Errno 54] Connection reset by peer"),
                httpx.Response(200, json=OK),
            ]
        )
        refresh = Mock()
        with mock_transport(replies) as factory, redirect_stdout(io.StringIO()):
            self.assertEqual(self.api.complete({}, before_attempt=refresh), OK)
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(all(not item.kwargs["trust_env"] for item in factory.call_args_list))
        self.assertEqual(refresh.call_count, 2)
        self.sleep.assert_called_once_with(1)

    def test_authentication_failure_is_not_retried_and_key_is_redacted(self) -> None:
        replies = Mock(return_value=httpx.Response(401, text=f"Bad credential: {TEST_KEY}"))
        with mock_transport(replies), self.assertRaises(RuntimeError) as error:
            self.api.complete({})
        self.assertEqual(replies.call_count, 1)
        self.assertIn("401", str(error.exception))
        self.assertNotIn(TEST_KEY, str(error.exception))
        self.sleep.assert_not_called()

    def test_service_error_retries_with_explicit_proxy_opt_in(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_TRUST_ENV": "true"}):
            api = DeepSeekAPI()
        replies = Mock(side_effect=[httpx.Response(503), httpx.Response(200, json=OK)])
        with mock_transport(replies) as factory, redirect_stdout(io.StringIO()):
            self.assertEqual(api.complete({}), OK)
        self.assertEqual(replies.call_count, 2)
        self.assertTrue(all(item.kwargs["trust_env"] for item in factory.call_args_list))

    def test_retries_are_bounded_and_errors_redact_key(self) -> None:
        replies = Mock(side_effect=httpx.ReadError(f"Connection reset: {TEST_KEY}"))
        output = io.StringIO()
        with mock_transport(replies), redirect_stdout(output), self.assertRaises(RuntimeError) as error:
            self.api.complete({})
        self.assertEqual(replies.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])
        self.assertNotIn(TEST_KEY, str(error.exception) + output.getvalue())
        self.assertIn("--check-api", str(error.exception))

    def test_simulator_reset_aborts_before_next_request(self) -> None:
        replies = Mock(side_effect=httpx.ConnectError("reset"))
        refresh = Mock(side_effect=[None, RuntimeError("Simulator reset")])
        with (
            mock_transport(replies),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "Simulator reset"),
        ):
            self.api.complete({}, before_attempt=refresh)
        self.assertEqual(replies.call_count, 1)

    def test_agent_retry_does_not_reexecute_previous_tool(self) -> None:
        def completion(name, arguments):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": name,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        arguments = {"arm": "right", "opening": "open"}
        replies = Mock(
            side_effect=[
                completion("set_gripper", arguments),
                httpx.ReadError("reset"),
                completion("done", {}),
            ]
        )
        controller = Mock()
        controller.execute.side_effect = [{"held_by": None}, {"success": True}]
        controller.success.return_value = True
        with (
            tempfile.TemporaryDirectory() as directory,
            mock_transport(replies),
            redirect_stdout(io.StringIO()),
            patch(
                "sim_agent.__main__.observation_message",
                return_value={"role": "user", "content": "fresh state"},
            ),
        ):
            trace = Path(directory) / "trace.jsonl"
            self.assertTrue(run_deepseek(controller, 2, trace, True, "Pick up red cube"))
            self.assertEqual(len(trace.read_text().splitlines()), 2)
        self.assertEqual(replies.call_count, 3)
        self.assertEqual(
            controller.execute.call_args_list, [call("set_gripper", arguments), call("done", {})]
        )

    def test_check_api_never_connects_to_simulator(self) -> None:
        with (
            mock_transport(lambda request: httpx.Response(200, json=OK)),
            patch("sys.argv", ["agent", "--check-api"]),
            patch("sim_agent.__main__.load_dotenv"),
            patch("sim_agent.__main__.SimulationClient") as robot,
            redirect_stdout(io.StringIO()),
        ):
            main()
        robot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
