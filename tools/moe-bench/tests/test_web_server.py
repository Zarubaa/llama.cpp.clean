#!/usr/bin/env python3
"""Integration tests for llama-moe-bench-server.

Run with either:

    MOE_BENCH_SERVER_BIN=/path/to/llama-moe-bench-server python3 -m unittest -v \
        tools/moe-bench/tests/test_web_server.py

or:

    python3 tools/moe-bench/tests/test_web_server.py \
        --server-bin /path/to/llama-moe-bench-server -v
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _consume_server_bin_arg() -> str | None:
    value = os.environ.get("MOE_BENCH_SERVER_BIN")
    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--server-bin":
            if index + 1 >= len(sys.argv):
                raise SystemExit("--server-bin requires a path")
            value = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            continue
        if arg.startswith("--server-bin="):
            value = arg.split("=", 1)[1]
            del sys.argv[index]
            continue
        index += 1
    return value


def _find_server_bin(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    candidates = (
        repo_root / "build-moe/bin/llama-moe-bench-server",
        repo_root / "build/bin/llama-moe-bench-server",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    found = shutil.which("llama-moe-bench-server")
    return Path(found).resolve() if found else None


SERVER_BIN = _find_server_bin(_consume_server_bin_arg())


FAKE_BENCH = r'''#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]
ledger = Path(os.environ["FAKE_BENCH_LEDGER"])


def option(*names):
    for index, arg in enumerate(args):
        for name in names:
            if arg == name and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def record(event, **values):
    row = {
        "event": event,
        "time": time.time(),
        "pid": os.getpid(),
        "argv": args,
    }
    row.update(values)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


prompt_arg = option("-p")
prompt_file = option("--prompt-file")
if prompt_arg is not None:
    prompt_mode = "text"
    prompt = prompt_arg
elif prompt_file is not None:
    prompt_mode = "file"
    prompt = Path(prompt_file).read_text(encoding="utf-8")
else:
    prompt_mode = "default"
    prompt = ""


def stop(signum, _frame):
    print("fake bench received signal %d" % signum, file=sys.stderr, flush=True)
    record("signal", signal=signum, prompt=prompt, prompt_mode=prompt_mode)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
record("start", prompt=prompt, prompt_mode=prompt_mode)
print("fake bench stdout: started", flush=True)
print("fake bench stderr: started", file=sys.stderr, flush=True)

if "[fail]" in prompt:
    print("fake bench requested failure", file=sys.stderr, flush=True)
    record("failed", prompt=prompt, prompt_mode=prompt_mode)
    raise SystemExit(7)

if "[parent-death]" in prompt:
    while True:
        time.sleep(1)

if "[slow]" in prompt:
    for step in range(30):
        print("fake bench progress %d" % step, flush=True)
        if step % 5 == 0:
            print("fake bench diagnostic %d" % step, file=sys.stderr, flush=True)
        time.sleep(0.04)

summary = """model: fake-moe-model
predictor: lru       cache: 2048 MB   ssd: /fake
n_prompt: 32  n_gen: 4  repeats: 1

phase     tokens   total_ms   per_token_ms   tok/s
prefill       32      120.5          3.77      265
decode         4       29.0          7.25      138

cache hit rate (prefill): 61.5%
cache hit rate (decode): 82.5%
SSD bytes read (decode): 105.87 GB  (avg 27102.72 MB/token)
TTFT: 120.5 ms
TPOT: 7.25 ms
total: 149.5 ms

Source bytes read (total): 128.35 GB
Process IO read_bytes: prepare=11 model_init=22 prefill=33 decode=44 total=110
VRAM peak (process approx): 7.50 GB / 24.00 GB
DRAM peak (process): 18.25 GB
Page cache resident pct: before=1.0 after_prepare=2.0 after_model_load=3.0 after_prefill=4.0 after_decode=42.5
"""

profile = "row_type,phase,token,cache_hits,source_bytes,physical_read_bytes\n"
if "[header-only-profile]" not in prompt:
    profile += "row,prefill,0,10,1024,64\nrow,decode,0,20,2048,46\n"
tokens = "repeat,step,token\n"
if "[header-only-tokens]" not in prompt:
    tokens += "0,0,101\n0,1,102\n"
artifacts = {
    option("--moe-profile-summary"): summary,
    option("--moe-profile-csv"): profile,
    option("--token-trace"): tokens,
}
for path_text, contents in artifacts.items():
    if path_text:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

print(summary, end="", flush=True)
print("fake bench stderr: complete", file=sys.stderr, flush=True)
record("end", prompt=prompt, prompt_mode=prompt_mode)
'''


class MoeBenchWebServerTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        if SERVER_BIN is None:
            raise unittest.SkipTest(
                "llama-moe-bench-server not found; set MOE_BENCH_SERVER_BIN "
                "or pass --server-bin"
            )
        if not SERVER_BIN.is_file() or not os.access(SERVER_BIN, os.X_OK):
            raise unittest.SkipTest("server binary is not executable: %s" % SERVER_BIN)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="moe-bench-server-test-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "runs"
        self.web_root = self.root / "web"
        self.model = self.root / "fake-model.moe.gguf"
        self.prompt_file = self.root / "default-prompt.txt"
        self.fake_bench = self.root / "fake-bench.py"
        self.ledger = self.root / "ledger.jsonl"

        self.output_root.mkdir()
        self.web_root.mkdir()
        (self.web_root / "index.html").write_text(
            "<!doctype html><title>test web root</title>", encoding="utf-8"
        )
        self.model.write_bytes(b"fake model")
        self.prompt_file.write_text("PROMPT_FROM_FILE", encoding="utf-8")
        self.fake_bench.write_text(textwrap.dedent(FAKE_BENCH), encoding="utf-8")
        self.fake_bench.chmod(0o755)
        self.ledger.touch()

        self.port = self._unused_port()
        self.base_url = "http://127.0.0.1:%d" % self.port
        command = [
            str(SERVER_BIN),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--bench", str(self.fake_bench),
            "--model", str(self.model),
            "--prompt-file", str(self.prompt_file),
            "--output-root", str(self.output_root),
            "--web-root", str(self.web_root),
            "--cuda-visible-devices", "0",
        ]
        env = os.environ.copy()
        env["FAKE_BENCH_LEDGER"] = str(self.ledger)
        self.server = subprocess.Popen(
            command,
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self._stop_server)
        self._wait_until_ready()

    def _stop_server(self) -> None:
        if self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=3)
        self.server.communicate(timeout=1)

    @staticmethod
    def _unused_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 5
        last_error = None
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                stdout, stderr = self.server.communicate(timeout=1)
                self.fail(
                    "server exited during startup with code %s\nstdout:\n%s\nstderr:\n%s"
                    % (self.server.returncode, stdout, stderr)
                )
            try:
                status, _body, _headers = self.http("GET", "/api/config")
                if status == 200:
                    return
            except (OSError, urllib.error.URLError) as error:
                last_error = error
            time.sleep(0.03)
        self.fail("server did not become ready: %r" % (last_error,))

    def http(self, method: str, path: str, payload=None, timeout: float = 5):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                body = json.loads(raw.decode("utf-8"))
            else:
                body = raw
            return response.status, body, response.headers

    @staticmethod
    def run_from(body):
        if not isinstance(body, dict) or not isinstance(body.get("run"), dict):
            raise AssertionError("expected a {run: {...}} response, got %r" % (body,))
        return body["run"]

    def base_request(self, **overrides):
        payload = {
            "prompt_mode": "default",
            "prompt_text": "",
            "pp": 32,
            "tg": 4,
            "repeat": 1,
            "ubatch": 16,
            "cache_vram_mb": 2048,
            "predictor": "lru",
            "host_cache": "off",
            "host_cache_preload": "none",
            "page_cache_policy": "natural",
            "context_size": 256,
            "gpu_layers": 99,
            "reset_cache_between_repeats": False,
            "warm_cache": False,
            "hot_start": False,
        }
        payload.update(overrides)
        return payload

    def create_run(self, **overrides):
        status, body, _headers = self.http(
            "POST", "/api/runs", self.base_request(**overrides)
        )
        self.assertIn(status, (200, 201, 202), body)
        run = self.run_from(body)
        self.assertIsInstance(run.get("id"), str)
        self.assertIn(run.get("status"), ("queued", "running", "completed"))
        return run

    def get_run(self, run_id: str):
        status, body, _headers = self.http("GET", "/api/runs/%s" % run_id)
        self.assertEqual(status, 200, body)
        return self.run_from(body)

    def wait_for_status(self, run_id: str, expected, timeout: float = 8):
        expected = {expected} if isinstance(expected, str) else set(expected)
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.get_run(run_id)
            if last.get("status") in expected:
                return last
            time.sleep(0.03)
        self.fail(
            "run %s did not reach %s; last response: %r"
            % (run_id, sorted(expected), last)
        )

    def cancel_run(self, run_id: str):
        status, body, _headers = self.http(
            "POST", "/api/runs/%s/cancel" % run_id, {}
        )
        self.assertIn(status, (200, 202), body)
        return self.run_from(body)

    def ledger_rows(self):
        rows = []
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
        return rows

    def wait_for_ledger_row(self, predicate, timeout: float = 3):
        deadline = time.monotonic() + timeout
        rows = []
        while time.monotonic() < deadline:
            rows = self.ledger_rows()
            for row in rows:
                if predicate(row):
                    return row
            time.sleep(0.02)
        self.fail("fake bench ledger did not receive expected row: %r" % rows)

    @staticmethod
    def process_is_running(pid: int) -> bool:
        try:
            stat = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            return False
        fields = stat.split()
        return len(fields) > 2 and fields[2] != "Z"

    @staticmethod
    def kill_process_group(process_group: int) -> None:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def assert_validation_error(self, payload, field: str) -> None:
        status, body, _headers = self.http("POST", "/api/runs", payload)
        self.assertEqual(status, 400, body)
        self.assertIn(field, json.dumps(body, sort_keys=True))

    def test_config_and_empty_run_list(self) -> None:
        status, config, _headers = self.http("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(
            set(config), {"service", "defaults", "limits", "options", "paths"}
        )
        self.assertTrue(config["service"])
        self.assertIsInstance(config["defaults"], dict)
        self.assertIsInstance(config["limits"], dict)
        self.assertIsInstance(config["options"], dict)
        self.assertIsInstance(config["paths"], dict)

        status, body, _headers = self.http("GET", "/api/runs")
        self.assertEqual(status, 200, body)
        self.assertEqual(body, {"runs": []})

    def test_completed_run_metrics_output_and_artifacts(self) -> None:
        created = self.create_run(prompt_mode="text", prompt_text="FAST_METRICS")
        run = self.wait_for_status(created["id"], "completed")

        for key in (
            "created_at", "started_at", "finished_at", "elapsed_ms", "request",
            "command", "metrics", "validation_errors", "artifacts", "raw_summary",
            "stdout_tail", "stderr_tail",
        ):
            self.assertIn(key, run)
        self.assertEqual(run["request"]["pp"], 32)
        self.assertEqual(run["request"]["predictor"], "lru")
        self.assertIn("fake bench stdout", run["stdout_tail"])
        self.assertIn("fake bench stderr", run["stderr_tail"])
        self.assertIn("TTFT: 120.5 ms", run["raw_summary"])

        metrics = run["metrics"]
        self.assertEqual(
            set(metrics),
            {
                "ttft_ms", "prefill_tok_s", "tpot_ms", "decode_tok_s",
                "total_ms", "prefill_hit_pct", "decode_hit_pct", "source_bytes",
                "ssd_decode_bytes", "physical_read_bytes", "vram_peak_gib",
                "dram_peak_gib", "page_cache_after_decode_pct",
            },
        )
        self.assertAlmostEqual(metrics["ttft_ms"], 120.5)
        self.assertAlmostEqual(metrics["prefill_tok_s"], 265.0)
        self.assertAlmostEqual(metrics["tpot_ms"], 7.25)
        self.assertAlmostEqual(metrics["decode_tok_s"], 138.0)
        self.assertAlmostEqual(metrics["total_ms"], 149.5)
        self.assertAlmostEqual(metrics["prefill_hit_pct"], 61.5)
        self.assertAlmostEqual(metrics["decode_hit_pct"], 82.5)
        self.assertGreater(metrics["source_bytes"], 128 * 1024 ** 3)
        self.assertGreater(metrics["ssd_decode_bytes"], 105 * 1024 ** 3)
        self.assertEqual(metrics["physical_read_bytes"], 110)
        self.assertAlmostEqual(metrics["vram_peak_gib"], 7.5)
        self.assertAlmostEqual(metrics["dram_peak_gib"], 18.25)
        self.assertAlmostEqual(metrics["page_cache_after_decode_pct"], 42.5)

        expected_artifacts = {
            "summary.txt", "profile.csv", "tokens.csv", "stdout.log",
            "stderr.log", "request.json", "result.json", "command.txt",
            "metadata.json",
        }
        artifact_names = {item["name"] for item in run["artifacts"]}
        self.assertEqual(artifact_names, expected_artifacts)
        for artifact in run["artifacts"]:
            self.assertGreater(artifact["size"], 0)
            self.assertEqual(
                artifact["url"],
                "/api/runs/%s/artifacts/%s" % (run["id"], artifact["name"]),
            )
        for name in sorted(expected_artifacts):
            status, contents, _headers = self.http(
                "GET", "/api/runs/%s/artifacts/%s" % (run["id"], name)
            )
            self.assertEqual(status, 200, name)
            self.assertTrue(contents, name)

    def test_two_runs_are_serialized(self) -> None:
        first = self.create_run(prompt_mode="text", prompt_text="[slow] FIRST")
        self.wait_for_status(first["id"], "running")
        second = self.create_run(prompt_mode="text", prompt_text="SECOND")
        queued = self.get_run(second["id"])
        self.assertEqual(queued["status"], "queued")
        self.assertGreaterEqual(queued["queue_position"], 1)

        self.wait_for_status(first["id"], "completed")
        self.wait_for_status(second["id"], "completed")
        rows = self.ledger_rows()
        first_start = next(
            row for row in rows if row["event"] == "start" and row["prompt"].endswith("FIRST")
        )
        first_end = next(
            row for row in rows if row["event"] == "end" and row["prompt"].endswith("FIRST")
        )
        second_start = next(
            row for row in rows if row["event"] == "start" and row["prompt"] == "SECOND"
        )
        self.assertLessEqual(first_start["time"], first_end["time"])
        self.assertLessEqual(first_end["time"], second_start["time"])

    def test_validation_rejects_enums_numbers_unknown_fields_and_host_combo(self) -> None:
        cases = (
            ({"predictor": "random"}, "predictor"),
            ({"host_cache": "disk"}, "host_cache"),
            ({"host_cache_preload": "some"}, "host_cache_preload"),
            ({"page_cache_policy": "flush-everything"}, "page_cache_policy"),
            ({"prompt_mode": "url"}, "prompt_mode"),
            ({"pp": 0}, "pp"),
            ({"tg": -1}, "tg"),
            ({"repeat": 0}, "repeat"),
            ({"ubatch": 0}, "ubatch"),
            ({"context_size": 0}, "context_size"),
            ({"unexpected_option": True}, "unexpected_option"),
            (
                {"host_cache": "off", "host_cache_preload": "all"},
                "host_cache_preload",
            ),
            (
                {"prompt_mode": "file", "prompt_text": "must be empty"},
                "prompt_text",
            ),
        )
        for changes, field in cases:
            with self.subTest(changes=changes):
                self.assert_validation_error(self.base_request(**changes), field)

        valid = self.create_run(
            prompt_mode="text",
            prompt_text="HOST_PRELOAD_VALID",
            host_cache="pageable",
            host_cache_preload="all",
        )
        self.wait_for_status(valid["id"], "completed")

    def test_default_file_and_text_prompts_use_expected_argv(self) -> None:
        shell_marker = self.root / "must-not-exist"
        literal_prompt = "TEXT_PROMPT; touch %s; $(false); 'quoted'" % shell_marker
        requests = (
            self.create_run(prompt_mode="default", prompt_text=""),
            self.create_run(prompt_mode="file", prompt_text=""),
            self.create_run(prompt_mode="text", prompt_text=literal_prompt),
        )
        for run in requests:
            self.wait_for_status(run["id"], "completed")

        starts = [row for row in self.ledger_rows() if row["event"] == "start"]
        self.assertEqual(len(starts), 3)
        default, file_prompt, text_prompt = starts
        self.assertEqual(default["prompt_mode"], "default")
        self.assertEqual(default["prompt"], "")
        self.assertEqual(file_prompt["prompt_mode"], "file")
        self.assertEqual(file_prompt["prompt"], "PROMPT_FROM_FILE")
        self.assertEqual(text_prompt["prompt_mode"], "text")
        self.assertEqual(text_prompt["prompt"], literal_prompt)
        self.assertFalse(shell_marker.exists())

    def test_cancel_queued_and_running_runs(self) -> None:
        running = self.create_run(prompt_mode="text", prompt_text="[slow] CANCEL_RUNNING")
        self.wait_for_status(running["id"], "running")
        self.wait_for_ledger_row(
            lambda row: row["event"] == "start" and row["prompt"].endswith("CANCEL_RUNNING")
        )
        queued = self.create_run(prompt_mode="text", prompt_text="CANCEL_QUEUED")
        self.assertEqual(self.get_run(queued["id"])["status"], "queued")

        self.cancel_run(queued["id"])
        cancelled_queued = self.wait_for_status(queued["id"], "cancelled")
        self.assertIsNone(cancelled_queued["started_at"])
        run_directory = self.output_root / queued["id"]
        metadata = json.loads(
            (run_directory / "metadata.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (run_directory / "result.json").read_text(encoding="utf-8")
        )["run"]
        self.assertIsNotNone(metadata["finished_at"])
        self.assertEqual(metadata["finished_at"], cancelled_queued["finished_at"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["finished_at"], cancelled_queued["finished_at"])

        self.cancel_run(running["id"])
        self.wait_for_status(running["id"], "cancelled")
        rows = self.ledger_rows()
        self.assertFalse(
            any(row["event"] == "start" and row["prompt"] == "CANCEL_QUEUED" for row in rows)
        )
        signal_rows = [
            row for row in rows
            if row["event"] == "signal" and row["prompt"].endswith("CANCEL_RUNNING")
        ]
        self.assertTrue(signal_rows, rows)
        self.assertIn(signal_rows[-1]["signal"], (signal.SIGINT, signal.SIGTERM))

    def test_server_sigkill_terminates_running_bench(self) -> None:
        created = self.create_run(
            prompt_mode="text", prompt_text="[parent-death] SERVER_SIGKILL"
        )
        self.wait_for_status(created["id"], "running")
        started = self.wait_for_ledger_row(
            lambda row: row["event"] == "start" and row["prompt"].endswith("SERVER_SIGKILL")
        )
        bench_pid = int(started["pid"])
        process_group = os.getpgid(bench_pid)
        self.assertNotEqual(process_group, os.getpgrp())
        self.addCleanup(self.kill_process_group, process_group)

        self.server.kill()
        self.server.wait(timeout=3)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and self.process_is_running(bench_pid):
            time.sleep(0.03)
        self.assertFalse(
            self.process_is_running(bench_pid),
            "fake bench survived server SIGKILL (pid=%d, pgid=%d)"
            % (bench_pid, process_group),
        )

    def test_failed_process_preserves_diagnostics(self) -> None:
        created = self.create_run(prompt_mode="text", prompt_text="[fail]")
        failed = self.wait_for_status(created["id"], "failed")
        self.assertIn("requested failure", failed["stderr_tail"])
        self.assertTrue(failed["finished_at"])
        self.assertGreaterEqual(failed["elapsed_ms"], 0)

    def test_header_only_profile_or_tokens_is_not_success(self) -> None:
        for marker, artifact in (
            ("[header-only-profile]", "profile.csv"),
            ("[header-only-tokens]", "tokens.csv"),
        ):
            with self.subTest(artifact=artifact):
                created = self.create_run(prompt_mode="text", prompt_text=marker)
                failed = self.wait_for_status(created["id"], "failed")
                self.assertIn(artifact, "\n".join(failed["validation_errors"]))

    def test_sse_streams_snapshot_and_live_log(self) -> None:
        created = self.create_run(prompt_mode="text", prompt_text="[slow] SSE")
        self.wait_for_status(created["id"], "running")
        request = urllib.request.Request(
            self.base_url + "/api/runs/%s/events" % created["id"],
            headers={"Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
            event_name = None
            data_lines = []
            events = []
            for _ in range(120):
                line = response.readline().decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())
                elif line == "" and event_name and data_lines:
                    event_data = json.loads("\n".join(data_lines))
                    events.append((event_name, event_data))
                    if event_name == "log":
                        break
                    event_name = None
                    data_lines = []
            self.assertTrue(events, "SSE returned no events")
            self.assertEqual(events[0][0], "snapshot")
            log_events = [data for name, data in events if name == "log"]
            self.assertTrue(log_events, events)
            self.assertIn(log_events[-1]["stream"], ("stdout", "stderr"))
            self.assertIsInstance(log_events[-1]["line"], str)

        self.cancel_run(created["id"])
        self.wait_for_status(created["id"], "cancelled")

    def test_artifact_allowlist_blocks_unknown_names_and_traversal(self) -> None:
        created = self.create_run(prompt_mode="text", prompt_text="ARTIFACT_SECURITY")
        run = self.wait_for_status(created["id"], "completed")
        base = "/api/runs/%s/artifacts/" % run["id"]

        for suffix in (
            "secret.txt",
            "%2e%2e%2fmetadata.json",
            "%2fetc%2fpasswd",
            "..%252fmetadata.json",
        ):
            with self.subTest(suffix=suffix):
                status, _body, _headers = self.http("GET", base + suffix)
                self.assertIn(status, (400, 404))

        secret = self.root / "outside-secret.txt"
        secret.write_text("must not be served", encoding="utf-8")
        summary = self.output_root / run["id"] / "summary.txt"
        summary.unlink()
        summary.symlink_to(secret)
        status, _body, _headers = self.http("GET", base + "summary.txt")
        self.assertEqual(status, 404)
        refreshed = self.get_run(run["id"])
        self.assertNotIn(
            "summary.txt", {artifact["name"] for artifact in refreshed["artifacts"]}
        )


if __name__ == "__main__":
    unittest.main()
