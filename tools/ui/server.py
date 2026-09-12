"""server.py — a local UI in front of the real engine.

Standard library only: no Flask, no dependency to install, nothing to keep in
sync. It does not reimplement anything — every answer comes from running the
picoforge binary, so what the page shows is what the engine does.

    make ui        # then open http://127.0.0.1:8000

Bind address is 127.0.0.1 on purpose. This runs an executable with text taken
from a request; it is a development tool for one machine, not a service.
"""

import csv
import json
import shlex
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "picoforge"
MODEL = ROOT / "models" / "Qwen3-0.6B"
PORT = 8000

# The engine prints its architecture and tokenizer summaries before doing any
# work. Generation starts after a bare "---" line and ends where the stats
# block begins. Splitting on those two markers is enough, and keeps the engine
# free of a special "machine readable" mode that would then need its own tests.
GEN_START = "\n---\n"
GEN_END = "\n\n--- "


def run(args: list[str], timeout: int = 600) -> str:
    proc = subprocess.run([str(ENGINE), str(MODEL), *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def tokenize(text: str) -> dict:
    out = run(["--encode", text])
    body = out.split("=== tokens ===\n", 1)[1].split("=== ", 1)[0]
    tokens = []
    for line in body.strip().splitlines():
        if not line.strip():
            continue
        tid, hexbytes = line.split("\t")
        raw = bytes.fromhex(hexbytes)
        try:
            shown, valid = raw.decode(), True
        except UnicodeDecodeError:
            # A token that is a fragment of a multi-byte character. Showing it
            # as bytes rather than as a replacement glyph is the honest thing:
            # this is what byte-level BPE actually does to an emoji.
            shown, valid = " ".join(f"{b:02X}" for b in raw), False
        tokens.append({"id": int(tid), "hex": hexbytes, "text": shown, "valid": valid})
    return {"tokens": tokens}


def bench() -> dict:
    path = ROOT / "bench" / "matmul_m5pro.csv"
    if not path.exists():
        return {"rows": []}
    rows = []
    for r in csv.DictReader(open(path)):
        for k in ("M", "N", "K"):
            r[k] = int(r[k])
        for k in ("median_s", "p10_s", "p90_s", "gflops", "gbps", "pct_of_307"):
            r[k] = float(r[k])
        rows.append(r)
    return {"rows": rows}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # quieter than the default
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str, stream: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _chunk(self, text: str) -> None:
        data = text.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)

        try:
            if url.path == "/":
                html = (Path(__file__).parent / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif url.path == "/api/tokenize":
                self._json(tokenize(q.get("text", [""])[0]))
            elif url.path == "/api/bench":
                self._json(bench())
            elif url.path == "/api/generate":
                self.stream_generate(q)
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:                       # noqa: BLE001
            try:
                self._json({"error": str(exc)})
            except Exception:                          # client already gone
                pass

    def stream_generate(self, q) -> None:
        prompt = q.get("prompt", [""])[0]
        device = q.get("device", ["gpu"])[0]
        kernel = q.get("kernel", ["2"])[0]
        max_new = str(int(q.get("max", ["120"])[0]))
        mode = "--gpu-chat" if device == "gpu" else "--chat"

        args = [str(ENGINE), str(MODEL), mode, prompt, max_new, "20260912"]
        if device == "gpu":
            args.append(kernel)

        self._send(200, b"", "text/event-stream; charset=utf-8", stream=True)
        self._chunk("event: cmd\ndata: " +
                    json.dumps({"cmd": " ".join(shlex.quote(a) for a in args)}) + "\n\n")

        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=ROOT, bufsize=0)
        buf, started, tail = "", False, ""
        try:
            while True:
                chunk = proc.stdout.read(64)
                if not chunk:
                    break
                buf += chunk.decode(errors="replace")

                if not started:
                    if GEN_START in buf:
                        started = True
                        buf = buf.split(GEN_START, 1)[1]
                    else:
                        continue

                if GEN_END in buf:
                    text, tail = buf.split(GEN_END, 1)
                    if text:
                        self._chunk("event: token\ndata: " +
                                    json.dumps({"t": text}) + "\n\n")
                    tail += proc.stdout.read().decode(errors="replace")
                    break

                # Hold back the last few characters: the end marker may be
                # split across two reads, and emitting half of it would print
                # the statistics into the chat window.
                if len(buf) > 8:
                    self._chunk("event: token\ndata: " +
                                json.dumps({"t": buf[:-8]}) + "\n\n")
                    buf = buf[-8:]
        finally:
            proc.wait()

        stats = {}
        for line in tail.splitlines():
            if line.startswith("prefill") or line.startswith("decode"):
                key = line.split(":", 1)[0].strip()
                nums = [float(x) for x in __import__("re").findall(r"[\d.]+", line)]
                stats[key] = {"seconds": nums[0], "tok_s": nums[1]}
                if key == "decode" and len(nums) > 2:
                    stats[key]["gbps"] = nums[2]
            if "prompt tokens," in line:
                nums = [int(x) for x in __import__("re").findall(r"\d+", line)]
                stats["prompt_tokens"], stats["generated"] = nums[0], nums[1]
        self._chunk("event: done\ndata: " + json.dumps(stats) + "\n\n")
        self._chunk("")


def main() -> None:
    if not ENGINE.exists():
        sys.exit(f"{ENGINE} not built — run make first")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"picoforge UI on http://127.0.0.1:{PORT}  (ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
