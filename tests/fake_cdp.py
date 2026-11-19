"""A fake Chrome DevTools endpoint for pf-agent's web tests: HTTP /json/* and one
WebSocket per tab, answering Page.* and Runtime.evaluate with canned values.
Before each answer it sends an event (to be skipped), once a ping (to be
answered), and splits the answers into two frames (continuation)."""
import base64, hashlib, json, socket, struct, threading


class FakeChrome:
    def __init__(self, search_value, page_value):
        self.search_value, self.page_value = search_value, page_value
        self.requests, self.navigated, self.pongs = [], [], 0
        self.sock = socket.socket(); self.sock.bind(('127.0.0.1', 0)); self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        data = b''
        while b'\r\n\r\n' not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk
        head = data.split(b'\r\n\r\n')[0].decode()
        method, path = head.split(' ')[:2]
        self.requests.append((method, path))
        if 'Upgrade: websocket' in head:
            return self.websocket(conn, head)
        ws = f'ws://127.0.0.1:{self.port}/devtools/page/T1'
        body = {'/json/version': {'Browser': 'Fake/1', 'webSocketDebuggerUrl': f'ws://127.0.0.1:{self.port}/devtools/browser/x'},
                '/json/new?about:blank': {'id': 'T1', 'type': 'page', 'webSocketDebuggerUrl': ws}}.get(path, 'Target is closing')
        if path == '/json/new?about:blank' and method != 'PUT':
            conn.sendall(b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n'); conn.close(); return
        payload = json.dumps(body).encode()
        conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(payload) + payload)
        conn.close()

    @staticmethod
    def frame(op, payload, fin=True):
        n = len(payload)
        head = bytes([(0x80 if fin else 0) | op]) + (bytes([n]) if n < 126 else struct.pack('>BH', 126, n) if n < 65536 else struct.pack('>BQ', 127, n))
        return head + payload

    def read_frame(self, conn, buf):
        def need(k):
            nonlocal buf
            while len(buf) < k:
                chunk = conn.recv(65536)
                if not chunk:
                    raise EOFError
                buf += chunk
        need(2); op, n = buf[0] & 15, buf[1] & 127; h = 2
        assert buf[1] & 0x80, 'client frames must be masked'
        if n == 126: need(4); n = struct.unpack('>H', buf[2:4])[0]; h = 4
        elif n == 127: need(10); n = struct.unpack('>Q', buf[2:10])[0]; h = 10
        need(h + 4 + n); mask = buf[h:h+4]
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(buf[h+4:h+4+n]))
        return op, payload, buf[h+4+n:]

    def websocket(self, conn, head):
        key = next(l.split(': ', 1)[1] for l in head.split('\r\n') if l.lower().startswith('sec-websocket-key'))
        accept = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        conn.sendall(f'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n'.encode())
        buf, ready, pinged = b'', 0, False
        try:
            while True:
                op, payload, buf = self.read_frame(conn, buf)
                if op == 10:
                    self.pongs += 1; continue
                if op != 1:
                    continue
                msg = json.loads(payload); result = {}
                if msg['method'] == 'Page.navigate':
                    self.navigated.append(msg['params']['url']); result = {'frameId': 'F'}
                elif msg['method'] == 'Runtime.evaluate':
                    expr = msg['params']['expression']
                    if expr == 'document.readyState':
                        ready += 1; value = 'loading' if ready == 1 else 'complete'
                    else:
                        value = self.search_value if 'a h3' in expr else self.page_value
                    result = {'result': {'type': 'string', 'value': value}}
                conn.sendall(self.frame(1, b'{"method":"Page.frameNavigated","params":{"id":%d}}' % msg['id']))
                if not pinged:
                    conn.sendall(self.frame(9, b'hi')); pinged = True
                answer = json.dumps({'id': msg['id'], 'result': result}, separators=(',', ':')).encode()   # compact, as Chrome
                conn.sendall(self.frame(1, answer[:5], fin=False) + self.frame(0, answer[5:]))
        except (EOFError, OSError):
            pass
