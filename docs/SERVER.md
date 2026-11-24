# pf-server

`pf-server` puts the picoforge engine behind an HTTP API, so coding agents and chat clients that
speak the OpenAI or Anthropic formats (opencode, Claude Code, Pi, scripts) can use the local model.

```sh
make pf-server
./pf-server -m q2                       # http://127.0.0.1:8080/v1, context 100000 tokens
./pf-server -m code5                    # the code-tuned file (README: Code-tuned weights)
```

The model loads once (a minute or so for Q2); the server prints `serving <id> on ...` when ready.
It serves one request at a time: one model, one graph. Others wait in the connection queue, so set
generous client timeouts. Ctrl+C stops it and keeps the current conversation for the next start
(see [checkpoints](#prompt-cache-and-checkpoints)).

It listens on 127.0.0.1 only. `--host 0.0.0.0` opens it to the network: then also set
`--api-key KEY`, and do not expose it to the Internet without TLS and authentication in front.

## Endpoints

| Endpoint | Format |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming (SSE) or not, tools |
| `POST /v1/messages` | Anthropic Messages, streaming or not, tools, thinking blocks |
| `POST /v1/messages/count_tokens` | Anthropic token count of a prompt |
| `GET /v1/models`, `GET /v1/models/{id}` | The loaded model, with its context length |
| `GET /health` | `{"status":"ok"}` once the model is loaded |

Any model name is accepted: the GGUF given with `-m` is the model. The id the server reports is the
file name (`Qwen3.8-Flash-Next-Q2`), or `--alias NAME`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "pf", "stream": true,
  "messages": [{"role": "user", "content": "Explain a KV cache in two sentences."}]}'
```

**Reasoning.** Qwen3.8 thinks before it answers. The reasoning comes back separately: as
`reasoning_content` (OpenAI format) or `thinking` blocks (Anthropic format), never mixed into the
text. It is on by default at effort `medium`. A request can change it with `reasoning_effort`
(`none`, `low`, `medium`, `high`), `chat_template_kwargs.enable_thinking`, or Anthropic's
`thinking: {"type": "disabled"}`; the server default with `--no-think` / `--effort`.

**Tools.** Tool definitions go into the system prompt in the model's own format; the model's tool
calls are parsed and returned as `tool_calls` / `tool_use` blocks, the arguments typed by the tool's
JSON schema (an `integer` parameter comes back as a number). The server never runs tools: the client
does.

**Sampling.** Defaults are temperature 0.6, top-k 20, top-p 0.95 (the model's recommended
settings); `temperature`, `top_p`, `top_k`, `min_p` in a request override them, `temperature: 0`
decodes greedily. A reply is limited by `max_tokens` (or `max_completion_tokens`), else by `-n`
(32768), and always by the room left in the context. Speculative decoding (MTP) is on and never
changes the output; `--no-mtp` turns it off.

**Not supported:** images (the model reads text only: the request is refused), `stop` sequences
(ignored), the OpenAI Responses API (`/v1/responses`, used by Codex CLI), `n > 1`.

**Errors.** A prompt longer than the context gets HTTP 400 with code `context_length_exceeded` and
the usual "maximum context length is N tokens" message, which agents take as the signal to compact.

## Prompt cache and checkpoints

The APIs are stateless: an agent resends the whole conversation at every step, and prefilling 30K
tokens takes over a minute on this machine. So the server avoids prefilling what it has already seen:

- **The live graph.** The tokens of the last request (prompt and reply) stay in the graph; a request
  that extends them prefills only the new part. The usage reports it as `cached_tokens`
  (`cache_read_input_tokens` in the Anthropic format).
- **The model's own reply.** Clients send the reply back in their own way: without the reasoning,
  with the tool arguments re-serialized. Rendered again, that is not what the model generated, and a
  plain token match would miss from there on. When the conversation is a remembered request, our
  reply and new messages, and the client's copy says the same thing as ours, the server keeps the
  exact tokens the model sampled, reasoning included, and renders only the new messages. The last 8
  exchanges are remembered, so a request from another conversation in between does not lose them.
- **Checkpoints on disk.** The recurrent layers of the model cannot be rewound, so a graph that went
  past the common prefix would start from zero. The server saves the graph state instead: the live
  conversation before an unrelated request replaces it (a title generator, a second session), the
  system prompt and tools of a long new prompt (shared by every session of the same agent), the end of
  a prompt that needed a long prefill (a retry is then immediate), and the conversation at shutdown.
  A request starts from the longest saved prefix of its tokens.

Checkpoints live in `~/.picoforge/kv-cache` (`--cache-dir`, or `$PICOFORGE_KV_CACHE`), up to
16 GiB (`--cache-mb`); the least recently used go first. Prefixes under 4096 tokens are not saved
(`--cache-min`): they are faster to prefill than to keep. A checkpoint grows with its conversation,
not with the context size. `--no-cache` turns them off. The files contain prompt text and model state:
treat the directory as private. It is disposable; stop the server before deleting it.

Every request logs one line: prompt tokens and how many were reused, prefill and decode speed, why
it stopped.

## Options

| Option | Default | |
|---|---|---|
| `-m MODEL` | the only model | a path or a short name (`./pf models`) |
| `-c N` | 100000 | context in tokens; set the client's limit to the same value |
| `--host`, `--port` | 127.0.0.1, 8080 | |
| `--api-key KEY` | none | require `Authorization: Bearer KEY` or `x-api-key: KEY` |
| `--alias NAME` | the file name | the model id clients see |
| `--no-think`, `--effort E` | on, medium | reasoning defaults (`low`, `medium`, `high`) |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | 0.6, 0.95, 20, 0 | sampling defaults |
| `-n N` | 32768 | reply limit when the request sets none |
| `--no-mtp` | MTP on | |
| `--power N` | 100 | GPU duty cycle in percent, for a cooler machine; output unchanged |
| `--cors` | off | browser cross-origin headers (no access control) |
| `--cache-dir`, `--cache-mb`, `--cache-min`, `--no-cache` | see above | checkpoints |

Memory: the Q2 model keeps ~42 GiB resident and the graph grows with `-c`. On a 64 GB Mac, close
other large processes, and run one picoforge (or other GPU-heavy) process at a time.

## Client setup

Start the server first, with the context you want the client to use. In every example
`pf-local` is a placeholder key: the server ignores it unless you set `--api-key`. Replace the model
id with the one `GET /v1/models` reports (or start with `--alias pf` and use `pf`).

### opencode

Merge into `~/.config/opencode/opencode.json` (or `opencode.json` in a project). For opencode 2.x:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "picoforge/pf",
  "providers": {
    "picoforge": {
      "name": "picoforge",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "pf-local" },
      "models": {
        "pf": {
          "modelID": "pf",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "compatibility": { "reasoningField": "reasoning_content" },
          "limit": { "context": 100000, "output": 32768 }
        }
      }
    }
  }
}
```

with `./pf-server -m q2 --alias pf`. `reasoningField` makes opencode show the reasoning and send it
back with the conversation; the server's cache works either way.

opencode 1.x has an older schema:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "picoforge": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "picoforge",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "pf-local" },
      "models": { "pf": { "name": "picoforge Qwen3.8", "limit": { "context": 100000, "output": 32768 } } }
    }
  }
}
```

Select `picoforge/pf` with `/models`. opencode also asks the model for a session title in a separate
request: with checkpoints on, that costs the main conversation one checkpoint save and load, not a
new prefill. To spare even that, point opencode's `small_model` at another provider.

### Claude Code

Claude Code speaks the Anthropic format. A wrapper script (give it another name than `claude`):

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_AUTH_TOKEN="pf-local"
export ANTHROPIC_MODEL="pf"
export ANTHROPIC_DEFAULT_OPUS_MODEL="pf" ANTHROPIC_DEFAULT_SONNET_MODEL="pf" ANTHROPIC_DEFAULT_HAIKU_MODEL="pf"
export CLAUDE_CODE_SUBAGENT_MODEL="pf"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS=3000000
exec claude "$@"
```

Its system prompt and tool definitions are long (tens of thousands of tokens): start the server with
`-c 100000` or more, and expect the first request to prefill for a few minutes. Later requests, and
new sessions after the system block is checkpointed, start from the cache.

### Pi

Add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "picoforge": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "pf-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [{
        "id": "pf", "name": "picoforge Qwen3.8", "reasoning": true, "input": ["text"],
        "contextWindow": 100000, "maxTokens": 32768,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

### Any OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="pf-local")
r = client.chat.completions.create(model="pf", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

## Tests

`make test-pf-server-model` runs both formats, tool calls, the exact replay and the checkpoints on
the synthetic checkpoint (under the GPU lock). Canned replies (`--script`) stand in for the model
where the text matters; real generations check that a restored state decodes the same tokens.
