# Recorded chat session

Command: `picoforge -m <Flash-Next Q2 GGUF> -c 4096 -n 700` (stdin piped, so no prompt is echoed; each input is shown as `picoforge>`).

```
picoforge> /help
Commands:
  /help          Show this help.
  /read FILE     Send a text file as your message.
  /ctx N         New context size (clears the conversation).
  /reset         Start a new conversation.
  /think, /nothink  Reasoning on or off for the next turns.
  /quit, /exit   Leave (Ctrl+D too).
  Ctrl+C         Stop the reply and come back here.
Anything else is your next chat message.
```

```
picoforge> Hi! In two sentences, what is a mixture-of-experts language model?
A mixture-of-experts (MoE) language model is an architecture that uses a gating network to dynamically route each input token to a small subset of specialized expert networks. This approach allows the model to scale to massive parameter counts while maintaining low computational costs during inference, as only the relevant experts are activated for any given task.
```

`picoforge: prefill 32.03 t/s, generation 24.28 t/s, context 94/4096`

```
picoforge> /read bench/qwen38/terminal_20261101/note.txt
picoforge is a lightweight C and Metal inference engine for Apple Silicon that executes Qwen3.8 Flash-Next Q2 models via the terminal while validating its fast kernels against a reference oracle.
```

`picoforge: prefill 52.31 t/s, generation 24.17 t/s, context 198/4096`

```
picoforge> Write a long poem about the sea, at least forty lines.   ⟵ Ctrl+C sent mid-reply
The horizon stretches, thin and sharp as glass,
Where sky and water meet in silent truce,
A line that moves no matter how you pass,
Or chase the ghost of light’s elusive juice.
The tide pulls back, revealing wetter sand,
With shells like broken teeth from ancient jaws,
And footprints fading in the shifting land,
Erased by foam and nature’s gentle laws.
The waves arrive in rhythmic, rolling breath,
```

`picoforge: prefill 52.77 t/s, generation 23.90 t/s, context 321/4096 [interrupted]`

```
picoforge> In one line: what were you writing before I stopped you?
I was writing a long poem about the sea.
```

`picoforge: prefill 52.78 t/s, generation 23.40 t/s, context 359/4096`

```
picoforge> /think
Thinking on.
```

```
picoforge> What is 17 * 23? Reply with the number only.
The user wants me to calculate 17 * 23 and reply with just the number.

17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391


391
```

`picoforge: prefill 34.45 t/s, generation 24.09 t/s, context 449/4096`

```
picoforge> /nothink
Thinking off.
```

```
picoforge> /ctx 2048
Context 2048 tokens, graph buffers 1096.6 MiB. New conversation.
```

```
picoforge> Do you remember what we talked about?
No, I don't have a memory of past conversations. Each session with me starts fresh, so I don't retain information from previous chats for privacy and security reasons.

If there's something specific you'd like to continue discussing or if you have a new question, feel free to share the details, and I'll be happy to help!
```

`picoforge: prefill 28.59 t/s, generation 24.35 t/s, context 90/2048`

```
picoforge> /quit

```

