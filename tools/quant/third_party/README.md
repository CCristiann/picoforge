# Third-party code

`ds4_quants.c` and `quants.h` are copied unchanged from
[antirez/ds4](https://github.com/antirez/ds4) at commit
`b3f50180ec2c7324892be1f50bc83555ccb8d5e7` (`gguf-tools/quants.c`, `gguf-tools/quants.h`).
They derive from the GGML/llama.cpp quantizers. MIT License:

    Copyright (c) 2026 The ds4.c authors
    Copyright (c) 2023-2026 The ggml authors
    Copyright (c) 2023 DeepSeek

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Why a verbatim copy: qw4q's compat mode must reproduce the bytes of the
published Q2 file. clang emits `a*b+c` as a "may fuse" FMA and the backend
decides per call site after inlining, so a faithful rewrite of the Q2_K encoder
still differed in the last bit on 3 of 192 golden blocks (2026-11-22), and so
did a verbatim copy of only the Q2_K functions in another translation unit.
Only the whole, unchanged translation unit makes the same decisions. Do not
edit these files; picoforge's own encoders are in `../qw4q.c`.
