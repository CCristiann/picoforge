/* main.c — picoforge entry point.
 *
 * Usage: ./picoforge [model_dir]
 */
#include "picoforge.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* The config and the checkpoint are two files that can disagree. Nothing
 * downstream re-checks them, so every dimension the forward pass will index
 * with gets confronted with the shape actually on disk, once, here. A
 * mismatch is a wrong model directory or a truncated download — both of
 * which would otherwise show up as plausible-looking wrong logits. */
static void check_shape(const SafeTensors *st, const char *name, long d0, long d1) {
    const Tensor *t = st_find(st, name);
    int want_ndim = (d1 > 0) ? 2 : 1;
    if (t->ndim != want_ndim || t->shape[0] != d0 || (d1 > 0 && t->shape[1] != d1))
        die("\"%s\": checkpoint has shape [%ld,%ld] (ndim %d), config implies [%ld,%ld]",
            name, t->shape[0], t->ndim > 1 ? t->shape[1] : 0L, t->ndim, d0, d1);
}

/* First few weights of a tensor, printed with enough digits to round-trip
 * exactly (%.9g). These are diffed against NumPy: bf16 -> fp32 is a shift,
 * so agreement must be EXACT, not approximate. Any difference at all means
 * the offsets are wrong, not that the arithmetic drifted. */
static void fingerprint(const SafeTensors *st, const char *name, int n) {
    const Tensor *t = st_find(st, name);
    if (t->dtype != DT_BF16) die("fingerprint expects a bf16 tensor");
    const uint16_t *src = t->data;
    printf("%-44s", name);
    for (int i = 0; i < n; i++) printf(" %.9g", (double)bf16_to_f32(src[i]));
    printf("\n");
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "models/Qwen3-0.6B";

    Qwen3Config cfg;
    config_load(model_dir, &cfg);
    config_print(&cfg);

    SafeTensors st;
    st_open(model_dir, &st);
    st_summary(&st);

    int q_dim  = cfg.num_attention_heads  * cfg.head_dim;
    int kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
    check_shape(&st, "model.norm.weight", cfg.hidden_size, 0);
    check_shape(&st, "model.layers.0.self_attn.q_norm.weight", cfg.head_dim, 0);
    /* Every projection, bf16 or quantised, in every layer: binding validates
     * each shape against the config (model.c, bind_linear) and dies on the
     * first disagreement. Binding is 311 lookups and no copies, so it is
     * cheap enough to do here just for the check. */
    Weights probe_w;
    weights_bind(&st, &cfg, &probe_w);
    printf("shape cross-check     : all %d layers' dimensions agree with config (%s)\n",
           cfg.num_hidden_layers,
           probe_w.layers[0].q_proj.w ? "bf16" :
           probe_w.layers[0].q_proj.q.bits == 8 ? "8-bit codes" : "4-bit codes");
    if (!probe_w.layers[0].q_proj.w)
        printf("quantisation          : %d-bit, blocks of %d (q_proj), embedding %s\n",
               probe_w.layers[0].q_proj.q.bits, probe_w.layers[0].q_proj.q.group,
               probe_w.embed.w ? "bf16" : "quantised");
    weights_free(&probe_w);
    (void)q_dim; (void)kv_dim;

    printf("\n=== fingerprint (first 4 values, fp32) ===\n");
    fingerprint(&st, "model.layers.0.input_layernorm.weight", 4);
    if (st_try(&st, "model.layers.27.mlp.down_proj.weight"))
        fingerprint(&st, "model.layers.27.mlp.down_proj.weight", 4);

    Tokenizer tok;
    tokenizer_load(model_dir, &tok);
    tokenizer_summary(&tok);

    /* --dump-vocab / --dump-merges FILE: write the whole loaded tables out so
     * tests/test_tokenizer.py can check every entry against Python, rather
     * than spot-checking a handful and hoping the rest followed. */
    if (argc > 3 && strcmp(argv[2], "--dump-vocab") == 0) {
        FILE *f = fopen(argv[3], "wb");
        if (!f) die("cannot write %s", argv[3]);
        for (int id = 0; id < tok.n_tokens; id++) {
            fprintf(f, "%d\t", id);
            for (int i = tok.offset[id]; i < tok.offset[id + 1]; i++)
                fprintf(f, "%02x", tok.blob[i]);
            fputc('\n', f);
        }
        fclose(f);
        printf("dumped %d tokens -> %s\n", tok.n_tokens, argv[3]);
    }

    if (argc > 3 && strcmp(argv[2], "--dump-merges") == 0) {
        FILE *f = fopen(argv[3], "wb");
        if (!f) die("cannot write %s", argv[3]);
        /* Walk the hash table, not the file: this prints what we actually
         * stored, so a bug in insertion or probing shows up instead of being
         * papered over by re-reading merges.txt. */
        for (unsigned i = 0; i <= tok.merge_mask; i++) {
            if (tok.merge_rank[i] < 0) continue;
            fprintf(f, "%d\t%u\t%u\n", tok.merge_rank[i],
                    (unsigned)(tok.merge_key[i] >> 32),
                    (unsigned)(tok.merge_key[i] & 0xFFFFFFFFu));
        }
        fclose(f);
        printf("dumped %d merge rules -> %s\n", tok.n_merges, argv[3]);
    }

    /* --bench OUT.csv : the full sweep, per docs/BENCHMARKS.md. */
    if (argc > 3 && strcmp(argv[2], "--bench") == 0) {
        MetalContext *mtl = metal_init("picoforge.metallib");
        metal_info(mtl);
        bench_matmul(mtl, argv[3]);
        metal_shutdown(mtl);
    }

    /* --bench-e2e OUT.csv : prefill, decode and 32-token verify of this
     * model, one row each appended to OUT. */
    if (argc > 3 && strcmp(argv[2], "--bench-e2e") == 0)
        bench_e2e(model_dir, argv[3]);

    /* --bench-dispatch OUT.csv : one matmul repeated 1..1024 times inside a
     * single command buffer, to separate the cost of a call from its bytes. */
    if (argc > 3 && strcmp(argv[2], "--bench-dispatch") == 0) {
        MetalContext *mtl = metal_init("picoforge.metallib");
        metal_info(mtl);
        bench_dispatch(mtl, argv[3]);
        metal_shutdown(mtl);
    }

    /* --profile OUT.csv : one pass's GPU time by op group, from encoder
     * timestamps, at decode and verify sizes. */
    if (argc > 3 && strcmp(argv[2], "--profile") == 0)
        bench_profile(model_dir, argv[3]);

    /* --bench-moe OUT.csv : one MoE block's GPU time against tokens verified
     * and distinct experts touched, routing forced. */
    if (argc > 3 && strcmp(argv[2], "--bench-moe") == 0)
        bench_moe(model_dir, argv[3]);

    /* --bench-quant OUT.csv : the quantised kernels, same protocol. */
    if (argc > 3 && strcmp(argv[2], "--bench-quant") == 0) {
        MetalContext *mtl = metal_init("picoforge.metallib");
        metal_info(mtl);
        bench_qmatmul(mtl, argv[3]);
        metal_shutdown(mtl);
    }

    /* --metal-check : every GPU kernel against the CPU matmul, on the shapes
     * the engine actually issues.
     *
     * The CPU path is the oracle here (CLAUDE.md: chain of oracles). It was
     * itself verified against NumPy, so a kernel that agrees with it is
     * transitively verified against transformers. Shapes are the real ones,
     * not round numbers: a kernel that works on 256x256x256 and not on
     * 1x1024x151936 has not been tested on this engine's problem. */
    if (argc > 2 && strcmp(argv[2], "--metal-check") == 0) {
        MetalContext *mtl = metal_init("picoforge.metallib");
        metal_info(mtl);
        printf("dispatch floor        : %.1f us (an empty kernel, best of 45)\n",
               metal_dispatch_floor(mtl) * 1e6);

        struct { int M, N, K; const char *what; } cases[] = {
            {  1, 1024, 1024, "decode  q_proj  (M=1: the bandwidth-bound case)" },
            {  1, 3072, 1024, "decode  gate_proj" },
            { 40, 2048, 1024, "prefill q_proj, 40 tokens" },
            {128, 3072, 1024, "prefill gate_proj, 128 tokens" },
            {  7,  333,  517, "awkward sizes: nothing divides anything" },
        };

        printf("\n=== GPU kernels vs the CPU oracle ===\n");
        printf("%-46s %10s %10s %9s %9s\n", "case", "max |rel|", "GPU ms", "GFLOP/s", "GB/s");

        bool all_ok = true;
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            int M = cases[c].M, N = cases[c].N, K = cases[c].K;
            float *A = malloc((size_t)M * (size_t)K * sizeof *A);
            uint16_t *B = malloc((size_t)N * (size_t)K * sizeof *B);
            float *gpu = malloc((size_t)M * (size_t)N * sizeof *gpu);
            float *ref = malloc((size_t)M * (size_t)N * sizeof *ref);
            if (!A || !B || !gpu || !ref) die("out of memory for the %dx%dx%d case", M, N, K);

            uint32_t rng = 12345u;
            for (int i = 0; i < M * K; i++) {
                rng = rng * 1664525u + 1013904223u;
                A[i] = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
            }
            for (int i = 0; i < N * K; i++) {
                rng = rng * 1664525u + 1013904223u;
                float v = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
                uint32_t bits;
                memcpy(&bits, &v, sizeof bits);
                B[i] = (uint16_t)(bits >> 16);
            }

            for (int m = 0; m < M; m++)
                matmul(ref + (size_t)m * (size_t)N, A + (size_t)m * (size_t)K, B, K, N);

            for (int which = 0; which < 3; which++) {
                if (!metal_has_kernel(mtl, which)) continue;
                double secs = metal_matmul(mtl, which, gpu, A, B, M, N, K);

                double worst = 0.0;
                for (int i = 0; i < M * N; i++) {
                    double d = fabs((double)gpu[i] - (double)ref[i]);
                    double scale = fabs((double)ref[i]);
                    double rel = (scale > 1e-6) ? d / scale : d;
                    if (rel > worst) worst = rel;
                }
                double flops = 2.0 * M * N * K;
                double bytes = (double)M * K * 4 + (double)N * K * 2 + (double)M * N * 4;
                char label[96];
                snprintf(label, sizeof label, "[%d] %s", which, cases[c].what);
                printf("%-46.46s %10.2e %10.3f %9.1f %9.1f\n", label, worst,
                       secs * 1e3, flops / secs / 1e9, bytes / secs / 1e9);
                /* fp32 accumulation in a different ORDER is the only
                 * difference available, so a few ulps is the whole budget. */
                if (worst > 1e-4) { printf("    ^^ FAIL\n"); all_ok = false; }
            }
            free(A); free(B); free(gpu); free(ref);
        }
        printf("\n%s\n", all_ok ? "every GPU kernel agrees with the CPU oracle"
                                 : "A GPU KERNEL DIVERGES");
        metal_shutdown(mtl);
    }

    /* --ppl CORPUS REFERENCE_DIR OUT.csv : this model's perplexity on CORPUS,
     * paired against REFERENCE_DIR on the same windows. */
    if (argc > 5 && strcmp(argv[2], "--ppl") == 0)
        eval_perplexity(model_dir, argv[4], argv[3], argv[5]);

    /* --quant-check : the Phase 3 kernels against matmul_q. Exit status says
     * whether they agree, so a test script can use it directly. */
    if (argc > 2 && strcmp(argv[2], "--quant-check") == 0) {
        MetalContext *mtl = metal_init("picoforge.metallib");
        bool ok = quant_check(mtl);
        metal_shutdown(mtl);
        if (!ok) return 1;
    }

    /* --greedy TEXT MAX_NEW OUT.txt : greedy generation, ids written to OUT.
     * Greedy because it is the only setting under which two implementations
     * can be compared token for token — with sampling on, matching output
     * would only prove the two RNGs agree. */
    if (argc > 5 && strcmp(argv[2], "--greedy") == 0) {
        int max_new = atoi(argv[4]);
        Weights w;
        weights_bind(&st, &cfg, &w);
        RunState state;
        state_alloc(&state, &cfg, 1024, 1);

        int *ids = malloc((size_t)max_new * sizeof *ids);
        if (!ids) die("out of memory for %d generated ids", max_new);
        int got = generate(&tok, &w, &cfg, &state, NULL, argv[3], max_new,
                           0.0f, 1.0f, 0, 0, ids, true);

        FILE *f = fopen(argv[5], "wb");
        if (!f) die("cannot write %s", argv[5]);
        for (int i = 0; i < got; i++) fprintf(f, i ? " %d" : "%d", ids[i]);
        fputc('\n', f);
        fclose(f);
        printf("greedy: %d tokens -> %s\n", got, argv[5]);

        free(ids);
        state_free(&state);
        weights_free(&w);
    }

    /* --gpu-greedy TEXT MAX_NEW OUT.txt       : plain greedy, one token per pass.
     * --spec-greedy TEXT MAX_NEW DRAFT OUT.txt : speculative, up to DRAFT
     * prompt-lookup tokens verified per pass. Both on the GPU, ids to OUT, so
     * tests/test_speculate.py can hold the second to the first token for
     * token. --gpu-greedy goes through generate(), NOT the speculative loop:
     * the reference must not share the code it is judging. */
    /* --spec-model DRAFT_DIR TEXT MAX_NEW DRAFT OUT.txt : the same, drafted by
     * a second model over the same vocabulary instead of by prompt lookup. */
    const bool spec_model = argc > 7 && strcmp(argv[2], "--spec-model") == 0;
    if (spec_model || (argc > 5 && (strcmp(argv[2], "--gpu-greedy") == 0 ||
                     (argc > 6 && strcmp(argv[2], "--spec-greedy") == 0)))) {
        const char **a = spec_model ? (const char **)argv + 1 : (const char **)argv;
        const bool spec = spec_model || strcmp(argv[2], "--spec-greedy") == 0;
        const int max_new = atoi(a[4]), draft = spec ? atoi(a[5]) : 0;
        const char *out_path = a[spec ? 6 : 5];
        MetalContext *mtl = metal_init("picoforge.metallib");
        GpuModel *gpu = gpu_model_create(mtl, &st, &cfg, 1024, 32);
        SafeTensors dst;
        Qwen3Config dcfg;
        GpuModel *drafter = NULL;
        if (spec_model) {
            config_load(argv[3], &dcfg);
            if (dcfg.vocab_size != cfg.vocab_size)
                die("the drafter's vocabulary (%d) is not the target's (%d)", dcfg.vocab_size, cfg.vocab_size);
            st_open(argv[3], &dst);
            drafter = gpu_model_create(mtl, &dst, &dcfg, 1024, 1);
        }
        int *ids = malloc((size_t)max_new * sizeof *ids);
        if (!ids) die("out of memory for %d generated ids", max_new);

        int got;
        if (spec) {
            SpecStats ss;
            got = generate_speculative(&tok, &cfg, gpu, drafter, 1024, 32, a[3], max_new, draft,
                                       ids, &ss);
            printf("spec: %d tokens in %d passes (%.2f per pass), %d/%d drafts accepted, "
                   "decode %.3f s (%.1f tok/s; drafting %.3f s, verifying %.3f s)\n", got,
                   ss.passes, ss.passes ? (double)(got - 1) / ss.passes : 0.0, ss.accepted,
                   ss.drafted, ss.decode_s, ss.decode_s > 0 ? (got - 1) / ss.decode_s : 0.0,
                   ss.draft_s, ss.verify_s);
        } else {
            Weights w;
            weights_bind(&st, &cfg, &w);
            RunState state;
            state_alloc(&state, &cfg, 1024, 1);
            got = generate(&tok, &w, &cfg, &state, gpu, a[3], max_new, 0.0f, 1.0f, 0, 0, ids, true);
            state_free(&state);
            weights_free(&w);
        }
        FILE *f = fopen(out_path, "wb");
        if (!f) die("cannot write %s", out_path);
        for (int i = 0; i < got; i++) fprintf(f, i ? " %d" : "%d", ids[i]);
        fputc('\n', f);
        fclose(f);
        free(ids);
        if (drafter) { gpu_model_free(drafter); st_close(&dst); }
        gpu_model_free(gpu);
        metal_shutdown(mtl);
    }

    /* --chat TEXT [MAX_NEW] [SEED] : wrap TEXT in the chat template and
     * generate. --complete TEXT continues raw text with no template at all,
     * which is what the base model actually does.
     *
     * The seed defaults to a constant rather than to the clock. This is a
     * measurement instrument: a run that cannot be repeated cannot be
     * compared, and a benchmark whose output changes every invocation is not
     * a benchmark. Pass a seed explicitly to vary it. */
    if (argc > 3 && (strcmp(argv[2], "--chat") == 0
                  || strcmp(argv[2], "--complete") == 0
                  || strcmp(argv[2], "--gpu-chat") == 0)) {
        bool templated = strcmp(argv[2], "--complete") != 0;
        bool on_gpu = strncmp(argv[2], "--gpu-", 6) == 0;
        int max_new = (argc > 4) ? atoi(argv[4]) : 128;
        uint64_t seed = (argc > 5) ? strtoull(argv[5], NULL, 10) : 20260912ULL;

        char prompt[16384];
        if (templated)
            chat_format(prompt, (int)sizeof prompt, argv[3], /*thinking=*/false);
        else
            snprintf(prompt, sizeof prompt, "%s", argv[3]);

        Weights w;
        weights_bind(&st, &cfg, &w);
        RunState state;
        /* One row of logits is all generation ever looks at: 0.6 MB instead
         * of the 311 MB a full prompt's worth would cost. */
        state_alloc(&state, &cfg, 1024, 1);

        MetalContext *mtl = NULL;
        GpuModel *gpu = NULL;
        if (on_gpu) {
            mtl = metal_init("picoforge.metallib");
            gpu = gpu_model_create(mtl, &st, &cfg, 1024, 1);
            /* Optional 6th argument picks the matmul kernel: 0 naive,
             * 1 simdgroup, 2 TensorOps. The benchmark says the answer is not
             * obvious at M=1, so it is a knob and not a constant. */
            if (argc > 6) gpu_set_matmul_kernel(gpu, atoi(argv[6]));
        }

        printf("\n=== generating ===\n");
        printf("device                : %s\n", on_gpu ? "GPU (Metal)" : "CPU (scalar, 1 thread)");
        printf("sampling              : temperature %.2f, top_p %.2f, top_k %d, seed %llu\n",
               (double)cfg.gen_temperature, (double)cfg.gen_top_p, cfg.gen_top_k,
               (unsigned long long)seed);
        printf("context               : 1024 positions (%.0f MB of K/V cache)\n",
               (double)cfg.num_hidden_layers * 1024.0 *
               (double)(cfg.num_key_value_heads * cfg.head_dim) * 8.0 / 1e6);
        printf("---\n");

        generate(&tok, &w, &cfg, &state, gpu, prompt, max_new, cfg.gen_temperature,
                 cfg.gen_top_p, cfg.gen_top_k, seed, NULL, false);

        if (gpu) gpu_model_free(gpu);
        if (mtl) metal_shutdown(mtl);
        state_free(&state);
        weights_free(&w);
    }

    /* --encode TEXT : one line per token, "id<TAB>bytes-as-hex". Hex rather
     * than the text itself because a token can be half of a multi-byte
     * character, and printing that raw would produce mojibake that looks like
     * a tokenizer bug rather than a display one. */
    if (argc > 3 && strcmp(argv[2], "--encode") == 0) {
        int cap = 1 << 16;
        int *ids = malloc((size_t)cap * sizeof *ids);
        if (!ids) die("out of memory for the token buffer");
        int n = tokenizer_encode(&tok, argv[3], (int)strlen(argv[3]), ids, cap);

        printf("=== tokens ===\n");
        char piece[512];
        for (int i = 0; i < n; i++) {
            int len = tokenizer_decode(&tok, &ids[i], 1, piece, (int)sizeof piece);
            printf("%d\t", ids[i]);
            for (int b = 0; b < len; b++) printf("%02x", (unsigned char)piece[b]);
            printf("\n");
        }
        printf("=== %d tokens ===\n", n);
        free(ids);
    }

    /* --nfc-file IN.bin OUT.bin : normalise each NUL-separated text. Exists so
     * tests/test_nfc.py can fuzz nfc_normalize against Python's unicodedata
     * directly. The 30 hand-picked prompts in test_encode.py show NFC works
     * where it was aimed; only a fuzz shows it works where it was not. */
    if (argc > 4 && strcmp(argv[2], "--nfc-file") == 0) {
        size_t flen;
        char *blob = slurp(argv[3], &flen);
        FILE *f = fopen(argv[4], "wb");
        if (!f) die("cannot write %s", argv[4]);

        int cap = (int)flen * 4 + 16;
        char *norm = malloc((size_t)cap);
        if (!norm) die("out of memory for the normalisation buffer");

        size_t pos = 0;
        int texts = 0;
        while (pos < flen) {
            size_t n = strlen(blob + pos);
            int m = nfc_normalize(blob + pos, (int)n, norm, cap);
            fwrite(norm, 1, (size_t)m, f);
            fputc('\0', f);
            pos += n + 1;
            texts++;
        }
        fclose(f);
        free(norm); free(blob);
        printf("normalised %d texts -> %s\n", texts, argv[4]);
    }

    /* --encode-file IN.bin OUT.txt : IN holds NUL-separated texts, OUT gets one
     * line of ids per text. NUL-separated rather than line-based because the
     * cases most likely to break a tokenizer are exactly the ones a line
     * format cannot carry: newline runs, trailing whitespace, embedded CR. */
    if (argc > 4 && strcmp(argv[2], "--encode-file") == 0) {
        size_t flen;
        char *blob = slurp(argv[3], &flen);
        FILE *f = fopen(argv[4], "wb");
        if (!f) die("cannot write %s", argv[4]);

        int cap = 1 << 20;
        int *ids = malloc((size_t)cap * sizeof *ids);
        char *back = malloc(flen * 4 + 16);
        if (!ids || !back) die("out of memory for the encode buffers");

        size_t pos = 0;
        int texts = 0;
        while (pos < flen) {
            size_t n = strlen(blob + pos);
            int count = tokenizer_encode(&tok, blob + pos, (int)n, ids, cap);
            for (int i = 0; i < count; i++)
                fprintf(f, i ? " %d" : "%d", ids[i]);
            fputc('\n', f);

            /* Decode straight back and insist on the same bytes. Encoding is
             * only half a tokenizer, and a round trip catches whole classes
             * of bug — a dropped byte, a mangled multi-byte sequence — that
             * comparing ids alone would not.
             *
             * The comparison is against the NFC-normalised text, not the
             * input: normalisation is deliberately lossy, and "e" + combining
             * acute is meant to come back as a composed accent. */
            char *want = malloc(n * 4 + 8);
            if (!want) die("out of memory for the round-trip buffer");
            int wlen = nfc_normalize(blob + pos, (int)n, want, (int)n * 4 + 8);
            int blen = tokenizer_decode(&tok, ids, count, back, (int)flen * 4 + 16);
            if (blen != wlen || memcmp(back, want, (size_t)wlen) != 0)
                die("round trip failed on text %d (%d bytes expected, %d back)",
                    texts, wlen, blen);
            free(want);

            pos += n + 1;
            texts++;
        }
        fclose(f);
        free(ids); free(back); free(blob);
        printf("encoded %d texts -> %s (all round-tripped)\n", texts, argv[4]);
    }

    /* --forward OUT.bin ID ID ... : run the model on those token ids, dump
     * the raw fp32 logits for tests/test_forward.py to judge. Tokenisation is
     * step 1.6; until then the ids come from the command line. */
    if (argc > 4 && (strcmp(argv[2], "--forward") == 0
                  || strcmp(argv[2], "--forward-incr") == 0
                  || strcmp(argv[2], "--forward-narrow") == 0
                  || strcmp(argv[2], "--gpu-forward") == 0
                  || strcmp(argv[2], "--gpu-forward-incr") == 0)) {
        /* --forward-incr feeds the prompt ONE TOKEN AT A TIME, which is the
         * path generation actually uses. It must produce bit-comparable
         * logits to the batch path: same weights, same positions, only the
         * scheduling differs. Testing only the batch path would leave the
         * cache's real failure mode untested — a RoPE rotation by t instead
         * of pos + t is invisible until the second call. */
        bool incremental = strstr(argv[2], "-incr") != NULL;
        /* --forward-narrow: the CPU batch path with Q4 inputs narrowed to
         * bf16, the GPU's arithmetic without the GPU (see model.c). */
        set_q4_bf16_activations(strcmp(argv[2], "--forward-narrow") == 0);
        bool use_gpu = strncmp(argv[2], "--gpu-", 6) == 0;
        const char *out_path = argv[3];
        int seq = argc - 4;

        int *tokens = malloc((size_t)seq * sizeof *tokens);
        if (!tokens) die("out of memory for %d tokens", seq);
        for (int i = 0; i < seq; i++) {
            tokens[i] = atoi(argv[4 + i]);
            /* An out-of-range id indexes past the embedding matrix: a read
             * into whatever follows it in the mapping, with no crash and no
             * sign that anything went wrong. */
            if (tokens[i] < 0 || tokens[i] >= cfg.vocab_size)
                die("token id %d is outside [0, %d)", tokens[i], cfg.vocab_size);
        }

        Weights w;
        weights_bind(&st, &cfg, &w);
        RunState state;
        state_alloc(&state, &cfg, seq, seq);

        size_t vocab = (size_t)cfg.vocab_size;
        float *all = malloc((size_t)seq * vocab * sizeof *all);
        if (!all) die("out of memory for %d rows of logits", seq);

        MetalContext *mtl = NULL;
        GpuModel *gpu = NULL;
        if (use_gpu) {
            mtl = metal_init("picoforge.metallib");
            gpu = gpu_model_create(mtl, &st, &cfg, seq, seq);
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (incremental) {
            for (int i = 0; i < seq; i++) {
                if (use_gpu) gpu_forward(gpu, &tokens[i], 1, i, 0, all + (size_t)i * vocab);
                else {
                    forward(&tokens[i], 1, i, 0, &w, &cfg, &state);
                    memcpy(all + (size_t)i * vocab, state.logits, vocab * sizeof *all);
                }
            }
        } else if (use_gpu) {
            gpu_forward(gpu, tokens, seq, 0, 0, all);
        } else {
            forward(tokens, seq, 0, 0, &w, &cfg, &state);
            memcpy(all, state.logits, (size_t)seq * vocab * sizeof *all);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec)
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

        printf("\n=== forward ===\n");
        printf("mode                  : %s, %s\n",
               use_gpu ? "GPU" : "CPU (scalar, 1 thread)",
               incremental ? "incremental (1 token per call)" : "batch (all at once)");
        printf("tokens                : %d\n", seq);
        printf("prefill               : %.3f s  (%.1f tok/s)\n", secs, (double)seq / secs);

        /* Top-5 of the last row: what the model thinks comes next. softmax is
         * in place and destructive, so it runs on a copy of that one row —
         * 600 KB, against the 2 seconds a second forward pass would cost. */
        float *probs = malloc(vocab * sizeof *probs);
        if (!probs) die("out of memory for the probability row");
        memcpy(probs, all + (size_t)(seq - 1) * vocab, vocab * sizeof *probs);
        softmax(probs, cfg.vocab_size);

        printf("top-5 next tokens     :");
        for (int rank = 0; rank < 5; rank++) {
            int best = 0;
            for (int i = 1; i < cfg.vocab_size; i++) if (probs[i] > probs[best]) best = i;
            printf(" %d(%.1f%%)", best, (double)probs[best] * 100.0);
            probs[best] = -1.0f;                    /* pop it and look again */
        }
        printf("\n");
        free(probs);

        FILE *f = fopen(out_path, "wb");
        if (!f) die("cannot write %s", out_path);
        size_t n = (size_t)seq * vocab;
        if (fwrite(all, sizeof(float), n, f) != n)
            die("short write to %s", out_path);
        fclose(f);
        printf("logits                : %zu floats -> %s\n", n, out_path);

        if (gpu) gpu_model_free(gpu);
        if (mtl) metal_shutdown(mtl);
        free(all);
        state_free(&state);
        weights_free(&w);
        free(tokens);
    }

    tokenizer_free(&tok);
    st_close(&st);
    return 0;
}
