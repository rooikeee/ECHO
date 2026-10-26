# 910C synthetic-KV English book summary benchmark

16 devices, GLM-5.2 W4A8, TP16/EP16, MTP3, 64k input / 256 output tokens. Output text is not an accuracy result. Maximum concurrency is sampled every second.

| Case | Success | Output token/s | Mean TTFT (s) | Mean TPOT (ms) | Max running | Max waiting |
|---|---:|---:|---:|---:|---:|---:|
| shared30-rate-0.3 | 30/30 | 55.69 | 0.44 | 66.54 | 8 | 0 |
| optimized60-rate-0.3 | 60/60 | 63.08 | 0.41 | 73.85 | 11 | 0 |
| optimized60-rate-0.5 | 60/60 | 99.37 | 0.47 | 96.36 | 16 | 1 |
| optimized60-rate-0.7 | 60/60 | 117.09 | 5.26 | 101.09 | 16 | 8 |
| optimized60-rate-inf | 60/60 | 139.66 | 39.76 | 98.77 | 16 | 44 |
| original60-rate-0.3 | 60/60 | 65.10 | 0.62 | 49.11 | 5 | 2 |
| original60-rate-0.5 | 60/60 | 76.81 | 20.14 | 60.61 | 5 | 16 |
| original60-rate-0.7 | Not completed | — | — | — | — | — |
| original60-rate-inf | Not completed | — | — | — | — | — |

Run status: stopped. Error: RuntimeError('original60-rate-0.7 failed').

See ../ECHO_INFINITEBENCH_910C.md for setup, methodology, memory accounting and limitations.
