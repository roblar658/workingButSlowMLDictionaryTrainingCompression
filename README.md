# Expected output when using a compiled solve.py, given a dictionary

<img width="959" height="462" alt="image" src="https://github.com/user-attachments/assets/8197c7b2-7717-496c-bbae-5e1821d5fcb4" />

---

# DP + Neural Dictionary Compression (`dict.txt` → `< 88 KiB` standalone UTF-8 `solve.py` & `solve.cpp`)

This repository contains standalone, lossless dictionary compressors and self-extracting solvers in both **Python 3** ([`solve.py`](solve.py)) and **C++20** ([`solve.cpp`](solve.cpp)) for [`dict.txt`](dict.txt) (`972,104` bytes, `102,399` English words):

| Metric | Python ([`solve.py`](solve.py), Dual-Tier PAQ / C-Bridge) | C++20 ([`solve.cpp`](solve.cpp), Single-Core `RemOnly`) |
|---|---|---|
| Standalone UTF-8 file size | **`89,234` B (`87.14 KiB`)** (`9.18%` of input, `+878 B` headroom) | **`89,975` B (`87.87 KiB`)** (`9.26%` of input, `+137 B` headroom) |
| External compression / build deps | **None** (pure Python stdlib + optional background C-acceleration bridge with zero-dependency fallback) | **None** (zero threads, zero pthreads, zero OpenMP, standard `<cmath>`, `<cstdint>`, `<cstdio>`, `<cstring>`) |
| Compile time | **None** (runtime standalone script) | `1.15–1.40 s` (`g++ -O2 -std=c++20 -Wall -Wextra -Wbidi-chars=any`, 0 warnings) |
| Warm subprocess wall (`min` / `median`) | **`~0.27–0.32 s` / `~0.35 s`** (exact lossless match, 0 bidi warnings) | **`12.80 ms` / `13.40 ms`** (quiet warm runs) |
| Internal core runtime (`clock_gettime`) | — | **`8.87 ms` (min) / `8.99 ms` (median)** with `-O3` (**`~2.0x` single-core speedup** vs `44e41a2` baseline) |
| Raw payload (`bin_tot`) | `73,848 B` (`ml_bin=67,370 B`, `cs_b=3,682 B`, `w_b=1,790 B`, `m_b=1,423 B`, `py_b=3,259 B`) | `74,012 B` (`ml_bin=57,112 B`, `meta_lz=1,217 B`, `pci_b=15,675 B` + 8 B header) |

---

## Architecture & Pipeline Flowcharts

### 1. End-to-End Compression Pipeline

```mermaid
flowchart TD
    DICT["dict.txt (102,399 words, 972 KB)"] --> DISCOV["Morphological Operator Discovery (52 rules)"]
    DISCOV --> GRAPH["Derivation Graph Partitioning"]
    GRAPH --> COVER_DP["Exact Cover DP (Tree DP + Bitmask Subset DP)"]
    COVER_DP --> DATASET["ml_dataset.pkl (35,364 base words + 200 inflection masks)"]

    DATASET --> ROUTER["Context Router (Decompose into p, c0, rem, inf)"]

    ROUTER --> TARGET_CPP["C++ Pipeline (solve.cpp)"]
    ROUTER --> TARGET_PY["Python Pipeline (solve.py)"]

    subgraph "C++ Neural PAQ Pipeline"
        TARGET_CPP --> RANS_CPP["Unified Context rANS (p, c0, inf - RP 143x256)"]
        TARGET_CPP --> PAQ_CPP["3-Order Bit-FSM + Logistic Model (418k bits)"]
        PAQ_CPP --> ADAM_CPP["Adam Optimizer & int8 Quantization (T1, T2, T3)"]
        ADAM_CPP --> RANGE_CPP["32-bit Multi-Precision Range Encoder"]
        RANS_CPP & RANGE_CPP --> LZSS_CPP["DP-Optimal LZSS (Metadata & Bytecode)"]
        LZSS_CPP --> UTF8_CPP["Mixed-Radix Non-Bidi UTF-8 (R literal)"]
        UTF8_CPP --> SOLVE_CPP["solve.cpp (89,975 B, 9ms runtime)"]
    end

    subgraph "Python Dual-Tier Pipeline"
        TARGET_PY --> PAQ_PY["PAQ 160-state Bit-FSM Model"]
        TARGET_PY --> BRIDGE_PY["Dual-Tier C-Bridge / Python Fallback Engine"]
        PAQ_PY & BRIDGE_PY --> LZSS_PY["DP-Optimal LZSS Compressor"]
        LZSS_PY --> UTF8_PY["Safe Non-Bidi UTF-8 String Literal"]
        UTF8_PY --> SOLVE_PY["solve.py (89,234 B, 0.35s runtime)"]
    end

    SOLVE_CPP --> LOSSLESS["Exact Lossless Reconstruction (100% SHA-256 match)"]
    SOLVE_PY --> LOSSLESS
```

---

### 2. Morphological Cover Dynamic Programming

```mermaid
flowchart LR
    subgraph "Derivation Graph Analysis"
        W1["Word Component (e.g. play, plays, played, player, replay)"]
        RULES["52 Morphological Rules (phi_0 .. phi_51)"]
        W1 --> RULES
    end

    subgraph "Exact DP Partitioning"
        RULES --> CHECK{"Component Topology"}
        CHECK -->|"Tree (|E| = |V| - 1)"| TREE_DP["Bottom-Up Tree DP (covP, covSelf, up, covChild)"]
        CHECK -->|"Small Cyclic (|V| <= 20)"| BITMASK_DP["Exact Bitmask Subset DP (2^|V| state space)"]
        CHECK -->|"Dense Component"| GREEDY_DP["Iterative Closed-Group Window DP"]
    end

    subgraph "Optimal Partition"
        TREE_DP & BITMASK_DP & GREEDY_DP --> BASES["35,364 Base Words (LCP prefix encoded)"]
        TREE_DP & BITMASK_DP & GREEDY_DP --> MASKS["200 Pre-Folded Derivation Bitmasks"]
    end
```

---

### 3. Runtime Solver Decompression Architecture

```mermaid
flowchart TD
    subgraph "C++ Solver (solve.cpp - 8.9ms)"
        U_CPP["Embedded UTF-8 Literal (R-string)"] --> DEC_UTF8_C["rANS Mixed-Radix UTF-8 Decoder"]
        DEC_UTF8_C --> BLOB_C["Binary Blob (Bytecode, Quantized Weights, Streams)"]
        BLOB_C --> LZSS_DEC_C["DP-LZSS Decompressor (Metadata & Affix VM)"]
        LZSS_DEC_C --> T_INLINE["Inlined Folded Tables: T1, T2, T3 (L2 Cache Resident)"]
        BLOB_C --> RANS_DEC_C["Interleaved Single-Stream rANS: p, c0, inf"]
        BLOB_C --> RANGE_DEC_C["Zero-NN PAQ 32-bit Range Decoder: rem"]
        RANS_DEC_C & RANGE_DEC_C --> STEM_C["Direct Stem Reassembly (W buffer)"]
        STEM_C --> AP_VM["Affix Expansion Virtual Machine (AP)"]
        AP_VM --> RADIX_C["Fast 8-byte Radix Sort & Unaligned Copy"]
        RADIX_C --> STDOUT_C["Direct fwrite to stdout"]
    end

    subgraph "Python Solver (solve.py - 0.35s)"
        U_PY["Embedded UTF-8 String (U)"] --> DEC_UTF8_PY["Pure-Python Mixed-Radix UTF-8 Decoder"]
        DEC_UTF8_PY --> BLOB_PY["Binary Blob (Weights, Python bytecode, Bitstream)"]
        BLOB_PY --> LZSS_DEC_PY["DP-LZSS Python Decompressor"]
        LZSS_DEC_PY --> C_BRIDGE{"GCC available in background?"}
        C_BRIDGE -->|"Yes (gcc Popen)"| SHARED_LIB["Compile & Load Micro C Shared Library"]
        C_BRIDGE -->|"No (Fallback)"| PURE_PY["Instant Pure-Python Decompression Engine"]
        SHARED_LIB & PURE_PY --> EXPAND_PY["Morphological Affix Expansion (pl, ed, ing, ...)"]
        EXPAND_PY --> TIMSORT_PY["Native Python Timsort"]
        TIMSORT_PY --> STDOUT_PY["sys.stdout.write"]
    end
```

---

## Modular Code Architecture (`compressor/` Package)

To ensure clean separation of concerns, transparency, and reusability, all non-solver logic is organized into a modular package:

```
gh_repo/
├── compressor/                     # Core compression & generation framework
│   ├── __init__.py                 # Top-level package exports
│   │
│   ├── morphology/                 # Word analysis & graph partition
│   │   ├── __init__.py
│   │   ├── operators.py            # MorphologicalRules: 52 derivation & inflection rules
│   │   ├── cover_dp.py             # ExactCoverDP: Tree DP & Bitmask Subset DP solvers
│   │   └── dataset.py              # DatasetBuilder: Graph builder & candidate mask selection
│   │
│   ├── models/                     # Statistical & machine-learning modeling
│   │   ├── __init__.py
│   │   ├── paq_fsm.py              # PAQBitFSM: 80/160 state bit-history finite state machine
│   │   ├── context_router.py       # ContextRouter: Causal (p, c0, rem, inf) routing & hashing
│   │   └── neural_trainer.py       # NeuralLogisticTrainer: Adam, int8 quantization, range encoder
│   │
│   ├── encoding/                   # Entropy coding primitives
│   │   ├── __init__.py
│   │   ├── huffman.py              # CanonicalHuffman: Canonical tree generation & lengths
│   │   ├── rans.py                 # RangeANS: Order-0 & unified multi-context rANS
│   │   ├── lzss.py                 # DPOptimalLZSS: 4-pass rate-distortion optimal LZSS
│   │   └── utf8_rans.py            # UTF8Encoder: Non-bidi safe mixed-radix stream encoders
│   │
│   └── codegen/                    # Standalone self-extracting code generators
│       ├── __init__.py
│       ├── cpp_generator.py        # CppSolverGenerator: Emits self-contained solve.cpp
│       └── py_generator.py         # PySolverGenerator: Emits self-contained solve.py
│
├── save_ml_dataset.py              # CLI: Discovers cover rules and creates ml_dataset.pkl
├── mlDictionaryTrainingCompressionCpp.py # CLI: Trains C++ model and generates solve.cpp
├── mlDictionaryTrainingCompression.py    # CLI: Trains Python model and generates solve.py
├── solve.cpp                       # Verified standalone C++ solver (89,975 B, ~9ms)
├── solve.py                        # Verified standalone Python solver (89,234 B, ~0.35s)
├── dict.txt                        # Target dictionary (102,399 words, 972,104 bytes)
└── README.md                       # Architecture documentation & benchmarks
```

---

## Single-Core Benchmark (Baseline `44e41a2` vs New `RemOnly` Solver)

All runs verified byte-for-byte (`exact=True`, `fc.exe /b` = 0 differences) against [`dict.txt`](dict.txt) (`972,104` bytes, `102,399` words) on Windows 11 (MinGW-W64 GCC `15.2.0`, single CPU core):

### Interleaved Single-Core Benchmark (15 Warm Runs, `clock_gettime(CLOCK_MONOTONIC)`)

| Implementation | Compiler Flags | File Size | `min` | `p25` | `median` | `p75` | `max` | Speedup vs Baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline C++ (`44e41a2`, `K=1`) | `g++ -O2 -std=c++20` | `89,955 B` (`87.85 KiB`) | `17.41 ms` | `17.71 ms` | `17.83 ms` | `18.00 ms` | `18.64 ms` | `1.00x` (baseline) |
| **New C++ (`RemOnly`, `solve.cpp`)** | `g++ -O2 -std=c++20` | **`89,975 B` (`87.87 KiB`)** | **`9.09 ms`** | **`9.14 ms`** | **`9.16 ms`** | **`9.19 ms`** | **`9.25 ms`** | **`1.92x` (min) / `1.95x` (med)** |
| **New C++ (`RemOnly`, `solve.cpp`)** | `g++ -O3 -std=c++20` | **`89,975 B` (`87.87 KiB`)** | **`8.87 ms`** | **`8.95 ms`** | **`8.99 ms`** | **`9.08 ms`** | **`9.28 ms`** | **`1.96x` (min) / `1.98x` (med)** |

*Under normal background desktop load conditions, baseline wall time often throttles to `31–34 ms`, where the new solver executes in `13.8–15.2 ms` (**`> 2.2x` faster**).*

---

## Deep Mathematical Architecture

The compression pipeline minimizes the Kolmogorov-style joint description length

$$L_{\text{total}} = L_{\text{struct}}(\text{bases}, \text{masks}) + L_{\text{model}}(\theta_q) + L_{\text{data}}(\text{bitstream} \mid \theta_q) + L_{\text{code}}(\text{C} + \text{Py}) + L_{\text{UTF8}}$$

across five tightly coupled mathematical stages:

### 1. Morphological Derivation Graph & Dynamic Programming Cover
We define $N_{\text{ops}} = 52$ deterministic English morphological operators $\phi_0, \dots, \phi_{51} : \Sigma^* \to \mathcal{P}(\Sigma^*)$ covering apostrophe-s, regular/irregular plurals, `-ed`, `-ing`, `-er`, `-est`, `-ly`, `-ness`, `-ion`, `-ment`, `-able`, `-ity`, `-ize`, consonant doubling, `-y`/`-al`/`-ic`/`-ive`/`-ous`, combined `un-`/`re-`/`dis-` inflections, and 15 standalone prefixes.

Finding a lossless morphological cover partitions $D$ into disjoint stars rooted at base words $B \subseteq D$:

$$D = \bigsqcup_{b \in B} \big(\lbrace b\rbrace \cup \Delta(b, m_b)\big), \qquad \text{minimize } \sum_{b \in B} c(b, m_b)$$

Solved via exact bottom-up Tree DP for tree components, exact Bitmask Subset DP for small cyclic components ($|V| \le 20$), and iterative closed-group DP for larger components, reducing the base dictionary to **$35,364$ bases** and **$112,375$ suffix characters**.

### 2. Suffix Decomposition & Causal Context Routing
Sorted base words $w_0 < w_1 < \dots < w_{N_B - 1}$ are decomposed into `(p_i, c0_i, rem_i, inf_i)`:
1. **Prefix length $p_i = \text{LCP}(w_{i-1}, w_i) \in [0, 19]$:** Decoded via static rANS conditioned on $48$ compact contexts.
2. **First differing character $c_{0,i} = w_i[p_i]$:** Decoded via $55$ contexts keyed by previous character location.
3. **Suffix remainder (`rem_i`) + `EOW`:** Handled by the 3-order PAQ bit-FSM predictor in C++ and entropy-quantized context models in Python.
4. **Inflection mask index $\text{inf}_i$:** Decoded conditioning on uppercase status and trailing character n-grams.

### 3. PAQ-Style 3-Order Bit-FSM (`solve.cpp` Single-Core Architecture)
- **Collision-Free Direct $F_2$ Indexing:** $\text{idx}_{F_2} = (c_2 \cdot 64 + c_1) \cdot 64 + u$ ($225,280$ bytes), eliminating hash operations.
- **L2 Cache Residency:** Sizing $F_3$ to $2^{18} + 53 = 262,197$ bytes keeps the entire active working set ($\approx 529$ KB) resident in L2 cache.
- **Inlined Folded Tables:** $T_1$, $T_2$, and $T_3$ pre-fold weights, biases, and padding so logit computation is a single 3-term addition: $\text{logit}_d = T_1[u][s_1] + T_2[u][s_2] + T_3[u][s_3]$.
- **Half-Range Sigmoid:** Startup initialization is restricted to $|i - \text{sig\_pad}| \le 850$, reducing startup time to $< 0.18$ ms.

### 4. Custom DP-Optimal LZSS + Static rANS
Auxiliary metadata and bytecode are parsed using 4-pass backward dynamic programming evaluating optimal literal vs match parsing ($l \in [3, 288]$, $d \le 32,767$) with dynamic Shannon entropy feedback.

### 5. Mixed-Radix UTF-8 rANS Source Embedding
Payloads are packed into high-density UTF-8 characters excluding control characters, quotes, and bidirectional markers (`U+061C`, `U+200E–200F`, `U+202A–202E`, `U+2066–2069`), achieving $> 7.04$ bits per UTF-8 byte with zero compiler warnings under `-Wbidi-chars=any`.

---

## Reproducing From Scratch

```bash
# 1. Build DP-optimal morphological dataset (requires dict.txt)
python save_ml_dataset.py

# 2a. Train & generate Python standalone solver (solve.py, ~0.35s)
python mlDictionaryTrainingCompression.py
python solve.py > output_py.txt

# 2b. Train & generate standalone single-core C++20 solver (solve.cpp, ~9ms)
python mlDictionaryTrainingCompressionCpp.py
g++ -O3 -std=c++20 -Wall -Wextra -Wbidi-chars=any solve.cpp -o solve_cpp
./solve_cpp > output_cpp.txt
```
