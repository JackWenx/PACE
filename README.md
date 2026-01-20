<div align="center">
  <h1>PACE: Prefix-Protected and Difficulty-Aware Compression for Efficient Reasoning</h1>

  <!-- Optional: add badges here (paper / project page / models / colab) -->

  <p>

  </p>
</div>

<br />
PACE is a dual-level RL method that reduces reasoning tokens while maintaining (or improving) accuracy by combining (1) prefix-protected optimization and (2) difficulty-aware length penalties.
## How to Use

### Installation

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_DIR>

conda create -n pace python=3.10
conda activate pace
```

Install PyTorch (pick the right wheel for your CUDA setup):

```bash
pip install torch torchvision torchaudio
```

Install **verl (required: v0.6.0)**:

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
cd ..
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

### Prepare Datasets

PACE is designed for verifiable reasoning tasks where a correctness reward can be computed (e.g., exact-match final answer).

In our paper setting:

* Training set: **Skywork-o1**
* In-domain eval: **AIME24**, **AIME25**, **MATH500**, **GSM8K**
* Out-of-domain eval: **GPQA-D** (science),  **LiveCodeBench** (code), **IFEval** (instruction following)
---

### Train

PACE is typically run with a GRPO loop in **verl**, using:
* Sequence-level control: prefix-protected rollouts with a decay schedule
* Group-level control: difficulty-aware scaling via group pass rate
Paper hyperparameters (PACE-7B special setting):
* `L_init`: 512
* prefix protection steps `K`: 100

Quick Start:
```bash
python -m pace.train \
  --config configs/pace_7b.yaml \
  --data_path data/skywork_o1_train.jsonl
```
---

### Evaluate

#### 1) lightEval Benchmarks (AIME24 / AIME25 / MATH500 / GSM8K / GPQA-D)

We use **lightEval** for:

* AIME24
* AIME25
* MATH500
* GSM8K
* GPQA-D

Follow the official lightEval setup and run evaluation with your model checkpoint according to the lightEval instructions.

```bash
pip install "lighteval[vllm,math]"
```
### Run

```bash
MODEL_ARGS="pretrained=${MODEL},dtype=float16,tensor_parallel_size=${NUM_GPUS},max_model_length=32768,gpu_memory_utilization=0.8,use_chat_template=True"
MODEL_ARGS="${MODEL_ARGS},generation_parameters={max_new_tokens:30000,temperature:0.6,top_p:0.95}"
TASKS="lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gsm8k|0|0,lighteval|gpqa:diamond|0|0"
lighteval vllm "${MODEL_ARGS}" "${TASKS}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-details
```

#### 2) LiveCodeBench (Code)

For code evaluation, we use the official **LiveCodeBench** pipeline.

#### 3) IFEval (Instruction Following)

For instruction-following evaluation, we use the official **IFEval** repository and its evaluation scripts.

---

### Replicate Paper Results (Quick Path)

---

## Acknowledgments

This project is built on top of **verl (v0.6.0)** and uses **lightEval** for standard benchmark evaluation. We also rely on the official repositories for **LiveCodeBench** and **IFEval** evaluation.

* **verl (v0.6.0)**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
* **lightEval**: [https://github.com/huggingface/lighteval](https://github.com/huggingface/lighteval)
* **LiveCodeBench**: [https://github.com/LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
* **IFEval**: [https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval)

---
