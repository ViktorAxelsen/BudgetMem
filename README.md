<div align="center">
  <img src="assets/logo.png" alt="LLMRouter Logo" width="250">
</div>



<h1 align="center">BudgetMem: Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory</h1>




<div align="center">
  <p>
    <a href='https://viktoraxelsen.github.io/BudgetMem/'><img src='https://img.shields.io/badge/Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white'></a>
    <a href='https://arxiv.org/abs/xxxx.xxxxx'><img src='https://img.shields.io/badge/arXiv-xxxx.xxxxx-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white'></a>
    <br>
    <a href="https://github.com/ViktorAxelsen/BudgetMem/stargazers"><img src='https://img.shields.io/github/stars/ViktorAxelsen/BudgetMem?color=f1e05a&style=for-the-badge&logo=star&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/BudgetMem/forks"><img src='https://img.shields.io/github/forks/ViktorAxelsen/BudgetMem?color=2ea44f&style=for-the-badge&logo=git&logoColor=white' /></a>
    <a href="https://github.com/ViktorAxelsen/BudgetMem/issues"><img src='https://img.shields.io/github/issues/ViktorAxelsen/BudgetMem?color=d73a49&style=for-the-badge&logo=github&logoColor=white' /></a>
    <a href="https://www.python.org/downloads/release/python-3109/"><img src="https://img.shields.io/badge/PYTHON-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <!-- <a href="x" style="text-decoration:none;"><img src="https://img.shields.io/badge/TWITTER-ANNOUNCEMENTS-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="Twitter"></a> -->
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>




## ✨ Overview

**BudgetMem** is a runtime agent memory framework that enables **explicit performance–cost control** for **on-demand memory extraction**. Instead of building a fixed memory once and using it for all future queries, BudgetMem triggers memory computation at runtime and makes it **budget-aware** through **module-level budget tiers** and **learned routing**.

At a high level, BudgetMem organizes memory extraction as a **modular pipeline**. Each module exposes three budget tiers (**Low / Mid / High**), which can be instantiated along three complementary axes:

- **Implementation tiering**: vary the module implementation (e.g., lightweight heuristics → task-specific models → LLM-based processing)
- **Reasoning tiering**: vary inference behavior (e.g., direct → CoT → multi-step/reflection)
- **Capacity tiering**: vary model capacity (e.g., small → medium → large LLM backbones)

A lightweight **budget-tier router** selects tiers module-wise based on the query and intermediate states, and is trained with **reinforcement learning** under a cost-aware objective to provide **controllable performance–cost behavior**.

BudgetMem is designed as a unified testbed to study how different tiering strategies translate compute into downstream gains. We evaluate BudgetMem on **LoCoMo**, **LongMemEval**, and **HotpotQA**, demonstrating strong performance in performance-first settings and clear performance–cost frontiers under tighter budgets.


<div align="center">
  <img src="./assets/model.png" width="800" alt="BudgetMem">
</div>




## 📰 News

- 🚀 **[2026-02]**: **BudgetMem** is officially released — a runtime agent memory framework that enables **explicit performance–cost control** via **module-level budget tiers** and **learned budget-tier routing**, supporting controllable on-demand memory extraction across diverse benchmarks ✨. **Stay tuned! More detailed instruction updates coming soon.**




## 🔗 Links

- [Overview](#-overview)





## 🚀 Get Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ViktorAxelsen/BudgetMem
cd BudgetMem

# Create and activate virtual environment
conda create -n budgetmem python=3.10
conda activate budgetmem

# specific lib
pip install xxx
# Others
pip install -r requirements.txt
```




### 📊 Preparing Training Data

BudgetMem builds training and evaluation data from the datasets below. Please download data from the official sources and place them under `data/`. Unless otherwise noted, splits are handled by our codebase.

#### **1) LoCoMo**
- Download LoCoMo from the official repo: [LoCoMo](https://github.com/snap-research/locomo)  
- Put the downloaded file under:
  - `data/locomo10.json`
- **More instructions (splits / preprocessing) will be added here.**

#### **2) LongMemEval**
- Download LongMemEval from the official repo: [LongMemEval](https://github.com/xiaowu0162/LongMemEval)  
- Put the processed file under:
  - `data/longmemeval_s_cleaned.json`
- Use our split file:
  - `data/longmemeval_s_splits.json` (train/val/test)

#### **3) HotpotQA**
- Download HotpotQA from: [HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main) (Source: [HotpotQA](https://hotpotqa.github.io/))
- We construct a training set by randomly sampling **7K** examples from the full training data (~32K) and place it under:
  - `data/xxxxx.json`  
- For evaluation, we use the test file:
  - `data/eval_200.json`

❗ **Extending to more datasets and runtime pipelines.** BudgetMem is designed to be easy to extend: you can plug in new datasets by defining (i) the data loader and evaluation protocol, and (ii) the module set along with their **Low/Mid/High budget-tier implementations** under a chosen tiering strategy (implementation / reasoning / capacity). See [Extending to New Datasets and Pipelines](#-extending-to-new-datasets-and-pipelines) for step-by-step instructions.



## 🧪 Experiments



### 🖥️ Training



### 🧭 Evaluation




## 🔧 Extending to New Datasets and Pipelines






## 🙏 Acknowledgments

We thank the authors and maintainers of **[LoCoMo](https://github.com/snap-research/locomo)**, **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)**, and **[HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main)** (source: **[HotpotQA](https://hotpotqa.github.io/)**) for releasing their datasets, evaluation protocols, and supporting code. Their efforts in building and open-sourcing high-quality benchmarks make it possible to develop, evaluate, and reproduce research on agent memory.

We also thank **[LightMem](https://github.com/zjunlp/LightMem)** for pioneering performance–efficiency considerations in memory systems, which helped motivate our focus on explicit budget control, and **[GAM](https://github.com/VectorSpaceLab/general-agentic-memory)** for advancing runtime agent memory frameworks.



## Citation

```bibtex
@article{BudgetMem,
  title={Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory},
  author={Haozhen Zhang and Haodong Yue and Tao Feng and Quanyu Long and Jianzhu Bao and Bowen Jin and Weizhi Zhang and Xiao Li and Jiaxuan You and Chengwei Qin and Wenya Wang},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```