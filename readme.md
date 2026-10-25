# [FLEP-RAG: Progressive Fault Localization and Experience Persistence]

This repository contains the official implementation of the paper:

**FLEP-RAG: Progressive Fault Localization and Experience Persistence**.
---
## 📌 Abstract
Self-reflective Retrieval-Augmented Generation improves reasoning quality through self-evaluation and refinement. However, existing methods still face three challenges in complex multi-hop reasoning: (1) static paradigms fail to dynamically route heterogeneous sub-queries to appropriate retrieval mechanisms; (2) existing RAG methods lack efficient and precise fault localization mechanisms: coarse-grained corrections struggle to localize erroneous steps, whereas fine-grained verification incurs substantial overhead; and (3) current frameworks lack persistent procedural memory, making them unable to store and reuse successful reasoning experiences to avoid redundant computation when encountering similar queries. To address these challenges, we propose FLEP-RAG, a self-reflective framework integrating Progressive \textbf{F}ault \textbf{L}ocalization and \textbf{E}xperience \textbf{P}ersistence. Specifically, the framework decomposes complex queries into sub-tasks and adaptively selects distinct retrieval strategies. To balance reliability and diagnostic cost, it employs a two-level progressive reflection mechanism: macro-level gating screens for risks based on execution completeness and semantic relevance, triggering micro-level diagnosis only when necessary to localize faults such as insufficient evidence and incorrect dependency bindings. The framework then preserves verified results and repairs faulty steps and their downstream dependency regions, reducing the overhead of repeated verification and global restarts. Furthermore, the experience persistence mechanism abstracts successful execution trajectories into reusable templates that retain semantic slots, retrieval strategies, and step dependencies, while novelty-based filtering reduces storage redundancy. For new queries, the framework retrieves relevant experiences to guide sub-query rewriting and retrieval strategy adjustment, aiming to reduce initial planning errors and repeated repairs. Experiments on general multi-hop benchmarks and two proprietary datasets demonstrate that FLEP-RAG outperforms competitive baselines, showcasing strong cross-domain generalization while reducing redundant computation.
---

## 📁 Directory Structure

```text
BLEP-RAG/
├── BLEPRAG/                # Core library and algorithm implementation
├── examples/               # Usage examples and execution scripts
│   ├── preprocess/         # Dataset storage directory
│   ├── .env                # Environment configuration for examples
│   ├── examplev2.py        # Step 1: Knowledge graph construction script
│   ├── example3.py         # Step 2: Dataset retrieval and reasoning script
│   ├── musique_evaluetion.py # 
│   └── wiki2_evalution.py  # 
├── prompts/                # Prompt templates and configurations
├── .env                    # Global environment configuration
└── requirements.txt        # Project dependencies
```

## 🛠️  Environment Setup

1. Install Dependencies
Ensure your environment uses Python 3.10 or higher, then run:
```
pip install -r requirements.txt
```

2. Configure Environment Variables
Create and configure a .env file in the root or examples/ directory with your API credentials:
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=[https://api.openai.com/v1](https://api.openai.com/v1)  # Or compatible API endpoint
```

## 🚀 Quick Start

1. Prepare Dataset
Place your raw datasets inside the examples/preprocess/ directory:
```
examples/preprocess/
└── your_dataset.json
```


2. Build Knowledge Graph
Run extract.py to extract entities/relations from the dataset in preprocess/ and construct the knowledge graph:

```
python examples/extract.py
```

3. Execute Retrieval & Inference
Once graph construction is complete, run Retrieval.py to perform knowledge graph retrieval and multi-hop reasoning:
```
3. python examples/Retrieval.py
```
## 📝 Prompts

All prompt templates used in this work are available in the prompts/ directory.
