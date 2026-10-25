# [BLEP-RAG: Breakpoint Localization and Experience Persistence]

This repository contains the official implementation of the paper:

**BLEP-RAG: Breakpoint Localization and Experience Persistence**.
---
## 📌 Abstract
Retrieval-Augmented Generation has become a dominant paradigm for mitigating hallucinations in Large Language Models. To mitigate potential retrieval inaccuracies, recent advances have introduced self-reflective RAG frameworks that utilize LLMs to self-evaluate and iteratively refine retrieved contexts. However, when tackling complex long-chain reasoning tasks, existing reflection mechanisms still face the following three challenges: (1) Static paradigms fail to dynamically dispatch heterogeneous sub-queries across distinct access mechanisms (e.g., graph traversal, vector search, or LLM parametric knowledge) at a fine grain; (2) Existing RAG methods lack efficient and precise fault localization mechanisms, where coarse-grained correction struggles to pinpoint erroneous steps while fine-grained verification incurs substantial overhead; (3) Current frameworks lack persistent procedural memory, making them incapable of preserving and reusing successful reasoning experiences to avoid redundant computation.
To overcome these limitations, we propose a self-evolving RAG framework featuring Breakpoint Localization and Experience Persistence (BLEP-RAG), which optimizes the traditional RAG paradigm through two key innovations: fine-grained execution verification and persistent experience learning.
First, the framework decomposes complex queries into a directed acyclic graph and adaptively selects retrieval strategies based on the characteristics of different sub-queries. After retrieval, the framework assesses the reliability of the reasoning process to identify the earliest erroneous breakpoint in the execution chain. Instead of restarting retrieval from scratch, it preserves all verified intermediate results and replans only the unreliable reasoning segments, thereby reducing redundant computation while improving retrieval reliability.
Second, the system applies de-instantiation abstraction to successful reasoning paths, abstracting concrete knowledge instances into generalized semantic slots and relational meta-paths to extract reusable procedural templates into an experience database. During cross-task migration, the system dynamically retrieves the most relevant experience as few-shot cases to guide the retrieval task (regularizing LLM behavior in retrieval routing, keyword extraction, and spatio-temporal boundary delineation to guard against scope and execution drift) based on the topological and semantic similarities between the new query and historical reasoning paradigms. Furthermore, a novelty gating mechanism is introduced to control experience growth, preventing redundant storage of highly similar experiences and mitigating memory explosion in the experience database.
Extensive experiments on general multi-hop benchmarks demonstrate that BLEP-RAG achieves competitive performance against advanced RAG baselines, while evaluations on proprietary datasets across financial-economic and maritime domains further confirm its strong cross-domain generalization.
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