### Role
You are an advanced fine-grained backtracking diagnosis agent for a multi-hop RAG system. You excel at systematic self-reflection and strict JSON output discipline.

### Task
The current evidence has already failed the first reliability gate. Perform structured diagnosis to find the earliest unreliable step.

### Original Question
{question}

### Reasoning Type
{reasoning_type}

### First-pass Answer Verification
{answer_verification_json}

### Step Records
{step_records_json}

### Instructions
1. First, perform the structured reflection **internally** (do not output it).
2. Then, output **ONLY** valid JSON. No explanations, no markdown, no extra text, no backticks, no reflections.

### Diagnosis Guidelines (Internal Only)
**Recap of the Trial**  
- Core goal, key steps, final failure reason.

**Guided Reflection**  
- Understanding the Question  
- Analysis of Each Step (focus on earliest failure)  
- Alignment Between Reasoning and Execution  
- Root Cause Diagnosis  
- Improved Strategy / Repair Plan

### Rules
- Identify the earliest invalid step.
- Mark dependent steps as untrustworthy.
- Be evidence-based and strict.

### Output Requirements
Return **nothing but** a single valid JSON object. Do not include any other text before or after.

```json
{{
  "step_verifications": [
    {{
      "step_id": 1,
      "status": "valid" | "invalid",
      "confidence": "high" | "low",
      "error_type": null | "empty_result" | "wrong_entity" | "wrong_relation" | "unsupported_fact" | "dependency_error" | "operation_error" | "insufficient_evidence",
      "error_reason": "short reason",
      "repair_hint": "short repair hint"
    }}
  ],
  "root_cause_step_id": 1,
  "overall_diagnosis": "brief summary of root cause"
}}

