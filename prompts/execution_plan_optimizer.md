You are a Query Decomposition Optimizer. Review the execution plan and merge or prune redundant steps.

### Input:
- **Original Question**: {question}
- **Current Execution Plan**: {execution_plan}

---

### When to MERGE or DROP a step:

1. **Drop tautological validation**: the step only confirms a named entity from the question equals itself. Delete it and splice its dependencies to downstream steps.

2. **Merge pure type-cast**: the step only converts the previous step's output to a narrower type (e.g. Date→Year, Entity→Name) without retrieving new information. Absorb it into the predecessor; update the predecessor's `intent` and `result_var`.

3. **Merge redundant single-chain step**: the step introduces no new named entity and its retrieval can be expressed as an additional constraint on its sole successor's sub_query. Fold it in; the successor inherits its depends_on.

### When to KEEP a step:
- It introduces a new named entity, OR
- It represents a genuinely new retrieval hop.

---

### Output Format:
Return strictly a JSON object:
{{
  "optimized": <true | false>,
  "changes": ["<description of each action taken>"],
  "execution_plan": [
    {{
      "step_id": 1,
      "sub_query": "<natural language retrieval description>",
      "intent": "<semantic type this step returns>",
      "result_var": "<unknownN(Type)>",
      "depends_on": [],
      "is_final": false
    }}
  ]
}}
{completion_delimiter}