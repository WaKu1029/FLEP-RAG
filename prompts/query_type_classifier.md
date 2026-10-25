### Role
You are a query type classifier.
Analyze the following question and determine its reasoning type.
### Input Question:
Question: {question}

Possible types:
- "comparison": Questions that compare two or more entities (older/younger, larger/smaller, more/less, better/worse, first/last, etc.)
- "multi_hop": General multi-hop reasoning questions that require chaining facts.

### Output Format:
Return only JSON:
{{"reasoning_type": "comparison" or "multi_hop"}}

{completion_delimiter}