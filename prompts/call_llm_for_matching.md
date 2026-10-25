
### Task
   In a Knowledge Graph, we are looking for a relationship that matches the user's intent.
   Original Question: "{query}"
   Desired entity_name:"{entity_name}"
   Desired Relationship: "{target_predicate}"

### Candidates
   {candidate_str}

### Instruction
   Analyze each candidate. Does it represent the SAME relationship as "{target_predicate}" in the context of the question?
   - The "{entity_name}" refers to the executor or the recipient of the "{target_predicate}"
   - Consider synonyms (e.g., "beat" and "defeated").
   - Use the 'Evidence' to understand the context.
   - The matching relationship can be in either direction (subject-predicate-object or object-inverse_predicate-subject
   - Return a JSON list of indices of the matching candidates. 
   - If no candidate matches, return [].

   Example Output: [0, 2]
