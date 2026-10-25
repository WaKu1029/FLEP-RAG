import os
from pathlib import Path

PROMPTS: dict = {}

def load_prompt(filename: str) -> str:
    """加载 prompts 目录下的 .md 文件"""
    file_path = Path(__file__).parent / f"{filename}.md"
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()
            # 自动去除 markdown 代码块标记（如果有）
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]).strip()
            return content
    except FileNotFoundError:
        print(f"⚠️ Prompt file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"⚠️ Failed to load prompt {filename}: {e}")
        return ""

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "||"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "###"
GRAPH_FIELD_SEP = "<SEP>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "person", "organization", "Location", "Date", "Number",
    "Artwork", "Adjective", "Event", "Theory"
]

PROMPTS["entity_extraction"] = load_prompt("entity_extraction")
PROMPTS["multi_hop_query_decomposition_triple"] = load_prompt("multi_hop_query_decomposition_triple")
PROMPTS["multi_hop_query_decomposition_comparative"] = load_prompt("multi_hop_query_decomposition_comparative")
PROMPTS["query_type_classifier"] = load_prompt("query_type_classifier")
PROMPTS["query_type_classifier"] = load_prompt("query_type_classifier")
PROMPTS["verification_backtracking"] = load_prompt("verification_backtracking")
PROMPTS["Retrieval Strategy Router"] = load_prompt("Retrieval Strategy Router")
PROMPTS["Operation_Respond"] = load_prompt("Operation_Respond")
PROMPTS["call_llm_for_matching"] = load_prompt("call_llm_for_matching")
PROMPTS["reliability_verification"] = load_prompt("reliability_verification")
PROMPTS["step_verification"] = load_prompt("step_verification")
PROMPTS["DEFAULT_USER_PROMPT"] = load_prompt("DEFAULT_USER_PROMPT")
PROMPTS["fail_response"] = load_prompt("fail_response")
PROMPTS["rag_response"] = load_prompt("rag_response")
PROMPTS["entity_continue_extraction"] = load_prompt("entity_continue_extraction")
PROMPTS["entity_extraction"] = load_prompt("entity_extraction")
PROMPTS["execution_plan_optimizer"] = load_prompt("execution_plan_optimizer")
PROMPTS["entity_extraction_examples"] = load_prompt("entity_extraction_examples")
PROMPTS["entity_if_loop_extraction"] = load_prompt("entity_if_loop_extraction")
PROMPTS["keywords_extraction_examples"] = load_prompt("keywords_extraction_examples")
PROMPTS["keywords_extraction"] = load_prompt("keywords_extraction")
print("✅ Prompts loaded successfully!")