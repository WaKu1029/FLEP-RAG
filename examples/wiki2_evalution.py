import json
import os
import asyncio
import re
from typing import Dict, List, Any

from BLEPRAG import BLEPRAG, QueryParam
from BLEPRAG.llm.openai import openai_embed
from BLEPRAG.kg.shared_storage import initialize_pipeline_status
from BLEPRAG.utils import EmbeddingFunc, setup_logger
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
setup_logger("BLEPRAG", level="INFO")

# ─────────────────────────────────────────────
# 全局 Token 计数器
# ─────────────────────────────────────────────
token_counter = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}


def snapshot_tokens() -> Dict[str, int]:
    return dict(token_counter)


def diff_tokens(before: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": token_counter["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": token_counter["completion_tokens"] - before["completion_tokens"],
        "total_tokens": token_counter["total_tokens"] - before["total_tokens"],
    }


# ─────────────────────────────────────────────
# LLM 函数（支持 Token 统计）
# ─────────────────────────────────────────────
_llm_client: AsyncOpenAI = None


def get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return _llm_client


async def qwen_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    allowed_keys = {"temperature", "max_tokens", "top_p", "stream"}
    valid_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    response = await client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
        extra_body={"enable_thinking": False},
        **valid_kwargs
    )

    if response.usage:
        token_counter["prompt_tokens"] += response.usage.prompt_tokens
        token_counter["completion_tokens"] += response.usage.completion_tokens
        token_counter["total_tokens"] += response.usage.total_tokens

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# Embedding 函数
# ─────────────────────────────────────────────
async def qwen_embedding_func(texts: list[str]):
    batch_size = 10
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = await openai_embed(
            batch,
            model="text-embedding-v4",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        results.extend(resp)
    return results


# ─────────────────────────────────────────────
# RAG 初始化
# ─────────────────────────────────────────────
async def initialize_qwen_rag(working_dir: str) -> BLEPRAG:
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    rag = BLEPRAG(
        working_dir=working_dir,
        llm_model_func=qwen_llm_model_func,
        chunk_token_size=2500,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=qwen_embedding_func
        )
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# ─────────────────────────────────────────────
# LLM 评估器（保留，但当前未启用）
# ─────────────────────────────────────────────
class LLMEvaluator:
    def __init__(self):
        self.client = get_llm_client()
        self.metrics = {'total_questions': 0, 'correct_answers': 0, 'partial_correct': 0, 'wrong_answers': 0}

    # ...（LLMEvaluator 代码保持不变，省略以节省篇幅，如需要可保留完整版本）


# ─────────────────────────────────────────────
# HotpotEvaluator（含断点）
# ─────────────────────────────────────────────
class HotpotEvaluator:
    def __init__(self, checkpoint_file="evaluation_checkpoint_MultiHopRAG.json"):
        self.checkpoint_file = checkpoint_file
        self.total = 0
        self.answer_em = 0
        self.answer_f1_sum = 0.0
        self.sup_em = 0
        self.sup_f1_sum = 0.0
        self.joint_em = 0
        self.joint_f1_sum = 0.0
        self.load_metrics()

    def load_metrics(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    self.total = d.get("total", 0)
                    self.answer_em = d.get("answer_em", 0)
                    self.answer_f1_sum = d.get("answer_f1_sum", 0.0)
                    self.sup_em = d.get("sup_em", 0)
                    self.sup_f1_sum = d.get("sup_f1_sum", 0.0)
                    self.joint_em = d.get("joint_em", 0)
                    self.joint_f1_sum = d.get("joint_f1_sum", 0.0)
            except Exception as e:
                print(f"加载评估指标失败: {e}")

    def save_metrics(self):
        data = {
            "total": self.total, "answer_em": self.answer_em, "answer_f1_sum": self.answer_f1_sum,
            "sup_em": self.sup_em, "sup_f1_sum": self.sup_f1_sum,
            "joint_em": self.joint_em, "joint_f1_sum": self.joint_f1_sum,
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, answer_em: int, answer_f1: float, sup_em: int, sup_f1: float):
        self.total += 1
        self.answer_em += answer_em
        self.answer_f1_sum += answer_f1
        self.sup_em += sup_em
        self.sup_f1_sum += sup_f1

        joint_em = int(answer_em == 1 and sup_em == 1)
        joint_f1 = answer_f1 * sup_f1 if sup_f1 > 0 else 0.0
        self.joint_em += joint_em
        self.joint_f1_sum += joint_f1
        self.save_metrics()

    def summary(self):
        if self.total == 0: return {}
        return {
            "Total": self.total,
            "Answer_EM": self.answer_em / self.total,
            "Answer_F1": self.answer_f1_sum / self.total,
            "SupportingFact_EM": self.sup_em / self.total,
            "SupportingFact_F1": self.sup_f1_sum / self.total,
            "Joint_EM": self.joint_em / self.total,
            "Joint_F1": self.joint_f1_sum / self.total,
        }


# ─────────────────────────────────────────────
# 评估辅助函数（保持 V1 精简版）
# ─────────────────────────────────────────────
def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    if text in ["yes", "y", "1"]:
        return "yes"
    elif text in ["no", "n", "0"]:
        return "no"
    return text


def compute_answer_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_answer_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens: return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_gold_supporting_facts(sample: dict) -> set:
    title2sentences = {title: sents for title, sents in sample.get("context", [])}
    gold = set()
    for title, sent_idx in sample.get("supporting_facts", []):
        if title in title2sentences:
            sents = title2sentences[title]
            if 0 <= sent_idx < len(sents):
                gold.add(normalize_answer(sents[sent_idx]))
    return gold


def compute_supporting_fact_metrics(pred_set: set, gold_set: set):
    if not pred_set and not gold_set: return 1, 1.0
    if not pred_set or not gold_set: return 0, 0.0
    em = int(pred_set == gold_set)
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return em, f1


def extract_pred_supporting_facts(retrieved_context: str, gold_supports: set) -> set:
    retrieved_norm = normalize_answer(retrieved_context)
    return {sent for sent in gold_supports if sent in retrieved_norm}


# ─────────────────────────────────────────────
# 文件 I/O
# ─────────────────────────────────────────────
def load_json_safe(file_path: str) -> dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"summary": {}, "detailed_results": [], "metadata": {}, "token_stats": {}}


async def save_batch_results(new_results: list, file_path: str, evaluator: HotpotEvaluator):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = load_json_safe(file_path)
    existing = {r['batch_id']: r for r in data.get('detailed_results', [])}
    for res in new_results:
        existing[res['batch_id']] = res
    data['detailed_results'] = list(existing.values())
    data['summary'] = evaluator.summary()
    data['token_stats'] = token_counter.copy()

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ 评估结果已保存（含 token 统计）")


# ─────────────────────────────────────────────
# 主流程（已迁移 V2 路径）
# ─────────────────────────────────────────────
async def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return

    # === 使用 V2 的路径配置 ===
    full_dataset_path = "preprocess/2wikiMultiHopRAG.json"
    result_output_file = 'preprocess/result_agent/verifecation_2/2wiki_qwen3_8b.json'
    checkpoint_file = "evaluation_checkpoint_MultiHopRAG.json"

    IO_BATCH_SIZE = 10
    temp_results: list = []

    hotpot_evaluator = HotpotEvaluator(checkpoint_file=checkpoint_file)

    start_index = 2357
    TEST_LIMIT = 2376  # 可根据需要修改

    # 断点续跑
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
                start_index = ckpt.get('next_index', start_index)
                print(f"🔄 发现断点：从第 {start_index + 1} 个样本开始")
        except Exception as e:
            print(f"读取断点失败: {e}")

    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    total_samples = len(all_samples)
    print(f"📋 总计加载 {total_samples} 个样本")
    end_index = min(total_samples, TEST_LIMIT)

    for i in range(start_index, end_index):
        batch_num = i + 1
        print(f"\n{'─' * 60}")
        print(f"📦 处理第 {batch_num} 个样本")

        sample = all_samples[i]
        working_dir = f"./2wikimultihop/batch_{batch_num}"  # ← V2 路径

        if not os.path.exists(working_dir):
            print(f"⚠️ 目录 {working_dir} 不存在，跳过")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i + 1}, f)
            continue

        try:
            rag = await initialize_qwen_rag(working_dir)
            if hasattr(rag, 'llm_response_cache'):
                await rag.llm_response_cache.drop()

            query = sample.get("question")
            ground_truth = sample.get("answer") or sample.get("ground_truth")

            tokens_before = snapshot_tokens()
            response = await rag.aquery(query, param=QueryParam(mode="global"))
            tokens_used = diff_tokens(tokens_before)

            print(f"💰 本次 Token: {tokens_used['total_tokens']} (累计: {token_counter['total_tokens']})")

            if isinstance(response, (list, tuple)):
                model_answer = str(response[0]).strip() if len(response) > 0 else ""
                final_context = response[1] if len(response) > 1 else ""
            else:
                model_answer = str(response).strip()
                final_context = "RAG returned a single string"

            # 计算指标
            answer_em = compute_answer_em(model_answer, ground_truth)
            answer_f1 = compute_answer_f1(model_answer, ground_truth)
            gold_sup = extract_gold_supporting_facts(sample)
            pred_sup = extract_pred_supporting_facts(final_context, gold_sup)
            sup_em, sup_f1 = compute_supporting_fact_metrics(pred_sup, gold_sup)

            hotpot_evaluator.update(answer_em, answer_f1, sup_em, sup_f1)

            print(f"❓ 问题: {query}")
            print(f"🎯 真实答案: {ground_truth}")
            print(f"🤖 模型回答: {model_answer}")
            print(f"📐 Answer EM={answer_em} | Answer F1={answer_f1:.3f} | Sup F1={sup_f1:.3f} | Joint F1={(answer_f1 * sup_f1):.3f}")

            current_eval_result = {
                'question': query,
                'batch_id': batch_num,
                'working_dir': working_dir,
                'retrieved_context': final_context,
                'model_answer': model_answer,
                'ground_truth': ground_truth,
                'answer_em': answer_em,
                'answer_f1': answer_f1,
                'sup_f1': sup_f1,
                'joint_f1': answer_f1 * sup_f1,
                'method': 'BLEPRAG-Qwen3-8B',
                'tokens': tokens_used
            }

            temp_results.append(current_eval_result)

            if len(temp_results) >= IO_BATCH_SIZE:
                await save_batch_results(temp_results, result_output_file, hotpot_evaluator)
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({'next_index': i + 1}, f)
                temp_results = []

        except Exception as e:
            print(f"❌ 样本 {batch_num} 失败: {e}")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i}, f)
            break

    if temp_results:
        await save_batch_results(temp_results, result_output_file, hotpot_evaluator)

    print("\n🏁 评估完成！")
    print(json.dumps(hotpot_evaluator.summary(), indent=2, ensure_ascii=False))
    print("\n💰 Token 总消耗:", token_counter)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    asyncio.run(main())