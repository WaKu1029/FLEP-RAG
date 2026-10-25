import json
import os
import asyncio
import re
import time
from typing import Dict, List, Any, Optional, Union

from BLEPRAG import BLEPRAG, QueryParam
from BLEPRAG.llm.openai import openai_embed
from BLEPRAG.kg.shared_storage import initialize_pipeline_status
from BLEPRAG.utils import EmbeddingFunc, setup_logger
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
setup_logger("BLEPRAG", level="INFO")

WORKING_DIR = "./batch_full/qwen_batch16"

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
        "prompt_tokens":     token_counter["prompt_tokens"]     - before["prompt_tokens"],
        "completion_tokens": token_counter["completion_tokens"] - before["completion_tokens"],
        "total_tokens":      token_counter["total_tokens"]      - before["total_tokens"],
    }

_llm_client: Optional[AsyncOpenAI] = None

def get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return _llm_client


async def qwen_llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
) -> str:

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
        token_counter["prompt_tokens"]     += response.usage.prompt_tokens
        token_counter["completion_tokens"] += response.usage.completion_tokens
        token_counter["total_tokens"]      += response.usage.total_tokens

    return response.choices[0].message.content


async def qwen_embedding_func(texts: list[str]):
    """通义千问 Embedding，自动分批（每批最多 10 条）"""
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
# HotpotQA 评估器（含断点续跑）
# ─────────────────────────────────────────────
class HotpotEvaluator:
    """严格对齐 HotpotQA 官方 6 指标，支持断点续跑"""

    def __init__(self, checkpoint_file="evaluation_checkpoint.json"):
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
                    self.total         = d.get("total", 0)
                    self.answer_em     = d.get("answer_em", 0)
                    self.answer_f1_sum = d.get("answer_f1_sum", 0.0)
                    self.sup_em        = d.get("sup_em", 0)
                    self.sup_f1_sum    = d.get("sup_f1_sum", 0.0)
                    self.joint_em      = d.get("joint_em", 0)
                    self.joint_f1_sum  = d.get("joint_f1_sum", 0.0)
            except Exception as e:
                print(f"加载评估指标失败: {e}")

    def save_metrics(self):
        data = {
            "total":         self.total,
            "answer_em":     self.answer_em,
            "answer_f1_sum": self.answer_f1_sum,
            "sup_em":        self.sup_em,
            "sup_f1_sum":    self.sup_f1_sum,
            "joint_em":      self.joint_em,
            "joint_f1_sum":  self.joint_f1_sum,
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, answer_em: int, answer_f1: float, sup_em: int, sup_f1: float):
        self.total         += 1
        self.answer_em     += answer_em
        self.answer_f1_sum += answer_f1
        self.sup_em        += sup_em
        self.sup_f1_sum    += sup_f1

        joint_em  = int(answer_em == 1 and sup_em == 1)
        joint_f1  = answer_f1 * sup_f1 if sup_f1 > 0 else 0.0
        self.joint_em      += joint_em
        self.joint_f1_sum  += joint_f1

        self.save_metrics()

    def summary(self) -> Dict[str, Any]:
        if self.total == 0:
            return {}
        return {
            "Total":                  self.total,
            "Answer_EM":              self.answer_em     / self.total,
            "Answer_F1":              self.answer_f1_sum / self.total,
            "SupportingFact_EM":      self.sup_em        / self.total,
            "SupportingFact_F1":      self.sup_f1_sum    / self.total,
            "Joint_EM":               self.joint_em      / self.total,
            "Joint_F1":               self.joint_f1_sum  / self.total,
        }


# ─────────────────────────────────────────────
# 评估辅助函数
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
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
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
    if not pred_set and not gold_set:
        return 1, 1.0
    if not pred_set or not gold_set:
        return 0, 0.0
    em = int(pred_set == gold_set)
    common    = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall    = len(common) / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
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
        except Exception:
            pass
    return {"summary": {}, "detailed_results": [], "metadata": {}, "token_stats": {}}


async def save_batch_results(new_results: list, file_path: str, evaluator: HotpotEvaluator):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = load_json_safe(file_path)

    existing = {r['batch_id']: r for r in data.get('detailed_results', [])}
    for res in new_results:
        existing[res['batch_id']] = res
    data['detailed_results'] = list(existing.values())
    data['summary']           = evaluator.summary()

    # 同步保存全局 token 累计统计
    data['token_stats'] = {
        "total_prompt_tokens":     token_counter["prompt_tokens"],
        "total_completion_tokens": token_counter["completion_tokens"],
        "total_tokens":            token_counter["total_tokens"],
    }
    completed_results = data.get('detailed_results', [])
    retrieval_times = [
        r.get('retrieval_time_seconds')
        for r in completed_results
        if isinstance(r.get('retrieval_time_seconds'), (int, float))
    ]
    query_times = [
        r.get('query_time_seconds')
        for r in completed_results
        if isinstance(r.get('query_time_seconds'), (int, float))
    ]
    data['time_stats'] = {
        "total_retrieval_time_seconds": sum(retrieval_times),
        "avg_retrieval_time_seconds": (
            sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        ),
        "total_query_time_seconds": sum(query_times),
        "avg_query_time_seconds": (
            sum(query_times) / len(query_times) if query_times else 0
        ),
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ 评估结果已保存（含 token 统计）")


async def remove_sample_from_result(batch_id: int, file_path: str, evaluator: HotpotEvaluator):
    if not os.path.exists(file_path):
        return
    data = load_json_safe(file_path)
    updated = [r for r in data.get('detailed_results', []) if r.get('batch_id') != batch_id]
    if len(updated) < len(data.get('detailed_results', [])):
        data['detailed_results'] = updated
        data['summary']          = evaluator.summary()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"🗑️ 已移除 Batch {batch_id} 的不完整记录")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
async def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return

    full_dataset_path  = "preprocess/测试数据_hotpota_full.json"
    result_output_file = "preprocess/result_agent/flation/full_qwen3_8b_full_reretrieve.json"
    checkpoint_file    = "evaluation_checkpoint.json"
    IO_BATCH_SIZE      = 1
    TEST_LIMIT         = 100
    temp_results: list = []

    hotpot_evaluator = HotpotEvaluator(checkpoint_file=checkpoint_file)

    start_index = 50
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
                start_index = ckpt.get('next_index', start_index)
                print(f"🔄 发现断点：从第 {start_index + 1} 个样本开始执行")
        except Exception as e:
            print(f"⚠️ 读取断点文件失败: {e}，将从默认位置开始")
    else:
        print("🆕 未检测到断点文件，初始化评估指标为 0...")
        for attr in ["total", "answer_em", "answer_f1_sum",
                     "sup_em", "sup_f1_sum", "joint_em", "joint_f1_sum"]:
            setattr(hotpot_evaluator, attr, 0 if "sum" not in attr and attr != "total" else 0.0
                    if "sum" in attr else 0)
        hotpot_evaluator.save_metrics()

    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    total_samples = len(all_samples)
    print(f"📋 总计加载 {total_samples} 个样本")
    end_index = min(total_samples, TEST_LIMIT)

    for i in range(start_index, end_index):
        batch_num = i + 1
        print(f"\n{'─' * 50}")
        print(f"📦 当前处理第 {batch_num} 个样本")

        sample      = all_samples[i]
        working_dir = f"./hot_kg_full/batch_{batch_num}"

        if not os.path.exists(working_dir):
            print(f"⚠️ 目录 {working_dir} 不存在，跳过")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i + 1}, f)
            continue


        try:
            rag = await initialize_qwen_rag(working_dir)

            if hasattr(rag, 'llm_response_cache'):
                await rag.llm_response_cache.drop()

            query        = sample.get("question")
            ground_truth = sample.get("answer") or sample.get("ground_truth")
            print(f"❓ 问题: {query}")

            tokens_before = snapshot_tokens()

            query_param = QueryParam(mode="global", backtrack_mode="full_reretrieve")
            query_start_time = time.perf_counter()
            response = await rag.aquery(
                query,
                param=query_param,
            )
            query_time_seconds = time.perf_counter() - query_start_time
            retrieval_time_seconds = getattr(
                query_param,
                "retrieval_time_seconds",
                query_time_seconds,
            )

            tokens_used = diff_tokens(tokens_before)

            print(
                f"💰 本次查询 Token 消耗 | "
                f"Prompt={tokens_used['prompt_tokens']} | "
                f"Completion={tokens_used['completion_tokens']} | "
                f"Total={tokens_used['total_tokens']}"
            )
            print(
                f"📊 累计 Token 消耗 | "
                f"Prompt={token_counter['prompt_tokens']} | "
                f"Completion={token_counter['completion_tokens']} | "
                f"Total={token_counter['total_tokens']}"
            )

            print(
                f"Retrieval time={retrieval_time_seconds:.3f}s | "
                f"Query time={query_time_seconds:.3f}s"
            )

            if isinstance(response, (list, tuple)):
                model_answer  = str(response[0]).strip() if len(response) > 0 else ""
                final_context = response[1] if len(response) > 1 else "No context returned"
            else:
                model_answer  = str(response).strip()
                final_context = "RAG returned a single string"

            # ── 计算 HotpotQA 指标 ──
            answer_em  = compute_answer_em(model_answer, ground_truth)
            answer_f1  = compute_answer_f1(model_answer, ground_truth)
            gold_sup   = extract_gold_supporting_facts(sample)
            pred_sup   = extract_pred_supporting_facts(final_context, gold_sup)
            sup_em, sup_f1 = compute_supporting_fact_metrics(pred_sup, gold_sup)

            hotpot_evaluator.update(
                answer_em=answer_em,
                answer_f1=answer_f1,
                sup_em=sup_em,
                sup_f1=sup_f1
            )

            print(f"🎯 真实答案: {ground_truth}")
            print(f"🤖 模型回答: {model_answer}")
            print(
                f"📐 Hotpot Metrics | "
                f"Answer EM={answer_em} | "
                f"Answer F1={answer_f1:.3f} | "
                f"Sup F1={sup_f1:.3f} | "
                f"Joint F1={(answer_f1 * sup_f1):.3f}"
            )

            current_eval_result = {
                'question':         query,
                'batch_id':         batch_num,
                'working_dir':      working_dir,
                'retrieved_context': final_context,
                'model_answer':     model_answer,
                'ground_truth':     ground_truth,
                'method':           'Hybrid+LLM',
                'F1':               answer_f1,
                'EM':               answer_em,
                'retrieval_time_seconds': retrieval_time_seconds,
                'query_time_seconds': query_time_seconds,

                'tokens': {
                    'prompt_tokens':     tokens_used['prompt_tokens'],
                    'completion_tokens': tokens_used['completion_tokens'],
                    'total_tokens':      tokens_used['total_tokens'],
                }
            }
            temp_results.append(current_eval_result)

            if len(temp_results) >= IO_BATCH_SIZE:
                await save_batch_results(temp_results, result_output_file, hotpot_evaluator)
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({'next_index': i + 1}, f)
                temp_results = []

        except Exception as e:
            print(f"❌ 样本 {batch_num} 评估失败: {e}")
            await remove_sample_from_result(batch_num, result_output_file, hotpot_evaluator)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i}, f)
            break

    if temp_results:
        await save_batch_results(temp_results, result_output_file, hotpot_evaluator)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'next_index': end_index}, f)
        print(f"✅ 剩余的 {len(temp_results)} 条结果已保存")

    print("\n" + "=" * 60)
    print("🏁 评估完成，最终汇总：")
    summary = hotpot_evaluator.summary()
    if summary:
        avg_f1 = summary.get("Answer_F1", 0.0)
        avg_em = summary.get("Answer_EM", 0.0)
        print(f"   ✅ F1={avg_f1:.4f}  |  EM={avg_em:.4f}")
    else:
        print("   ⚠️ 未处理任何样本")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n💰 全局 Token 消耗汇总：")
    print(json.dumps({
        "total_prompt_tokens":     token_counter["prompt_tokens"],
        "total_completion_tokens": token_counter["completion_tokens"],
        "total_tokens":            token_counter["total_tokens"],
        "evaluated_samples":       hotpot_evaluator.total,
        "avg_tokens_per_sample":   (
            token_counter["total_tokens"] / hotpot_evaluator.total
            if hotpot_evaluator.total > 0 else 0
        )
    }, indent=2))

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🎉 断点文件已删除")


if __name__ == "__main__":
    print("通义千问 BLEPRAG 评估（含 Token 计数）")
    print("请确保已设置 DASHSCOPE_API_KEY 环境变量")
    print("=" * 60)
    asyncio.run(main())
