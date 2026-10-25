import json
import os
import asyncio
import re
import time
from typing import Dict, List, Any
import shutil

from BLEPRAG import BLEPRAG, QueryParam
from BLEPRAG.llm.openai import openai_complete_if_cache, openai_embed
from BLEPRAG.kg.shared_storage import initialize_pipeline_status
from BLEPRAG.utils import EmbeddingFunc, setup_logger
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# 设置日志
setup_logger("BLEPRAG", level="INFO")


# --- 1. 配置与指标计算工具 ---

def normalize_text(text: str) -> str:
    """标准文本归一化"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_em(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


class MuSiQueEvaluator:
    """针对 MuSiQue 的评测器，侧重于 Answer 和 Context 召回"""

    def __init__(self, checkpoint_file="musique_eval_metrics.json"):
        self.checkpoint_file = checkpoint_file
        self.reset()
        self.load_metrics()

    def reset(self):
        self.total = 0
        self.ans_em_sum = 0
        self.ans_f1_sum = 0.0
        # MuSiQue 评测中常看检索到的 context 是否覆盖了答案
        self.hit_rate_sum = 0

    def load_metrics(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    self.total = d.get("total", 0)
                    self.ans_em_sum = d.get("ans_em", 0)
                    self.ans_f1_sum = d.get("ans_f1", 0.0)
                    self.hit_rate_sum = d.get("hit_rate", 0)
            except:
                pass

    def save_metrics(self):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total": self.total,
                "ans_em": self.ans_em_sum,
                "ans_f1": self.ans_f1_sum,
                "hit_rate": self.hit_rate_sum
            }, f, indent=2)

    def update(self, pred_ans, gold_ans, retrieved_context):
        self.total += 1
        em = compute_em(pred_ans, gold_ans)
        f1 = compute_f1(pred_ans, gold_ans)

        self.ans_em_sum += em
        self.ans_f1_sum += f1
        # 检查检索到的内容是否包含了金标准答案
        if normalize_text(gold_ans) in normalize_text(retrieved_context):
            self.hit_rate_sum += 1

        self.save_metrics()
        return em, f1

    def summary(self):
        if self.total == 0: return {}
        return {
            "Total": self.total,
            "Answer_EM": self.ans_em_sum / self.total,
            "Answer_F1": self.ans_f1_sum / self.total,
            "Context_Hit_Rate": self.hit_rate_sum / self.total
        }


# --- 2. RAG 初始化函数 (通义千问) ---

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


async def qwen_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """通义千问LLM函数"""
    return await openai_complete_if_cache(
        "qwen3-8b",  # 可选择其他通义千问模型如 qwen-max, qwen-plus qwen-turbo
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
        **kwargs
    )


async def initialize_qwen_rag(WORKING_DIR):
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)
    rag = BLEPRAG(
        working_dir=WORKING_DIR,
        llm_model_func=qwen_llm_model_func,
        chunk_token_size=1200,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=qwen_embedding_func
        )
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# --- 3. 结果保存辅助 ---

def load_json_safe(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"summary": {}, "detailed_results": []}


async def save_results(new_results, file_path, evaluator):
    data = load_json_safe(file_path)
    data['detailed_results'].extend(new_results)
    data['summary'] = evaluator.summary()
    retrieval_times = [
        r.get('retrieval_time_seconds')
        for r in data.get('detailed_results', [])
        if isinstance(r.get('retrieval_time_seconds'), (int, float))
    ]
    query_times = [
        r.get('query_time_seconds')
        for r in data.get('detailed_results', [])
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


# --- 4. 主程序 ---

async def main():
    # 路径配置
    musique_data_path = "./preprocess/musique_ans_v1.0_dev.jsonl"
    result_output_file = "./preprocess/result_agent/flation/musique_withoutbacktracking.json"
    checkpoint_file = "musique_eval_checkpoint1.json"

    # 评测配置
    start_index = 550
    TEST_LIMIT =600# 评测前N条

    evaluator = MuSiQueEvaluator()


    samples = []
    with open(musique_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    end_index = min(len(samples), TEST_LIMIT)
    print(f"📋 任务载入：共 {len(samples)} 条，准备评测 {start_index} 到 {end_index}")

    for i in range(start_index, end_index):
        batch_id = i + 1
        sample = samples[i]
        sample_id = sample.get("id")

        # 指向你在上一阶段生成 KG 的对应目录
        WORKING_DIR = f'./musique_ans/sample_{batch_id}'

        if not os.path.exists(WORKING_DIR):
            print(f"⚠️ 跳过：未发现索引目录 {WORKING_DIR}")
            continue

        try:
            # 初始化并清理缓存
            rag = await initialize_qwen_rag(WORKING_DIR)
            if hasattr(rag, 'llm_response_cache'):
                await rag.llm_response_cache.drop()

            query = sample.get("question")
            gold_answer = sample.get("answer")

            print(f"\n🚀 [{batch_id}/{end_index}] 处理 ID: {sample_id}")
            print(f"Q: {query}")

            # 执行查询 (针对多步推理建议使用 global 模式)
            # 你也可以尝试 hybrid 模式，MuSiQue 的局部细节和全局关系都很重要
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

            # 解析结果
            if isinstance(response, (list, tuple)):
                model_answer = str(response[0]).strip()
                retrieved_context = str(response[1]) if len(response) > 1 else ""
            else:
                model_answer = str(response).strip()
                retrieved_context = ""

            # 评估指标更新
            em, f1 = evaluator.update(model_answer, gold_answer, retrieved_context)

            print(f"🤖 A: {model_answer}")
            print(f"🎯 Gold: {gold_answer}")
            print(f"📐 Metrics | EM: {em} | F1: {f1:.3f}")

            # 实时保存详细结果
            print(
                f"Retrieval time={retrieval_time_seconds:.3f}s | "
                f"Query time={query_time_seconds:.3f}s"
            )

            current_res = {
                "batch_id": batch_id,
                "sample_id": sample_id,
                "question": query,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "em": em,
                "f1": f1,
                "retrieval_time_seconds": retrieval_time_seconds,
                "query_time_seconds": query_time_seconds,
                "retrieved_context": retrieved_context[:500] + "..."  # 截断保存
            }
            await save_results([current_res], result_output_file, evaluator)

            # 更新断点文件
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i + 1}, f)

        except Exception as e:
            print(f"❌ 样本 {batch_id} 评测出错: {e}")
            # 如果发生严重错误，记录断点并停止，以便排查
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'next_index': i}, f)
            break

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("\n🎉 MuSiQue 评测任务圆满结束")


if __name__ == "__main__":
    asyncio.run(main())
