
import re

from BLEPRAG import BLEPRAG, QueryParam
from BLEPRAG.llm.openai import openai_complete_if_cache, openai_embed
from BLEPRAG.kg.shared_storage import initialize_pipeline_status
from BLEPRAG.utils import EmbeddingFunc, setup_logger
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union
from openai import AsyncOpenAI

load_dotenv()  



setup_logger("BLEPRAG", level="INFO")
async def qwen_llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=True, **kwargs
) -> str:
    """通义千问LLM函数"""
    return await openai_complete_if_cache(
        "qwen-",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs
    )


async def qwen_embedding_func(texts: list[str]):
    return await openai_embed(
        texts,
        model="text-embedding-v4",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
async def qwen_embedding_func(texts: list[str]):

    batch_size = 10
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = await openai_embed(
            batch,
            model="text-embedding-v4",  
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        results.extend(resp)

    return results


async def initialize_qwen_rag(WORKING_DIR):

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = BLEPRAG(
        working_dir=WORKING_DIR,
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


def load_ground_truth_data(file_path: str) -> Dict[str, Dict[str, str]]:

    ground_truth_data = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            objects = []
 
            start_index = 0
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        # 当括号计数归零时，找到一个完整的JSON对象
                        if brace_count == 0:
                            obj_str = content[start_index:i + 1]
                            objects.append(obj_str)
                            start_index = i + 1
                            # 跳过逗号和空白
                            while start_index < len(content) and content[start_index] in [',', ' ', '\n', '\t']:
                                start_index += 1

            print(f"找到 {len(objects)} 个JSON对象")

            valid_objects_count = 0
            for i, obj_str in enumerate(objects, 1):
                try:

                    item = json.loads(obj_str)
                    question = item.get("question", "")
                    answer = item.get("answer", "")

                    if question and answer:
                        ground_truth_data[question] = {
                            "ground_truth": answer,
                            "raw_data": item
                        }
                        valid_objects_count += 1
                        print(f"✅ 成功解析第 {valid_objects_count} 个样本: {question}")
                    else:
                        print(f"❌ 第 {i} 个对象缺少问题或答案")

                except json.JSONDecodeError as e:
                    print(f"❌ 第 {i} 个对象JSON解析失败: {e}")
                    print(f"   对象内容: {obj_str[:100]}...")

            print(f"\n🎯 最终结果: 成功加载 {valid_objects_count} 个样本")

            # 显示所有加载的问题
            if ground_truth_data:
                print("\n📋 加载的问题列表:")
                for j, (q, info) in enumerate(ground_truth_data.items(), 1):
                    print(f"  {j}. {q}")
                    print(f"     答案: {info['ground_truth']}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"❌ 加载真实答案数据时出错: {e}")

    return ground_truth_data

def validate_ground_truth_coverage(queries: List[str], ground_truth_data: Dict[str, Dict[str, str]]):

    missing_questions = []

    for query in queries:
        if query not in ground_truth_data:
            missing_questions.append(query)

    if missing_questions:
        print("\n⚠️  警告: 以下问题在真实答案数据集中找不到:")
        for i, question in enumerate(missing_questions, 1):
            print(f"  {i}. {question}")
        print(f"\n总计 {len(missing_questions)}/{len(queries)} 个问题缺少真实答案")
    else:
        print("✅ ")


def update_file_paths(input_file: str, output_file: str):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 正则匹配 新闻来源: 后的链接
    pattern = re.compile(r"新闻来源:(\S+)")

    for key, item in data.items():
        content = item.get("content", "")
        match = pattern.search(content)
        if match:
            source_link = match.group(1).strip()
            item["file_path"] = source_link  # 替换 file_path

    # 写入新 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_samples_in_batches(file_path, batch_size=3):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            global_sample_index = 1  # 全局样本编号，从1开始

            while True:
                batch = []
                current_sample = []
                batch_start_index = global_sample_index  # 记录当前批次的起始编号

                # 读取一批样本
                while len(batch) < batch_size:
                    char = file.read(1)
                    if not char:  # 文件结束
                        # 如果还有未完成的样本，添加到批次中
                        if current_sample:
                            batch.append({
                                'content': ''.join(current_sample),
                                'index': global_sample_index
                            })
                            global_sample_index += 1
                        break

                    current_sample.append(char)

                    # 检查是否遇到了样本结束标志
                    if char == '}':
                        batch.append({
                            'content': ''.join(current_sample),
                            'index': global_sample_index
                        })
                        global_sample_index += 1
                        current_sample = []  # 重置当前样本缓冲区


                if not batch:
                    break

                # 返回当前批次的样本和起始编号
                yield batch, batch_start_index

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return


def save_checkpoint(file_path, batch_number, next_sample_index, all_ground_truth, all_queries):

    checkpoint_data = {
        'batch_number': batch_number,
        'next_sample_index': next_sample_index,
        'all_ground_truth': all_ground_truth,
        'all_queries': all_queries,
        'timestamp': str(asyncio.get_event_loop().time()) 
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def flatten_context(context):
    """
    context: List[[entity_name, List[str]]]
    return: str
    """
    blocks = []

    for item in context:
        if not isinstance(item, list) or len(item) != 2:
            continue

        entity, sentences = item

        if not isinstance(sentences, list):
            continue

        block = entity + "\n" + "\n".join(s.strip() for s in sentences if s.strip())
        blocks.append(block)

    return "\n\n".join(blocks)

import os
import json
import shutil
import asyncio

async def main():



    input_source_file = "preprocess/测试数据_hotpota_full.json"
    checkpoint_file = "processing_checkpoint_full.json"
    save_interval = 1  

    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):

        return




    start_index = 2500
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                start_index = checkpoint.get('next_index', 0)
                print(f"🔄 {start_index + 1} ")
        except Exception as e:
            print(f"{e}")

    # --- 读取总数据集 ---
    try:
        with open(input_source_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
    except Exception as e:

        return

    total_samples = len(all_samples)


    # --- 逐个样本处理 ---
    try:
        for i in range(start_index, total_samples):
            batch_num = i + 1
            sample = all_samples[i]


            WORKING_DIR = f'./hot_kg_full/batch_{batch_num}'
            input_file = f'{WORKING_DIR}/kv_store_text_chunks.json'
            output_file = f'{WORKING_DIR}/kv_store_text_chunks.json'
            os.makedirs(WORKING_DIR, exist_ok=True)

            try:

                rag = await initialize_qwen_rag(WORKING_DIR)

                raw_content = sample.get("context")
                if isinstance(raw_content, list):
                    content = flatten_context(raw_content)
                elif isinstance(raw_content, str):
                    content = raw_content.strip()
                else:
                    raise ValueError(f"unknown content: {type(raw_content)}")

                if not content:
                    continue

                await rag.ainsert(
                    content,
                    split_by_character="}",
                    split_by_character_only=False
                )

                update_file_paths(input_file, output_file)


                if (i + 1) % save_interval == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump({'next_index': i + 1}, f)

            except Exception as e:
                # 删除当前未完成样本的目录
                if os.path.exists(WORKING_DIR):
                    shutil.rmtree(WORKING_DIR)
                # 保存断点，当前样本未完成，下次从它重新开始
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({'next_index': i}, f)
                break

    except KeyboardInterrupt:
        # 删除当前未完成样本的目录
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'next_index': i}, f)
        raise

    else:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)


if __name__ == "__main__":

    print("=" * 50)
    asyncio.run(main())