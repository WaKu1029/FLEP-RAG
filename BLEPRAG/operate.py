from __future__ import annotations
from functools import partial
from collections import deque
import asyncio
import json
import re
import os
import sqlite3
from typing import Any, AsyncIterator
from collections import Counter, defaultdict
from prompts import PROMPTS
from .prompt_2 import GRAPH_FIELD_SEP
from transformers.models.tapas.tokenization_tapas import Question


from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    Tokenizer,
    is_float_regex,
    normalize_extracted_info,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
    use_llm_func_with_cache,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)

import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def chunking_by_token_size(
        tokenizer: Tokenizer,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > max_token_size:
                    for start in range(
                            0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start: start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
                range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start: start + max_token_size])
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
        entity_or_relation_name: str,
        description: str,
        global_config: dict,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    tokenizer: Tokenizer = global_config["tokenizer"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    summary_max_tokens = global_config["summary_to_max_tokens"]

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = tokenizer.encode(description)

    ### summarize is not determined here anymore (It's determined by num_fragment now)
    # if len(tokens) < summary_max_tokens:  # No need for summary
    #     return description

    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = tokenizer.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")

    # Use LLM function with cache (higher priority for summary generation)
    summary = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        max_tokens=summary_max_tokens,
        cache_type="extract",
    )
    return summary


async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
        file_path: str = "unknown_source",
):
    if len(record_attributes) < 4 or '"entity"' not in record_attributes[0]:
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip()
    if not entity_name:
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Normalize entity name
    entity_name = normalize_extracted_info(entity_name, is_entity=True)

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3])
    entity_description = normalize_extracted_info(entity_description)

    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )


async def _handle_single_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
        file_path: str = "unknown_source",
):
    if len(record_attributes) < 5 or '"relationship"' not in record_attributes[0]:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    # Normalize source and target entity names
    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)
    if source == target:
        logger.debug(
            f"Relationship source and target are the same in: {record_attributes}"
        )
        return None

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[4]), is_entity=True
    )
    edge_keywords = edge_keywords.replace("，", ",")

    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"').strip("'"))
        if is_float_regex(record_attributes[-1].strip('"').strip("'"))
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )


async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        global_config: dict,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_file_paths.extend(
            split_string_by_multi_markers(already_node["file_path"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    file_path = GRAPH_FIELD_SEP.join(
        set([dp["file_path"] for dp in nodes_data] + already_file_paths)
    )

    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    num_fragment = description.count(GRAPH_FIELD_SEP) + 1
    num_new_fragment = len(set([dp["description"] for dp in nodes_data]))

    if num_fragment > 1:
        if num_fragment >= force_llm_summary_on_merge:
            status_message = f"LLM merge N: {entity_name} | {num_new_fragment}+{num_fragment - num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            description = await _handle_entity_relation_summary(
                entity_name,
                description,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
        else:
            status_message = f"Merge N: {entity_name} | {num_new_fragment}+{num_fragment - num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)

    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        global_config: dict,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
):
    if src_id == tgt_id:
        return None

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    split_string_by_multi_markers(
                        already_edge["file_path"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(
            set(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
            )
        )
    )

    # Split all existing and new keywords into individual terms, then combine and deduplicate
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )
    file_path = GRAPH_FIELD_SEP.join(
        set(
            [dp["file_path"] for dp in edges_data if dp.get("file_path")]
            + already_file_paths
        )
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):

            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "entity_id": need_insert_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                    "file_path": file_path,
                    "created_at": int(time.time()),
                },
            )

    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    num_fragment = description.count(GRAPH_FIELD_SEP) + 1
    num_new_fragment = len(
        set([dp["description"] for dp in edges_data if dp.get("description")])
    )

    if num_fragment > 1:
        if num_fragment >= force_llm_summary_on_merge:
            status_message = f"LLM merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment - num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            description = await _handle_entity_relation_summary(
                f"({src_id}, {tgt_id})",
                description,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
        else:
            status_message = f"Merge E: {src_id} - {tgt_id} | {num_new_fragment}+{num_fragment - num_new_fragment}"
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=int(time.time()),
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
    )

    return edge_data


async def merge_nodes_and_edges(
        chunk_results: list,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        current_file_number: int = 0,
        total_files: int = 0,
        file_path: str = "unknown_source",
) -> None:
    """Merge nodes and edges from extraction results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
    """
    # Get lock manager from shared storage
    from .kg.shared_storage import get_graph_db_lock

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    # Centralized processing of all nodes and edges
    entities_data = []
    relationships_data = []

    # Merge nodes and edges
    # Use graph database lock to ensure atomic merges and updates
    graph_db_lock = get_graph_db_lock(enable_logging=False)
    async with graph_db_lock:
        async with pipeline_status_lock:
            log_message = (
                f"Merging stage {current_file_number}/{total_files}: {file_path}"
            )
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        # Process and update all entities at once
        for entity_name, entities in all_nodes.items():
            entity_data = await _merge_nodes_then_upsert(
                entity_name,
                entities,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            entities_data.append(entity_data)

        # Process and update all relationships at once
        for edge_key, edges in all_edges.items():
            edge_data = await _merge_edges_then_upsert(
                edge_key[0],
                edge_key[1],
                edges,
                knowledge_graph_inst,
                global_config,
                pipeline_status,
                pipeline_status_lock,
                llm_response_cache,
            )
            if edge_data is not None:
                relationships_data.append(edge_data)

        # Update total counts
        total_entities_count = len(entities_data)
        total_relations_count = len(relationships_data)

        log_message = f"Updating {total_entities_count} entities  {current_file_number}/{total_files}: {file_path}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Update vector databases with all collected data
        if entity_vdb is not None and entities_data:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "entity_name": dp["entity_name"],
                    "entity_type": dp["entity_type"],
                    "content": f"{dp['entity_name']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in entities_data
            }
            await entity_vdb.upsert(data_for_vdb)

        log_message = f"Updating {total_relations_count} relations {current_file_number}/{total_files}: {file_path}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        if relationships_vdb is not None and relationships_data:
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "keywords": dp["keywords"],
                    "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                    "source_id": dp["source_id"],
                    "file_path": dp.get("file_path", "unknown_source"),
                }
                for dp in relationships_data
            }
            await relationships_vdb.upsert(data_for_vdb)


async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        global_config: dict[str, str],
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
) -> list:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        # examples = "\n".join(
        #     PROMPTS["entity_extraction_examples"][: int(example_number)]
        examples = "\n\n".join(PROMPTS["entity_extraction_examples"])

    else:
        # examples = "\n".join(PROMPTS["entity_extraction_examples"])
        examples = PROMPTS["entity_extraction_examples"]

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_extraction_result(
            result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        """Process a single extraction result (either initial or gleaning)
        Args:
            result (str): The extraction result to process
            chunk_key (str): The chunk key for source tracking
            file_path (str): The file path for citation
        Returns:
            tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Get initial extraction
        '''New Get initial extraction'''
        hint_prompt = entity_extract_prompt.format(
            **{**context_base, "input_text": content}
        )

        try:
            final_result = await use_llm_func_with_cache(
                hint_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                cache_type="extract",
            )
        except Exception as e:
            if "data_inspection_failed" in str(e):
                logger.warning(f"跳过该分片（触发内容过滤）: {content[:]}...")
                return {}, {}  # 返回空结果，避免整个 pipeline 崩溃
            raise

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        '''New Process initial extraction with file path'''
        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        # Process additional gleaning results
        # 后续 gleaning 循环
        for now_glean_index in range(entity_extract_max_gleaning):
            try:
                glean_result = await use_llm_func_with_cache(
                    continue_prompt,
                    use_llm_func,
                    llm_response_cache=llm_response_cache,
                    history_messages=history,
                    cache_type="extract",
                )
            except Exception as e:
                if "data_inspection_failed" in str(e):
                    logger.warning(f"跳过该分片（glean触发过滤）: {content[:100]}...")
                    break  # 直接结束 gleaning 循环
                raise

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key, file_path
            )

            # Merge results - only add entities and edges with new names
            for entity_name, entities in glean_nodes.items():
                if (
                        entity_name not in maybe_nodes
                ):  # Only accetp entities with new name in gleaning stage
                    maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                if (
                        edge_key not in maybe_edges
                ):  # Only accetp edges with new name in gleaning stage
                    maybe_edges[edge_key].extend(edges)

            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func_with_cache(
                if_loop_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    llm_model_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(llm_model_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            return await _process_single_content(chunk)

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for all tasks to complete, collecting exceptions
    try:
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        # Cancel all remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for all tasks to be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Re-raise the exception
        raise e

    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results






from collections import deque


from collections import deque
from typing import Tuple


async def link_entities_from_kg(question_analysis: dict, entities_vdb: BaseVectorStorage, query_param: QueryParam) -> \
Tuple[deque, str]:
    """链接实体并返回:
    1) deque[(linked_entity, original_triple)]
    2) 合并后的上下文字符串
    """

    result_queue = deque()
    sentences_to_combine = []

    triples_list = question_analysis.get("triples", [])

    # ========= 1. 获取向量库实体 =========
    all_entity_raw = await entities_vdb.client_storage
    raw_data = all_entity_raw.get("data", []) if isinstance(all_entity_raw, dict) else []

    # ========= 2. 安全文本提取函数（核心防炸点） =========
    def extract_text_safe(content_obj) -> str:
        """将任意类型 content 安全转换为 string，永远不会返回 dict / list / iterator"""
        if content_obj is None:
            return ""
        if isinstance(content_obj, str):
            return content_obj.strip()
        if isinstance(content_obj, dict):
            # 常见字段优先
            for key in ("text", "content", "description", "summary"):
                val = content_obj.get(key)
                if isinstance(val, str):
                    return val.strip()
            # fallback：整个 dict 字符串化
            return str(content_obj)
        if isinstance(content_obj, (list, tuple, set)):
            return " ".join(str(x) for x in content_obj if x)
        return str(content_obj)

    # ========= 3. 构建实体索引 =========
    existing_entity_map = {
        item.get("entity_name"): item
        for item in raw_data
        if item.get("entity_name")
    }
    existing_entity_map_lc = {k.lower(): k for k in existing_entity_map.keys()}

    # ========= 4. 遍历三元组 =========
    for triple in triples_list:
        check_slots = [
            triple.get("subject", ""),
            triple.get("object", "")
        ]

        for entity_name in check_slots:
            if not entity_name or "unknown" in str(entity_name).lower():
                continue

            entity_str = str(entity_name).strip()
            entity_lc = entity_str.lower()

            # ===== 精准匹配 =====
            if entity_lc in existing_entity_map_lc:
                linked_name = existing_entity_map_lc[entity_lc]
                result_queue.append((linked_name, triple))

                # 提取上下文
                content = existing_entity_map[linked_name].get("content")
                safe_text = extract_text_safe(content)
                if safe_text:
                    sentences_to_combine.append(safe_text)
                print(f"实体 {entity_lc} 完成精确匹配")
            else:

                # ===== 向量相似度匹配 =====
                raw_results = await entities_vdb.query(
                    query=entity_str, top_k=query_param.top_k
                )
                results = list(raw_results) if raw_results else []
                top_candidates = results[:2]

                for res in top_candidates:
                    linked_name = res.get("entity_name")
                    if not linked_name:
                        continue

                    result_queue.append((linked_name, triple))

                    # 提取上下文
                    target_entity = existing_entity_map.get(linked_name)
                    if target_entity:
                        content = res.get("content")
                        safe_text = extract_text_safe(content)
                        if safe_text:
                            sentences_to_combine.append(safe_text)

                    print(f"实体 {entity_lc} 链接到候选 {linked_name}")

    # ========= 5. 合并上下文 =========
    combined_sentence = " ".join(
        s if isinstance(s, str) else str(s)
        for s in sentences_to_combine
        if s
    )

    return result_queue, combined_sentence

# async def link_entities_from_kg(
#         question_analysis: dict,
#         entities_vdb: BaseVectorStorage,
#         query_param: QueryParam
# ) -> Tuple[deque, str]:
#     """
#     链接实体并返回:
#     1) deque[(linked_entity, original_triple)]
#     2) 合并后的上下文字符串
#     """
#
#     result_queue = deque()
#     sentences_to_combine = []
#
#     triples_list = question_analysis.get("triples", [])
#
#     # ========= 1. 获取向量库实体 =========
#     all_entity_raw = await entities_vdb.client_storage
#     raw_data = all_entity_raw.get("data", []) if isinstance(all_entity_raw, dict) else []
#
#     # ========= 2. 安全文本提取函数（核心防炸点） =========
#     def extract_text_safe(content_obj) -> str:
#         """
#         将任意类型 content 安全转换为 string
#         永远不会返回 dict / list / iterator
#         """
#         if content_obj is None:
#             return ""
#
#         if isinstance(content_obj, str):
#             return content_obj.strip()
#
#         if isinstance(content_obj, dict):
#             # 常见字段优先
#             for key in ("text", "content", "description", "summary"):
#                 val = content_obj.get(key)
#                 if isinstance(val, str):
#                     return val.strip()
#             # fallback：整个 dict 字符串化
#             return str(content_obj)
#
#         if isinstance(content_obj, (list, tuple, set)):
#             return " ".join(str(x) for x in content_obj if x)
#
#         # 兜底
#         return str(content_obj)
#
#     # ========= 3. 构建实体索引 =========
#     existing_entity_map = {
#         item.get("entity_name"): item
#         for item in raw_data
#         if item.get("entity_name")
#     }
#     existing_entity_map_lc = {k.lower(): k for k in existing_entity_map.keys()}
#
#     # ========= 4. 遍历三元组 =========
#     for triple in triples_list:
#
#         check_slots = [
#             triple.get("subject", ""),
#             triple.get("object", "")
#         ]
#
#         for entity_name in check_slots:
#             if not entity_name or "unknown" in str(entity_name).lower():
#                 continue
#
#             entity_str = str(entity_name).strip()
#             entity_lc = entity_str.lower()
#
#             # ===== 策略 A：精确匹配 =====
#             if entity_lc in existing_entity_map_lc:
#                 linked_name = existing_entity_map_lc[entity_lc]
#                 result_queue.append((linked_name, triple))
#
#                 content = existing_entity_map[linked_name].get("content")
#                 safe_text = extract_text_safe(content)
#                 if safe_text:
#                     sentences_to_combine.append(safe_text)
#
#                 print(f"实体 {entity_lc} 完成精确匹配")
#
#             # ===== 策略 B：向量相似度匹配 =====
#             else:
#                 raw_results = await entities_vdb.query(
#                     query=entity_str,
#                     top_k=query_param.top_k
#                 )
#
#                 # 强制 list，防 iterator / generator
#                 results = list(raw_results) if raw_results else []
#                 top_candidates = results[:3]
#
#                 for res in top_candidates:
#                     linked_name = res.get("entity_name")
#                     if not linked_name:
#                         continue
#
#                     result_queue.append((linked_name, triple))
#
#                     target_entity = existing_entity_map.get(linked_name)
#                     if target_entity:
#                         content = res.get("content")
#                         safe_text = extract_text_safe(content)
#                         if safe_text:
#                             sentences_to_combine.append(safe_text)
#
#                     print(f"实体 {entity_lc} 链接到候选 {linked_name}")
#
#     # ========= 5. 终极防炸 join（最后一道保险） =========
#     combined_sentence = " ".join(
#         s if isinstance(s, str) else str(s)
#         for s in sentences_to_combine
#         if s
#     )
#
#     return result_queue, combined_sentence
async def verify_relationships(
        matched_queue: deque,
        relationships_vdb: BaseVectorStorage
) -> list:
    """
    从队列中取出 (实体, 原始三元组)，并在关系库中校验 predicate 是否一致
    """
    final_verified_results = []

    for linked_entity, original_triple in matched_queue:
        # 1. 弹出队列中的数据
        #linked_entity, original_triple = matched_queue.popleft()
        target_predicate = original_triple.get('predicate')

        print(f"🔍 正在校验实体: '{linked_entity}' 是否具有关系: '{target_predicate}'")

        # 2. 获取关系库底层数据 (参考 NanoVectorDBStorage 逻辑)
        # _get_client 确保多进程环境下数据是最新的
        client = await relationships_vdb._get_client()
        storage = getattr(client, "_NanoVectorDB__storage")
        all_relations = storage.get("data", [])

        # 3. 遍历关系库进行精准匹配
        for record in all_relations:
            # 检查实体是否匹配 (作为源或目标)
            if record.get("src_id") == linked_entity or record.get("tgt_id") == linked_entity:
                content = record.get("content", "")

                # 按照格式解析 content: 头实体\t尾实体\n关系\nsentence
                parts = content.split('\n')
                if len(parts) >= 2:
                    actual_predicate = parts[1].strip()

                    # 4. 核心校验：库中的关系是否等于原始三元组的关系
                    if actual_predicate == target_predicate:
                        print(f"  ✅ 匹配成功！在库中找到关系 '{actual_predicate}'")

                        # 保存验证通过的详细信息
                        final_verified_results.append({
                            "entity": linked_entity,
                            "predicate": actual_predicate,
                            "original_triple": original_triple,
                            "evidence_sentence": parts[2] if len(parts) > 2 else "",
                            "vdb_record_id": record.get("__id__")
                        })


    return final_verified_results
# async def call_llm_for_matching(query, entity_name,target_predicate, candidates, global_config):
#     """
#     封装 LLM 调用逻辑
#     """
#     # 构建候选列表的字符串描述
#     candidate_str = ""
#     for i, c in enumerate(candidates):
#         candidate_str += f"[{i}] head_entity:{c['src_id']} | tail_entity:{c['tgt_id']} | Predicate: {c['predicate']} | Evidence: {c['evidence']}\n"
#
#     # prompt = f"""
#     # [Task]
#     # In a Knowledge Graph, we are looking for a relationship that matches the user's intent.
#     # Original Question: "{query}"
#     # Desired entity_name:"{entity_name}"
#     # Desired Relationship: "{target_predicate}"
#     #
#     # [Candidates]
#     # {candidate_str}
#     #
#     # [Instruction]
#     # Analyze each candidate. Does it represent the SAME relationship as "{target_predicate}" in the context of the question?
#     # - The "{entity_name}" refers to the executor or the recipient of the "{target_predicate}"
#     # - Consider synonyms (e.g., "beat" and "defeated").
#     # - Use the 'Evidence' to understand the context.
#     # - The matching relationship can be in either direction (subject-predicate-object or object-inverse_predicate-subject
#     # - Return a JSON list of indices of the matching candidates.
#     # - If no candidate matches, return [].
#     #
#     # Example Output: [0, 2]
#     # """
#
#     use_model_func = global_config["llm_model_func"]
#     # Apply higher priority (5) to query relation LLM function
#     use_model_func = partial(use_model_func)  # must expose acomplete()
#
#
#     # 这里的 llm_client 逻辑根据你实际使用的模型（GPT, Claude, Ollama等）调整
#     response = await use_model_func(prompt, _priority=5)
#
#     # 解析 LLM 返回的索引列表 [0, 2]
#     try:
#         # 简单正则提取 JSON 数组
#         match = re.search(r'\[.*\]', response)
#         if match:
#             return json.loads(match.group())
#     except:
#         return []
#     return []
async def call_llm_for_matching(query, entity_name, target_predicate, candidates, global_config):
    """
    封装 LLM 调用逻辑
    """
    # 构建候选列表的字符串描述
    candidate_str = ""
    for i, c in enumerate(candidates):
        candidate_str += f"[{i}] head_entity:{c['src_id']} | tail_entity:{c['tgt_id']} | Predicate: {c['predicate']} | Evidence: {c['evidence']}\n"

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    template = PROMPTS.get("call_llm_for_matching")
    full_prompt = template.format(
        query=query,
        entity_name=entity_name,
        target_predicate=target_predicate,
        candidate_str=candidate_str,
        language=language
    )
    tokenizer = global_config["tokenizer"]
    logger.debug(f"[question_semantics] Tokens: {len(tokenizer.encode(full_prompt))}")


    # 3️⃣ Select model executor
    use_model_func = global_config["llm_model_func"]
    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(use_model_func)  # must expose acomplete()

    try:
        response = await use_model_func(full_prompt, _priority=5)
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return {}

    # 解析 LLM 返回的索引列表 [0, 2]
    try:
        # 简单正则提取 JSON 数组
        match = re.search(r'\[.*\]', response)
        if match:
            return json.loads(match.group())
    except:
        return []
    return []



async def _check_in_list(entity_name, target_predicate, all_relations, query, global_config):
    STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'with', 'for', 'of', 'to', 'is', 'are', 'was', 'were', 'by'}

    def get_keywords(text):
        if not text: return set()
        words = re.findall(r'\b\w+\b', text.replace('_', ' ').lower())
        return {w for w in words if w not in STOP_WORDS}

    # 获取目标谓词的关键字
    target_predicate_clean = target_predicate.strip()
    target_keywords = get_keywords(target_predicate_clean)

    # 存储结果的变量
    exact_matches = []
    fuzzy_candidates = []

    # 帮助函数：找公共子串
    def get_common_substrings(str1, str2, min_length=3):
        common_substrings = []
        for i in range(len(str1)):
            for j in range(i + min_length, len(str1) + 1):
                substr = str1[i:j]
                if substr in str2 and len(substr) >= min_length and substr not in common_substrings:
                    common_substrings.append(substr)
        return common_substrings

    # 遍历所有关系，进行匹配
    for record in all_relations:
        # 增加防御：确保 src_id 和 tgt_id 存在且为字符串
        src_id = str(record.get("src_id") or "")
        tgt_id = str(record.get("tgt_id") or "")

        if src_id == entity_name or tgt_id == entity_name:
            content = record.get("content", "")
            parts = content.split('\n')
            if len(parts) < 2: continue

            raw_predicate_str = parts[1].strip()
            # 统一使用常见的分割符
            current_predicate_list = [p.strip() for p in re.split(r'[,，/]', raw_predicate_str) if p.strip()]
            evidence = parts[2] if len(parts) > 2 else ""

            base_info = {
                "matched_node": entity_name,
                "predicate": raw_predicate_str,
                "evidence": evidence,
                "record_id": record.get("__id__"),
                "src_id": src_id,
                "tgt_id": tgt_id
            }

            # --- 策略 A: 精准匹配 ---
            if any(p.lower() == target_predicate_clean.lower() for p in current_predicate_list):
                # 显式给一个 match_score，并确保 common_count 也有值以防排序报错
                exact_matches.append({**base_info, "match_score": 1.0, "common_count": 0})
                continue

            # --- 策略 B: 使用公共子串匹配 ---
            common_substrings = get_common_substrings(target_predicate_clean, raw_predicate_str, min_length=3)
            if common_substrings:
                common_count = len(common_substrings)
                if common_count > 0:
                    fuzzy_candidates.append({**base_info, "common_count": common_count, "match_score": 0.0, "common_substrings": common_substrings})

    # 如果找到精准匹配的关系，直接返回
    if exact_matches:
        return exact_matches

    # 策略 B: 筛选公共单词匹配数量 > 3 的关系，并选出前3个
    fuzzy_candidates = sorted(fuzzy_candidates, key=lambda x: x["common_count"], reverse=True)
    top_candidates = [candidate for candidate in fuzzy_candidates if candidate["common_count"] > 3]

    # 如果找到了公共子串数量大于3的前5个关系
    if len(top_candidates) > 4:
    # if len(top_candidates) >0:
        return top_candidates[:5]

    return []


async def verify_relationships_with_fallback(
        query: str,
        matched_queue: deque,
        verified_queue: deque,
        relationships_vdb: BaseVectorStorage, # 注意：假设 BaseVectorStorage 已在外部导入
        global_config: dict[str, str]
):

    # 获取底层数据
    client = await relationships_vdb._get_client()
    storage = getattr(client, "_NanoVectorDB__storage")
    all_relations = storage.get("data", [])

    remaining_seeds = deque()
    # 记录本轮彻底失败的任务

    while matched_queue:
        seed_entity, original_triple = matched_queue.popleft()
        target_predicate = original_triple.get("predicate")

        print(f"\n--- 处理种子实体: {seed_entity} (目标关系: {target_predicate}) ---")

        # 1️⃣ 直接匹配
        direct_matches = await _check_in_list(seed_entity, target_predicate, all_relations, query, global_config)

        if direct_matches:
            is_exact = direct_matches[0].get("match_score") == 999
            print(f"  ✅ 直接匹配成功 ")
            for m in direct_matches:
                # 确保 m 包含 src_id 和 tgt_id 以供后续映射
                verified_queue.append({
                    "path_type": "direct",
                    "seed_entity": seed_entity,
                    "triple": {
                        "src_id": m.get("src_id"),
                        "tgt_id": m.get("tgt_id"),
                        "predicate": m.get("predicate"),
                        "evidence": m.get("evidence"),
                        "match_score": m.get("match_score")
                    },
                    "original_triple": original_triple
                })
            # 直接匹配成功，跳过当前循环，处理下一个 seed_entity
            continue
        found = False
        neighbors = set()
        for rel in all_relations:
            if rel.get("src_id") == seed_entity:
                neighbors.add(rel.get("tgt_id"))
            elif rel.get("tgt_id") == seed_entity:
                neighbors.add(rel.get("src_id"))

        all_neighbor_matches = []
        for neighbor in neighbors:
            res = await _check_in_list(neighbor, target_predicate, all_relations, query, global_config)
            if res:
                valid_res_for_neighbor = []
                for m in res:
                    other_end = m.get("tgt_id") if m.get("src_id") == neighbor else m.get("src_id")
                    if other_end == seed_entity:
                        continue

                    valid_res_for_neighbor.append(m)

                all_neighbor_matches.extend([(neighbor, m) for m in valid_res_for_neighbor])

        if all_neighbor_matches:
            all_neighbor_matches.sort(
                key=lambda x: (
                        (x[1].get("match_score") or 0) +
                        # 2. 其次尝试 common_count (注意：名字必须对上)
                        (x[1].get("common_count") or 0)
                ),
                reverse=True
            )

            best_score = (all_neighbor_matches[0][1].get("match_score") or 0) or \
                         (all_neighbor_matches[0][1].get("common_count") or 0)

            for neighbor, m in all_neighbor_matches:
                current_score = (m.get("match_score") or 0) or (m.get("common_count") or 0)

                if current_score < best_score:
                    break
                verified_queue.append({
                    "path_type": "1-hop",
                    "seed_entity": seed_entity,
                    "via": neighbor,
                    "triple": {
                        "src_id": m.get("src_id"),
                        "tgt_id": m.get("tgt_id"),
                        "predicate": m.get("predicate"),
                        "evidence": m.get("evidence"),
                        "match_score": m.get("match_score")
                    },
                    "original_triple": original_triple
                })
            found = True

        if not found:
            remaining_seeds.append((seed_entity, original_triple))


        # remaining_seeds.append((seed_entity, original_triple))

    matched_queue.extend(remaining_seeds)

from collections import deque


async def _expand_triples_with_map(current_triple, unknown_map):
    """
    根据映射表，将一个含有 unknown 的三元组裂变为多个具体的三元组。
    例如：(unknown1, parent, B) 且 unknown1 -> [A1, A2]
    结果：[(A1, parent, B), (A2, parent, B)]
    """
    sub = current_triple.get('subject')
    obj = current_triple.get('object')
    pred = current_triple.get('predicate')

    # 获取主语和宾语的所有可能候选
    # 如果是 unknown 且在映射表中，取所有映射；否则取原值
    subs = list(unknown_map.get(sub, [sub])) if _is_unknown(sub) else [sub]
    objs = list(unknown_map.get(obj, [obj])) if _is_unknown(obj) else [obj]

    expanded = []
    for s in subs:
        for o in objs:
            # 创建副本，防止引用冲突
            new_t = current_triple.copy()
            new_t['subject'] = s
            new_t['object'] = o
            expanded.append(new_t)
    return expanded
def extract_text_safe(content_obj) -> str:
    if content_obj is None:
        return ""

    if isinstance(content_obj, str):
        return content_obj.strip()

    if isinstance(content_obj, dict):
        for key in ("text", "content", "description", "summary"):
            val = content_obj.get(key)
            if isinstance(val, str):
                return val.strip()
        return str(content_obj)

    if isinstance(content_obj, (list, tuple, set)):
        return " ".join(str(x) for x in content_obj if x)

    return str(content_obj)


def get_unlinked_entity_id(q3_item: dict) -> str | None:
    seed = q3_item.get("seed_entity")
    triple = q3_item.get("triple", {})

    src = triple.get("src_id")
    tgt = triple.get("tgt_id")

    if not seed or not src or not tgt:
        return None

    # 核心判断
    if src == seed:
        return tgt
    if tgt == seed:
        return src

    # 理论上不应该发生，兜底
    return None

async def collect_unlinked_entity_contents(
        q3: list,
        entities_vdb: BaseVectorStorage
) -> list[str]:

    # 1. 读取实体库
    all_entity_raw = await entities_vdb.client_storage
    raw_data = all_entity_raw.get("data", []) if isinstance(all_entity_raw, dict) else []

    entity_map = {
        item.get("entity_name"): item
        for item in raw_data
        if item.get("entity_name")
    }

    # 2. 收集证据
    evidence_sentences = []

    for item in q3:
        target_entity_id = get_unlinked_entity_id(item)
        if not target_entity_id:
            continue

        entity_obj = entity_map.get(target_entity_id)
        if not entity_obj:
            continue

        content = entity_obj.get("content")
        safe_text = extract_text_safe(content)

        if safe_text:
            evidence_sentences.append(safe_text)

    return evidence_sentences

async def recursive_kg_filling_pipeline(
        initial_triples: list,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict,
        query: str,

):
    q1 = deque(initial_triples)

    # Q3: 最终验证通过的完整三元组结果池
    q3 = []
    q2 = deque()
    # 新增：黑名单集合，防止重复无效验证
    blacklisted_tasks = set()
    # 用于记录已确定的 unknown 映射，例如: {"unknown1(Monarch)": "John_of_Gaunt"}
    unknown_map = {}
    # 用于拼接所有获取的 sentence
    final_sentences = ""

    iteration = 0
    while q1 and iteration < 5:
        iteration += 1
        print(f"\n=== 第 {iteration} 轮迭代 ===")

        # --- 步骤 1: 实体链接 (Q1 -> Q2) ---
        # Q2 结构: (linked_entity_id, original_triple_from_q1)
        #q2 = deque()
        # --- 步骤 1: 实体链接 (Q1 -> Q2) ---
        remaining_q1 = deque()

        while q1:
            raw_triple = q1.popleft()

            # 1. 执行裂变：将 unknown 替换为所有可能的真实 ID
            # 如果 unknown1 有 2 个映射，这里会得到 2 个具体的三元组
            candidate_triples = await _expand_triples_with_map(raw_triple, unknown_map)

            for current_triple in candidate_triples:
                # 2. 执行实体链接

                temp_analysis = {'triples': [current_triple]}
                linked_results, entity_level_sentence = await link_entities_from_kg(temp_analysis, entities_vdb, query_param)

                if linked_results:
                    # 只要链接成功（此时 subject/object 已经是真实 ID 或库中存在的实体）
                    q2.extend(linked_results)
                    # 记录这个 linked_results 里包含了 original_triple (即这里的 current_triple)
                else:
                    # 如果依然无法链接，保留这个具体形态的三元组
                    print(f"{current_triple}实体链接失败")
                    remaining_q1.append(current_triple)

                if entity_level_sentence:
                    final_sentences += f" {entity_level_sentence}"


        # 将无法链接的放回 Q1 等待下一轮填充
        q1 = remaining_q1

        # --- 步骤 2: 关系验证 (Q2 -> Q3) ---
        verified_queue = deque()  # 临时存放验证结果
        failed_tasks = await verify_relationships_with_fallback(
            query=query,
            matched_queue=q2,
            verified_queue=verified_queue,
            relationships_vdb=relationships_vdb,
            global_config=global_config

        )
        # --- 步骤 2: 过滤已尝试过的验证 (核心改进点) ---
        # active_q2 = deque()
        # while q2:
        #     task = q2.popleft()
        #     entity_id = task["entity"]
        #     # 如果这个三元组之前已经在这个实体上验证失败过，就跳过本次验证，保留在队列中
        #     if entity_id in task["tried_cache"]:
        #         active_q2.append(task)  # 原样放回，等待下次迭代可能产生的变化
        #         continue
        #
        #     # 否则，准备进入验证阶段
        #     task["tried_cache"].add(entity_id)
        #
        #     # 临时构建一个 verify_relationships_with_fallback 能接受的格式
        #     temp_matched_queue = deque([(entity_id, task["triple"])])
        #     verified_queue = deque()
        #
        #     await verify_relationships_with_fallback(
        #         query=query,
        #         matched_queue=temp_matched_queue,
        #         verified_queue=verified_queue,
        #         relationships_vdb=relationships_vdb,
        #         global_config=global_config
        #     )[cite: 45, 46]
        # # --- 步骤 3: 填充与映射更新 (Q3 -> 更新 unknown_map) ---
        # # --- 步骤 3: 填充与映射更新 (Q3 -> 更新 unknown_map) ---
        # # --- 步骤 3: 填充与映射更新 ---

        while verified_queue:
            item = verified_queue.popleft()
            real_t = item['triple']  # 数据库真实三元组
            orig_t = item['original_triple']  # LLM 生成的带 unknown 的三元组
            seed = item['seed_entity']  # 链接到的实体

            # 【关键修改】：如果存在 via 则取 via（一跳），否则取 seed（直接匹配）
            anchor_node = item.get('via') or seed

            q3.append(item)

            unknown_key = None
            resolved_id = None

            # 确定 Unknown 对应的真实 ID
            # 原理：在查到的 real_t 中，剔除掉那个作为“已知锚点”的 ID，剩下的就是我们要的 ID
            if _is_unknown(orig_t.get('object')):
                unknown_key = orig_t['object']
                # 如果真实三元组的头节点是锚点，那尾节点就是答案；反之亦然
                resolved_id = real_t['tgt_id'] if real_t['src_id'] == anchor_node else real_t['src_id']

            elif _is_unknown(orig_t.get('subject')):
                unknown_key = orig_t['subject']
                resolved_id = real_t['src_id'] if real_t['tgt_id'] == anchor_node else real_t['tgt_id']

            if unknown_key and resolved_id:
                # 更新映射表
                if unknown_key not in unknown_map:
                    unknown_map[unknown_key] = set()

                # 排除掉解析结果等于锚点本身的情况（防止环形引用）
                if resolved_id != anchor_node:
                    unknown_map[unknown_key].add(resolved_id)
                    print(f"  [发现映射] {unknown_key} -> {resolved_id} (锚点: {anchor_node})")

        # if not unknown_map and not q2:
        #     print("无法进一步发现新关系，推理提前结束。")
        #     break
    unlinked_entity_sentences = await collect_unlinked_entity_contents(
        q3=q3,
        entities_vdb=entities_vdb
    )
    return q1,q2,q3,final_sentences



def _is_unknown(val):
    return val and "unknown" in str(val).lower()


def _fill_triple_with_map(triple, mapping):
    """使用映射表替换三元组中的占位符"""
    for slot in ['subject', 'object']:
        val = triple.get(slot)
        if val in mapping:
            triple[slot] = mapping[val]


async def find_paths_to_verified_context(
        q2: deque,
        q3: list,
        relationships_vdb: BaseVectorStorage,
        max_depth: int = 6
):
    client = await relationships_vdb._get_client()
    storage = getattr(client, "_NanoVectorDB__storage")
    all_relations = storage.get("data", [])

    # 1. 核心：构建双向邻接表
    # 这一步保证了：无论 A 是 src 还是 tgt，adj[A] 都能找到所有与其相连的边
    adj = defaultdict(list)
    for rel in all_relations:
        s, t = rel.get("src_id"), rel.get("tgt_id")
        if s: adj[s].append(rel)
        if t: adj[t].append(rel)

    # 2. 收集 Q3 目标池
    q3_entities = set()
    seed_entities = set()
    for item in q3:
        q3_entities.add(item['triple']['src_id'])
        q3_entities.add(item['triple']['tgt_id'])
        seed_entities.add(item['seed_entity'])

    path_results = []

    # 3. 针对 Q2 中的实体 A 进行路径搜索
    while q2:
        linked_name, original_triple = q2.popleft()

        # 如果库里根本没有这个 ID，直接跳过
        if linked_name not in adj:
            continue

        # BFS 寻找最短路径
        search_queue = deque([(linked_name, [])])
        visited = {linked_name}
        found_paths_for_A = []

        while search_queue:
            current_node, current_path = search_queue.popleft()

            if len(current_path) >= max_depth:
                continue

            # 获取所有关联的三元组（无论 current_node 是头还是尾）
            for rel in adj.get(current_node, []):
                # 确定邻居节点：如果是从 src 进来，邻居就是 tgt；反之亦然
                neighbor = rel['tgt_id'] if rel['src_id'] == current_node else rel['src_id']

                if neighbor in visited:
                    continue

                new_path = current_path + [rel]

                # 命中 Q3 目标
                if neighbor in q3_entities:
                    score = 2 if neighbor in seed_entities else 1
                    found_paths_for_A.append({
                        "start_node": linked_name, # Q2 的实体 A
                        "target_node": neighbor,   # 找到的 Q3 实体
                        "path": new_path,
                        "priority": score,
                        "original_triple": original_triple
                    })
                    # 找到最短路径后，停止当前分支的深挖
                    continue

                visited.add(neighbor)
                search_queue.append((neighbor, new_path))

        # 结果排序：优先选择路径最短且指向 seed_entity 的路径
        if found_paths_for_A:
            # 优先级第一，路径长度（负号表示越短越前）第二
            found_paths_for_A.sort(key=lambda x: (x['priority'], -len(x['path'])), reverse=True)
            path_results.append(found_paths_for_A[0])

    return path_results


async def extract_evidence_from_q3_simple(q3: list) -> str:
    """
    从 Q3 结果池中提取所有证据文本 (evidence)，合并为 RAG 的上下文内容
    """
    evidence_list = []

    for item in q3:
        # 情况 1：标准三元组结果 (包含单个 triple 字典)
        if isinstance(item.get('triple'), dict):
            evidence = item['triple'].get('evidence')
            if evidence:
                evidence_list.append(evidence)

        # 情况 2：路径搜索结果 (full_path 包含多个三元组字典的列表)
        elif isinstance(item.get('full_path'), list):
            for step in item['full_path']:
                evidence = step.get('evidence')
                if evidence:
                    evidence_list.append(evidence)

    # 去重处理，防止同一段证据被多次提取
    unique_evidence = list(dict.fromkeys(evidence_list))

    # 将所有证据拼接成一段完整的上下文文本
    context = "\n".join(unique_evidence)
    return context
import json
import re

import re
from typing import List


async def extract_evidence_from_q3(q3: list, entities_vdb: BaseVectorStorage) -> str:
    """
    从 Q3 结果池中提取所有证据文本 (evidence) 和实体的 description，
    合并为 RAG 的上下文内容。
    """
    evidence_list = []

    # 获取所有实体数据
    all_entity_raw = await entities_vdb.client_storage
    raw_data = all_entity_raw.get("data", []) if isinstance(all_entity_raw, dict) else []

    # 构建一个字典，将每个实体名称映射到其对应的 content
    entity_descriptions = {}
    for entity in raw_data:
        entity_name = entity.get("entity_name")
        content = entity.get("content")
        if entity_name and content:
            entity_descriptions[entity_name] = content

    for item in q3:
        # 情况 1：标准三元组结果 (包含单个 triple 字典)
        if isinstance(item.get('triple'), dict):
            # 提取证据文本
            evidence = item['triple'].get('evidence')
            if evidence:
                evidence_list.append(evidence)

            # 提取 src_id 和 tgt_id 对应的 description
            src_id = item['triple'].get('src_id')
            tgt_id = item['triple'].get('tgt_id')

            # 获取 src_id 和 tgt_id 的描述（content）
            if src_id:
                src_description = entity_descriptions.get(src_id, 'No description available')
                evidence_list.append(f"Description for {src_id}: {src_description}")

            if tgt_id:
                tgt_description = entity_descriptions.get(tgt_id, 'No description available')
                evidence_list.append(f"Description for {tgt_id}: {tgt_description}")

        # 情况 2：路径搜索结果 (full_path 包含多个三元组字典的列表)
        elif isinstance(item.get('full_path'), list):
            for step in item['full_path']:
                # 提取证据文本
                evidence = step.get('evidence')
                if evidence:
                    evidence_list.append(evidence)

                # 提取 src_id 和 tgt_id 对应的 description
                src_id = step.get('src_id')
                tgt_id = step.get('tgt_id')

                # 获取 src_id 和 tgt_id 的描述（content）
                if src_id:
                    src_description = entity_descriptions.get(src_id, 'No description available')
                    evidence_list.append(f"Description for {src_id}: {src_description}")

                if tgt_id:
                    tgt_description = entity_descriptions.get(tgt_id, 'No description available')
                    evidence_list.append(f"Description for {tgt_id}: {tgt_description}")

    # 去重处理，防止同一段证据被多次提取
    unique_evidence = list(dict.fromkeys(evidence_list))

    # 将所有证据拼接成一段完整的上下文文本
    context = "\n".join(unique_evidence)
    return context

async def find_paths_to_type_constrained_node(
        q2: deque,
        relationships_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        max_depth: int = 6
):
    """
    针对 Q2 中的三元组，寻找符合类型约束的未知实体，并返回最短的前 5 条路径。
    实体类型匹配采用模糊匹配，共有单词数越多优先级越高。
    """
    from collections import defaultdict, deque
    import re

    # 1. 获取关系邻接表
    rel_client = await relationships_vdb._get_client()
    rel_storage = getattr(rel_client, "_NanoVectorDB__storage")
    all_relations = rel_storage.get("data", [])

    adj = defaultdict(list)
    for rel in all_relations:
        s, t = rel.get("src_id"), rel.get("tgt_id")
        if s: adj[s].append(rel)
        if t: adj[t].append(rel)

    # 2. 预加载实体类型映射表
    ent_raw = await entities_vdb.client_storage
    entity_type_map = {
        item.get("entity_name"): item.get("entity_type", "")
        for item in ent_raw.get("data", [])
    }

    all_found_paths = []

    # 3. 遍历 Q2 任务
    while q2:
        linked_name, original_triple = q2.popleft()

        subj = original_triple.get('subject', '')
        obj = original_triple.get('object', '')
        is_obj_unknown = "unknown" in obj.lower()
        target_raw = obj if is_obj_unknown else subj

        match = re.search(r'\((.*?)\)', target_raw)
        target_type_constraint = match.group(1).lower() if match else None

        if not target_type_constraint or linked_name not in adj:
            continue

        target_words = set(target_type_constraint.split())

        # BFS 搜索
        search_queue = deque([(linked_name, [])])
        visited = {linked_name}
        task_candidates = []

        while search_queue:
            current_node, current_path = search_queue.popleft()
            if len(current_path) >= max_depth:
                continue

            for rel in adj.get(current_node, []):
                neighbor = rel['tgt_id'] if rel['src_id'] == current_node else rel['src_id']
                if neighbor in visited:
                    continue

                new_path = current_path + [rel]
                actual_entity_type = entity_type_map.get(neighbor, "").lower()
                entity_words = set(actual_entity_type.split())

                # 计算共有单词数
                common_word_count = len(target_words & entity_words)

                if common_word_count > 0:
                    # 模糊匹配成功，记录路径
                    task_candidates.append({
                        "start_node": linked_name,
                        "found_node": neighbor,
                        "path": new_path,
                        "path_len": len(new_path),
                        "found_type": actual_entity_type,
                        "common_word_count": common_word_count,
                        "original_triple": original_triple
                    })
                    # 找到目标后可以继续深挖同层，或者不再深挖视需求而定
                    continue

                visited.add(neighbor)
                search_queue.append((neighbor, new_path))

        # 汇总当前 Q2 实体路径
        all_found_paths.extend(task_candidates)

    # 4. 排序：先按共有单词数降序，再按路径长度升序
    all_found_paths.sort(key=lambda x: (-x['common_word_count'], x['path_len']))

    top_5_paths = all_found_paths[:5]

    print(f"🔍 总共找到路径数: {len(all_found_paths)}, 已筛选前 5 条最相关路径")
    return top_5_paths

def get_common_substrings(str1, str2):
    """
    获取两个字符串的共同子串，返回交集的部分。
    """
    common_substrings = set()

    # 遍历所有可能的子串，找出共同的子串
    for i in range(len(str1)):
        for j in range(i + 1, len(str1) + 1):
            substring = str1[i:j]
            if substring in str2:
                common_substrings.add(substring)

    return common_substrings


async def find_paths_to_intent_constrained_node(
        q2: deque,
        intent_types: list[str],
        relationships_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        max_depth: int = 6
):
    """
    针对 Q2 中的起始实体，从 KG 中进行 BFS，
    在遍历过程中，如果遇到的实体 entity_type 与 intent_types 中任一类型模糊匹配，
    则保存当前路径。

    intent_types: ["person", "organization", "location", ...]
    """
    from collections import defaultdict, deque

    # ----------------------------
    # 1. 构建关系邻接表
    # ----------------------------
    rel_client = await relationships_vdb._get_client()
    rel_storage = getattr(rel_client, "_NanoVectorDB__storage")
    all_relations = rel_storage.get("data", [])

    adj = defaultdict(list)
    for rel in all_relations:
        s, t = rel.get("src_id"), rel.get("tgt_id")
        if s:
            adj[s].append(rel)
        if t:
            adj[t].append(rel)

    # ----------------------------
    # 2. 实体 -> entity_type 映射
    # ----------------------------
    ent_raw = await entities_vdb.client_storage
    entity_type_map = {
        item.get("entity_name"): item.get("entity_type", "").lower()
        for item in ent_raw.get("data", [])
    }

    # ----------------------------
    # 3. intent 类型预处理
    # ----------------------------
    intent_word_sets = [
        set(intent.lower().split())
        for intent in intent_types
    ]

    all_found_paths = []

    # ----------------------------
    # 4. 遍历 Q2
    # ----------------------------
    popped_q2_items = []
    while q2:
        item = q2.popleft()
        popped_q2_items.append(item)

        linked_name, original_triple = item

        if linked_name not in adj:
            continue

        # BFS
        search_queue = deque([(linked_name, [])])
        visited = {linked_name}

        while search_queue:
            current_node, current_path = search_queue.popleft()

            if len(current_path) >= max_depth:
                continue

            for rel in adj.get(current_node, []):
                neighbor = (
                    rel["tgt_id"]
                    if rel["src_id"] == current_node
                    else rel["src_id"]
                )

                if neighbor in visited:
                    continue

                new_path = current_path + [rel]
                neighbor_type = entity_type_map.get(neighbor, "")
                neighbor_words = set(neighbor_type.split())

                # ----------------------------
                # intent 类型匹配（模糊）
                # ----------------------------
                best_common = 0
                matched_intent = None

                for intent_words in intent_word_sets:
                    intent_string = " ".join(intent_words)  # 将意图词集拼接为字符串

                    # 计算公共子串并根据长度评估匹配度
                    common_substrings = get_common_substrings(intent_string, neighbor_type)
                    common_cnt = sum(len(substring) for substring in common_substrings)

                    if common_cnt > best_common:
                        best_common = common_cnt
                        matched_intent = intent_string
                        best_matching_node = neighbor
                #     common_cnt = len(dintent_wors & neighbor_words)
                #     if common_cnt > best_common:
                #         best_common = common_cnt
                #         matched_intent = " ".join(intent_words)
                #
                if best_common > 0:
                    all_found_paths.append({
                        "start_node": linked_name,
                        "found_node": neighbor,
                        "path": new_path,
                        "path_len": len(new_path),
                        "found_type": neighbor_type,
                        "matched_intent": matched_intent,
                        "common_word_count": best_common,
                        "original_triple": original_triple
                    })
                    # ⚠️ 不 return，继续找其他路径
                    # 是否继续向下扩展该节点，可按需要调整
                    # continue

                visited.add(neighbor)
                search_queue.append((neighbor, new_path))

    # ----------------------------
    # 5. 可选排序（推荐）
    # ----------------------------
    all_found_paths.sort(
        key=lambda x: (-x["common_word_count"], x["path_len"])
    )

    print(f"🔍 共找到 intent 匹配路径数: {len(all_found_paths)}")
    # 恢复 q2
    for item in popped_q2_items:
        q2.append(item)

    return all_found_paths

async def fallback_one_hop_paths_from_q2(
        q2: deque,
        relationships_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage
):
    """
    当 intent 约束下未找到任何路径时，
    返回 q2 中所有【非 unknown】节点的【一跳邻居路径】。
    """
    from collections import defaultdict

    # ----------------------------
    # 1. 构建关系邻接表
    # ----------------------------
    rel_client = await relationships_vdb._get_client()
    rel_storage = getattr(rel_client, "_NanoVectorDB__storage")
    all_relations = rel_storage.get("data", [])

    adj = defaultdict(list)
    for rel in all_relations:
        s, t = rel.get("src_id"), rel.get("tgt_id")
        if s:
            adj[s].append(rel)
        if t:
            adj[t].append(rel)

    # ----------------------------
    # 2. 实体类型映射（可选但推荐）
    # ----------------------------
    ent_raw = await entities_vdb.client_storage
    entity_type_map = {
        item.get("entity_name"): item.get("entity_type", "").lower()
        for item in ent_raw.get("data", [])
    }

    fallback_paths = []

    # ----------------------------
    # 3. 遍历 Q2
    # ----------------------------
    for linked_name, original_triple in q2:

        # 跳过 unknown 节点
        if "unknown" in linked_name.lower():
            continue

        if linked_name not in adj:
            continue

        for rel in adj[linked_name]:
            neighbor = (
                rel["tgt_id"]
                if rel["src_id"] == linked_name
                else rel["src_id"]
            )

            fallback_paths.append({
                "start_node": linked_name,
                "found_node": neighbor,
                "path": [rel],          # 一跳路径
                "path_len": 1,
                "found_type": entity_type_map.get(neighbor, ""),
                "matched_intent": None, # 明确标记为 fallback
                "common_word_count": 0,
                "original_triple": original_triple,
                "fallback": True
            })

    print(f"🩹 fallback 启动：返回一跳路径数 = {len(fallback_paths)}")
    return fallback_paths


async def fallback_two_hop_paths_from_q2(
        q2: deque,
        relationships_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage
):
    """
    修改版：返回 q2 中所有【非 unknown】节点的【两跳邻居路径】。
    """
    from collections import defaultdict

    # ----------------------------
    # 1. 构建关系邻接表
    # ----------------------------
    rel_client = await relationships_vdb._get_client()
    rel_storage = getattr(rel_client, "_NanoVectorDB__storage")
    all_relations = rel_storage.get("data", [])

    adj = defaultdict(list)
    for rel in all_relations:
        s, t = rel.get("src_id"), rel.get("tgt_id")
        if s: adj[s].append(rel)
        if t: adj[t].append(rel)

    # ----------------------------
    # 2. 实体类型映射
    # ----------------------------
    ent_raw = await entities_vdb.client_storage
    entity_type_map = {
        item.get("entity_name"): item.get("entity_type", "").lower()
        for item in ent_raw.get("data", [])
    }

    fallback_paths = []

    # ----------------------------
    # 3. 遍历 Q2 进行两跳扩展
    # ----------------------------
    for linked_name, original_triple in q2:
        # 跳过 unknown 节点
        if "unknown" in linked_name.lower():
            continue

        if linked_name not in adj:
            continue

        # --- 第一跳 ---
        for rel1 in adj[linked_name]:
            neighbor1 = rel1["tgt_id"] if rel1["src_id"] == linked_name else rel1["src_id"]

            # 记录第一跳路径
            path_info1 = {
                "start_node": linked_name,
                "found_node": neighbor1,
                "path": [rel1],
                "path_len": 1,
                "found_type": entity_type_map.get(neighbor1, ""),
                "matched_intent": None,
                "common_word_count": 0,
                "original_triple": original_triple,
                "fallback": True
            }
            fallback_paths.append(path_info1)

            # --- 第二跳 ---
            if neighbor1 in adj:
                for rel2 in adj[neighbor1]:
                    neighbor2 = rel2["tgt_id"] if rel2["src_id"] == neighbor1 else rel2["src_id"]

                    # 避免回到起点 (linked_name)
                    if neighbor2 == linked_name:
                        continue

                    # 记录第二跳路径
                    path_info2 = {
                        "start_node": linked_name,
                        "found_node": neighbor2,
                        "path": [rel1, rel2],  # 包含两跳关系
                        "path_len": 2,
                        "found_type": entity_type_map.get(neighbor2, ""),
                        "matched_intent": None,
                        "common_word_count": 0,
                        "original_triple": original_triple,
                        "fallback": True
                    }
                    fallback_paths.append(path_info2)

    print(f"🩹 Two-hop fallback 启动：总路径数 = {len(fallback_paths)}")
    return fallback_paths

def _format_operation_step_results(step_results: list[dict]) -> str:
    if not step_results:
        return "No previous step results are available."

    formatted_results = []
    for item in step_results:
        if not isinstance(item, dict):
            formatted_results.append(str(item))
            continue

        result_payload = item.get("context", item.get("result", ""))
        if not isinstance(result_payload, str):
            result_payload = json.dumps(result_payload, ensure_ascii=False, indent=2)

        formatted_results.append(
            "\n".join(
                [
                    f"Step ID: {item.get('step_id', '')}",
                    f"Strategy: {item.get('strategy', '')}",
                    f"Result:\n{result_payload}",
                ]
            )
        )

    return "\n\n---\n\n".join(formatted_results)


def _dedupe_context_blocks(contexts: list[str]) -> str:
    seen_normalized = set()
    unique_blocks = []

    for context in contexts or []:
        if not context:
            continue
        text = context.strip()
        if not text:
            continue

        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if normalized in seen_normalized:
            continue

        seen_normalized.add(normalized)
        unique_blocks.append(text)

    return "\n\n".join(unique_blocks)


def _make_retrieval_memory(query: str) -> dict:
    created_at = time.time()
    run_id = compute_mdhash_id(f"{query}:{created_at}", prefix="blackboard-run-")
    memory = {
        "run_id": run_id,
        "original_query": query,
        "decomposition": {},
        "retrieval_decisions": [],
        "steps": [],
        "verifications": [],
        "backtracking_events": [],
        "control_events": [],
        "messages": [],
        "created_at": created_at,
    }
    memory["blackboard"] = {
        "run_id": run_id,
        "Qorig": query,
        "P": [],
        "D": [],
        "Rstep": [],
        "Vpast": [],
        "Mexp": {
            "enabled": False,
            "matched_count": 0,
            "matched_experiences": [],
            "store_events": [],
        },
        "static_boundary": {
            "entity_types": [],
            "relationship_types": [],
            "question_intent": "",
            "reasoning_type": "multi_hop",
        },
        "control_state": "monitoring",
        "iteration": 0,
        "created_at": created_at,
        "updated_at": created_at,
    }
    return memory


def _blackboard(memory: dict | None) -> dict:
    if memory is None:
        return {}
    return memory.setdefault("blackboard", {
        "Qorig": memory.get("original_query", ""),
        "P": [],
        "D": [],
        "Rstep": [],
        "Vpast": [],
        "Mexp": {"enabled": False, "matched_count": 0, "matched_experiences": [], "store_events": []},
        "static_boundary": {},
        "control_state": "monitoring",
        "iteration": 0,
    })


def _update_blackboard(memory: dict | None, **fields) -> None:
    board = _blackboard(memory)
    if not board:
        return
    for key, value in fields.items():
        if key == "static_boundary":
            board.setdefault("static_boundary", {}).update(value or {})
        elif key == "Mexp":
            board.setdefault("Mexp", {}).update(value or {})
        else:
            board[key] = value
    board["updated_at"] = time.time()


def _append_blackboard_evidence(
        memory: dict | None,
        step_result: dict,
        iteration: int = 0,
) -> None:
    board = _blackboard(memory)
    if not board or not isinstance(step_result, dict):
        return

    context = step_result.get("context", "")
    context_text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
    evidence_block = {
        "step_id": _safe_step_id(step_result.get("step_id")),
        "strategy": step_result.get("strategy", ""),
        "status": "active",
        "iteration": iteration,
        "has_context": bool(context_text.strip()),
        "context": context_text,
        "repair_context": step_result.get("repair_context"),
        "debug": step_result.get("debug", {}),
        "created_at": time.time(),
    }
    board.setdefault("Rstep", []).append(evidence_block)


def _append_blackboard_verification(memory: dict | None, verification: dict) -> None:
    board = _blackboard(memory)
    if not board or not isinstance(verification, dict):
        return

    board.setdefault("Vpast", []).append({
        "iteration": board.get("iteration", 0),
        "verification": verification,
        "created_at": time.time(),
    })


def _truncate_blackboard_for_backtrack(
        memory: dict | None,
        start_step_id: int,
        trusted_step_results: list[dict],
) -> list[dict]:
    board = _blackboard(memory)
    if not board:
        return []

    discarded = [
        {
            "step_id": item.get("step_id"),
            "strategy": item.get("strategy"),
            "iteration": item.get("iteration"),
            "had_context": item.get("has_context"),
        }
        for item in board.get("Rstep", [])
        if _safe_step_id(item.get("step_id")) >= start_step_id
    ]
    trusted_ids = {
        _safe_step_id(item.get("step_id"))
        for item in trusted_step_results or []
        if isinstance(item, dict)
    }
    retained = []
    for item in board.get("Rstep", []):
        step_id = _safe_step_id(item.get("step_id"))
        if step_id >= start_step_id:
            continue
        retained_item = dict(item)
        if step_id in trusted_ids:
            retained_item["status"] = "locked_trusted"
        retained.append(retained_item)

    board["Rstep"] = retained
    persist_counts = memory.get("_blackboard_persist_counts")
    if isinstance(persist_counts, dict):
        persist_counts["Rstep"] = min(
            persist_counts.get("Rstep", 0),
            len(retained),
        )
    board["control_state"] = "reverse_control"
    control_event = {
        "event_type": "selective_local_truncation",
        "start_step_id": start_step_id,
        "trusted_step_ids": sorted(trusted_ids),
        "discarded_suffix": discarded,
        "created_at": time.time(),
    }
    board.setdefault("control_events", []).append(control_event)
    memory.setdefault("control_events", []).append(control_event)
    return discarded


def _json_dumps_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_blackboard_persistence_enabled(global_config: dict[str, Any] | None) -> bool:
    if not global_config:
        return False
    addon_params = global_config.get("addon_params", {}) or {}
    value = (
        global_config.get("enable_blackboard_persistence")
        if "enable_blackboard_persistence" in global_config
        else addon_params.get("enable_blackboard_persistence")
    )
    if value is None:
        value = os.getenv("ENABLE_BLACKBOARD_PERSISTENCE", "true")
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


# def _get_blackboard_db_path(global_config: dict[str, Any] | None) -> str:
#     global_config = global_config or {}
#     addon_params = global_config.get("addon_params", {}) or {}
#     db_path = (
#         global_config.get("blackboard_db_path")
#         or addon_params.get("blackboard_db_path")
#         or os.getenv("BLACKBOARD_DB_PATH")
#     )
#     working_dir = (
#         global_config.get("working_dir")
#         or addon_params.get("working_dir")
#         or os.getcwd()
#     )
#     if not db_path:
#         db_path = os.path.join(
#             global_config.get("experience_memory_dir")
#             or addon_params.get("experience_memory_dir")
#             or working_dir,
#             "BLEPRAG_blackboard.sqlite3",
#         )
#     elif not os.path.isabs(db_path):
#         db_path = os.path.join(working_dir, db_path)
#     return os.path.abspath(db_path)
#

# def _ensure_blackboard_db(db_path: str) -> None:
#     db_dir = os.path.dirname(db_path)
#     if db_dir:
#         os.makedirs(db_dir, exist_ok=True)
#
#     with sqlite3.connect(db_path) as conn:
#         conn.execute("PRAGMA journal_mode=WAL")
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS blackboard_runs (
#                 run_id TEXT PRIMARY KEY,
#                 query TEXT NOT NULL,
#                 status TEXT NOT NULL,
#                 control_state TEXT,
#                 iteration INTEGER DEFAULT 0,
#                 created_at REAL,
#                 updated_at REAL,
#                 blackboard_json TEXT NOT NULL,
#                 retrieval_memory_json TEXT
#             )
#         """)
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS blackboard_evidence (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 run_id TEXT NOT NULL,
#                 step_id INTEGER,
#                 iteration INTEGER,
#                 strategy TEXT,
#                 status TEXT,
#                 has_context INTEGER,
#                 created_at REAL,
#                 payload_json TEXT NOT NULL,
#                 FOREIGN KEY(run_id) REFERENCES blackboard_runs(run_id)
#             )
#         """)
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS blackboard_verifications (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 run_id TEXT NOT NULL,
#                 iteration INTEGER,
#                 overall_valid INTEGER,
#                 first_invalid_step INTEGER,
#                 last_valid_step_id INTEGER,
#                 created_at REAL,
#                 payload_json TEXT NOT NULL,
#                 FOREIGN KEY(run_id) REFERENCES blackboard_runs(run_id)
#             )
#         """)
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS blackboard_events (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 run_id TEXT NOT NULL,
#                 event_type TEXT,
#                 step_id INTEGER,
#                 iteration INTEGER,
#                 created_at REAL,
#                 payload_json TEXT NOT NULL,
#                 FOREIGN KEY(run_id) REFERENCES blackboard_runs(run_id)
#             )
#         """)
#         conn.execute("""
#             CREATE INDEX IF NOT EXISTS idx_blackboard_evidence_run_step
#             ON blackboard_evidence(run_id, step_id)
#         """)
#         conn.execute("""
#             CREATE INDEX IF NOT EXISTS idx_blackboard_verifications_run
#             ON blackboard_verifications(run_id)
#         """)
#         conn.execute("""
#             CREATE INDEX IF NOT EXISTS idx_blackboard_events_run_type
#             ON blackboard_events(run_id, event_type)
#         """)


# def _persist_blackboard_snapshot(
#         memory: dict | None,
#         global_config: dict[str, Any] | None,
#         status: str = "running",
# ) -> None:
#     if memory is None or not _is_blackboard_persistence_enabled(global_config):
#         return
#
#     board = _blackboard(memory)
#     if not board:
#         return
#
#     try:
#         db_path = _get_blackboard_db_path(global_config)
#         _ensure_blackboard_db(db_path)
#         run_id = board.get("run_id") or memory.get("run_id")
#         if not run_id:
#             run_id = compute_mdhash_id(
#                 f"{memory.get('original_query', '')}:{time.time()}",
#                 prefix="blackboard-run-",
#             )
#             board["run_id"] = run_id
#             memory["run_id"] = run_id
#
#         now = time.time()
#         board["updated_at"] = now
#         board.setdefault("persistence", {}).update({
#             "enabled": True,
#             "db_path": db_path,
#             "last_status": status,
#             "last_persisted_at": now,
#         })
#         counts = memory.setdefault("_blackboard_persist_counts", {
#             "Rstep": 0,
#             "Vpast": 0,
#             "control_events": 0,
#         })
#
#         with sqlite3.connect(db_path) as conn:
#             conn.execute("""
#                 INSERT INTO blackboard_runs (
#                     run_id, query, status, control_state, iteration,
#                     created_at, updated_at, blackboard_json, retrieval_memory_json
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 ON CONFLICT(run_id) DO UPDATE SET
#                     status=excluded.status,
#                     control_state=excluded.control_state,
#                     iteration=excluded.iteration,
#                     updated_at=excluded.updated_at,
#                     blackboard_json=excluded.blackboard_json,
#                     retrieval_memory_json=excluded.retrieval_memory_json
#             """, (
#                 run_id,
#                 memory.get("original_query", ""),
#                 status,
#                 board.get("control_state", ""),
#                 _safe_step_id(board.get("iteration"), 0),
#                 memory.get("created_at", board.get("created_at", now)),
#                 now,
#                 _json_dumps_safe(board),
#                 _json_dumps_safe({
#                     key: value
#                     for key, value in memory.items()
#                     if key != "_blackboard_persist_counts"
#                 }),
#             ))
#
#             evidence_items = board.get("Rstep", [])
#             for item in evidence_items[counts.get("Rstep", 0):]:
#                 conn.execute("""
#                     INSERT INTO blackboard_evidence (
#                         run_id, step_id, iteration, strategy, status,
#                         has_context, created_at, payload_json
#                     )
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     run_id,
#                     _safe_step_id(item.get("step_id")),
#                     _safe_step_id(item.get("iteration"), 0),
#                     item.get("strategy", ""),
#                     item.get("status", ""),
#                     1 if item.get("has_context") else 0,
#                     item.get("created_at", now),
#                     _json_dumps_safe(item),
#                 ))
#             counts["Rstep"] = len(evidence_items)
#
#             verification_items = board.get("Vpast", [])
#             for item in verification_items[counts.get("Vpast", 0):]:
#                 verification = item.get("verification", {}) if isinstance(item, dict) else {}
#                 summary = verification.get("verification_summary", {}) if isinstance(verification, dict) else {}
#                 conn.execute("""
#                     INSERT INTO blackboard_verifications (
#                         run_id, iteration, overall_valid, first_invalid_step,
#                         last_valid_step_id, created_at, payload_json
#                     )
#                     VALUES (?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     run_id,
#                     _safe_step_id(item.get("iteration"), 0),
#                     1 if summary.get("overall_valid") else 0,
#                     _safe_step_id(summary.get("first_invalid_step"), None),
#                     _safe_step_id(summary.get("last_valid_step_id"), None),
#                     item.get("created_at", now),
#                     _json_dumps_safe(item),
#                 ))
#             counts["Vpast"] = len(verification_items)
#
#             control_events = board.get("control_events", [])
#             for item in control_events[counts.get("control_events", 0):]:
#                 conn.execute("""
#                     INSERT INTO blackboard_events (
#                         run_id, event_type, step_id, iteration, created_at, payload_json
#                     )
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 """, (
#                     run_id,
#                     item.get("event_type", ""),
#                     _safe_step_id(item.get("step_id") or item.get("start_step_id"), None),
#                     _safe_step_id(item.get("iteration"), board.get("iteration", 0)),
#                     item.get("created_at", now),
#                     _json_dumps_safe(item),
#                 ))
#             counts["control_events"] = len(control_events)
#     except Exception as e:
#         logger.warning(f"Blackboard persistence failed: {e}")


def _remember(memory: dict | None, event_type: str, payload: dict) -> None:
    if memory is None:
        return
    memory.setdefault("messages", []).append({
        "event_type": event_type,
        "payload": payload,
        "created_at": time.time(),
    })


def _normalize_experience_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^a-z0-9_\-]+", "", text)


def _canonicalize_triple_pattern(triple: dict | None) -> dict:
    triple = triple or {}

    def normalize_endpoint(value: Any) -> str:
        if _is_unknown_type(value):
            return "unknown"
        if value:
            return "entity"
        return "empty"

    return {
        "head": normalize_endpoint(triple.get("head")),
        "rel": _normalize_experience_token(triple.get("rel") or triple.get("predicate")),
        "tail": normalize_endpoint(triple.get("tail")),
    }


def _build_experience_question_type(
        query: str,
        decomposition_result: dict | None,
) -> dict:
    decomposition_result = decomposition_result or {}
    question_analysis = decomposition_result.get("question_analysis", {}) or {}
    execution_plan = decomposition_result.get("execution_plan", []) or []
    reasoning_type = decomposition_result.get("reasoning_type", "multi_hop")
    question_intent = (
        question_analysis.get("answer_type")
        or question_analysis.get("question_intent")
        or question_analysis.get("target_type")
        or ""
    )
    step_patterns = []
    for index, step in enumerate(execution_plan):
        if not isinstance(step, dict):
            continue
        step_patterns.append({
            "order": index + 1,
            "step_type": _normalize_experience_token(step.get("step_type", "")),
            "triple": _canonicalize_triple_pattern(step.get("triple", {})),
        })

    label_payload = {
        "reasoning_type": _normalize_experience_token(reasoning_type),
        "question_intent": _normalize_experience_token(question_intent),
        "step_count": len(step_patterns),
        "step_patterns": step_patterns,
    }
    label = json.dumps(label_payload, ensure_ascii=False, sort_keys=True)
    return {
        "key": compute_mdhash_id(label, prefix="exp-type-"),
        "label": label_payload,
        "sample_query": query,
    }


async def _load_experience_memories(
        experience_memory_kv: BaseKVStorage | None,
        question_type: dict,
        top_k: int = 1,
) -> list[dict]:
    if experience_memory_kv is None or not question_type:
        return []
    limit = max(0, top_k)
    if limit == 0:
        return []

    def sort_experiences(items: list[dict] | None) -> list[dict]:
        if not isinstance(items, list):
            return []
        return sorted(
            [item for item in items if isinstance(item, dict)],
            key=lambda item: (
                item.get("success_count", 0),
                item.get("updated_at", item.get("created_at", 0)),
            ),
            reverse=True,
        )

    type_key = question_type.get("key", "")
    target_label = question_type.get("label", {}) or {}
    exact_experiences: list[dict] = []

    try:
        record = await experience_memory_kv.get_by_id(type_key)
        exact_experiences = sort_experiences((record or {}).get("experiences", []))
    except Exception as e:
        logger.warning(f"Experience memory exact lookup failed: {e}")

    if len(exact_experiences) >= limit:
        return exact_experiences[:limit]

    recalled_experiences = list(exact_experiences[:limit])
    seen_ids = {
        item.get("id")
        for item in recalled_experiences
        if isinstance(item, dict) and item.get("id")
    }
    remaining = limit - len(recalled_experiences)
    if remaining <= 0:
        return recalled_experiences

    get_all = getattr(experience_memory_kv, "get_all", None)
    if not callable(get_all):
        return recalled_experiences

    try:
        all_records = await get_all()
    except Exception as e:
        logger.warning(f"Experience memory global scan failed: {e}")
        return recalled_experiences

    if not isinstance(all_records, dict):
        return recalled_experiences

    similarity_threshold = 0.5
    similar_candidates: list[tuple[float, float, int, float, dict]] = []
    for candidate_key, candidate_record in all_records.items():
        if candidate_key == type_key or not isinstance(candidate_record, dict):
            continue

        record_label = candidate_record.get("question_type", {}) or {}
        record_similarity = _topology_similarity(target_label, record_label)
        if record_similarity <= 0:
            continue

        for item in sort_experiences(candidate_record.get("experiences", [])):
            experience_id = item.get("id")
            if experience_id and experience_id in seen_ids:
                continue

            structural_topology = item.get("structural_topology", {}) or {}
            item_problem = item.get("problem", {}) or {}
            item_similarity = max(
                record_similarity,
                _topology_similarity(target_label, structural_topology.get("question_type", {}) or {}),
                _topology_similarity(target_label, item_problem.get("question_type_label", {}) or {}),
            )
            # if item_similarity <= 0:
            #     continue
            # 优化点 2：【核心防御门禁】
            # 只有相似度达到或超过了设定的阈值（例如 0.65），系统才承认它是“有价值的参考”
            # 否则宁可不要，直接舍弃，防止低相似度的垃圾经验污染 Prompt
            if item_similarity < similarity_threshold:
                continue
            similar_candidates.append((
                item_similarity,
                item.get("confidence_weight", 0.0),
                item.get("activation_count", item.get("success_count", 0)),
                item.get("updated_at", item.get("created_at", 0)),
                item,
            ))

    similar_candidates.sort(
        key=lambda item: (item[0], item[1], item[2], item[3]),
        reverse=True,
    )
    for _, _, _, _, item in similar_candidates:
        recalled_experiences.append(item)
        experience_id = item.get("id")
        if experience_id:
            seen_ids.add(experience_id)
        if len(recalled_experiences) >= limit:
            break

    return recalled_experiences[:limit]

#遍历经验列表，但用 experiences[:3] 严格限制最多只取前 3 条
# （即使传入了更多，也只取最优质的前 3 条，避免 Prompt 过长浪费 Token）。
def _format_experience_context_for_query(experiences: list[dict] | None) -> str:
    if not experiences:
        return ""

    lines = ["Experience memory for this question type:"]
    for index, item in enumerate(experiences[:3], start=1):
        # 1. 核心大模型反思成果（高层编译策略）
        compiled = item.get("compiled_experience", {}) or {}

        # 2. 核心机械拓扑结构（包含原始的或是泛化放宽后的底层元数据）
        structural = item.get("structural_topology", {}) or {}
        tool_schedule = (
                compiled.get("tool_policy", {}).get("preferred_schedule", [])
                or structural.get("tool_schedule", [])
        )
        relaxations = item.get("topology_relaxations", [])
        lines.extend([
            f"{index}. Pattern: {compiled.get('name') or compiled.get('retrieval_pattern') or 'general retrieval pattern'}",
            f"   Applicability: {compiled.get('applicability', '')}",
            f"   Failure diagnosis: {compiled.get('failure_diagnosis', '')}",
            f"   Useful strategy: {compiled.get('repair_policy') or 'hybrid'}",
            f"   Evidence requirements: {json.dumps(compiled.get('evidence_requirements', []), ensure_ascii=False)}",
            f"   Anti-patterns: {json.dumps(compiled.get('anti_patterns', []), ensure_ascii=False)}",
            f"   Generalized tool schedule: {json.dumps(tool_schedule, ensure_ascii=False)}",
            f"   Generalization hint: {compiled.get('prompt_hint') or structural.get('generalization_hint', '')}",
            f"   Predicate relaxation hints: {json.dumps(relaxations[-3:] if isinstance(relaxations, list) else [], ensure_ascii=False)}",
        ])
    lines.append("Use these experiences as hints, but still verify evidence from the current corpus.")
    return "\n".join(lines)


def _find_step_by_id(items: list[dict], step_id: int) -> dict:
    for item in items or []:
        if isinstance(item, dict) and _safe_step_id(item.get("step_id")) == step_id:
            return item
    return {}


def _normalize_strategy_name(value: Any) -> str:
    return _normalize_experience_token(value or "hybrid") or "hybrid"


def _infer_evidence_modality(strategy: str) -> str:
    normalized = _normalize_strategy_name(strategy)
    if "kg" in normalized or "graph" in normalized:
        return "knowledge_graph"
    if "vector" in normalized or "text" in normalized:
        return "vector_text"
    if "operation" in normalized or "op" == normalized:
        return "symbolic_operation"
    if "hybrid" in normalized:
        return "hybrid"
    return normalized


def _flatten_experience_tokens(value: Any, prefix: str = "") -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"sample_query", "context", "successful_context_preview", "created_at", "updated_at"}:
                continue
            tokens.update(_flatten_experience_tokens(child, f"{prefix}{key}:"))
    elif isinstance(value, list):
        for child in value:
            tokens.update(_flatten_experience_tokens(child, prefix))
    elif value is not None:
        token = _normalize_experience_token(value)
        if token:
            tokens.add(f"{prefix}{token}")
    return tokens


def _topology_similarity(left: dict | None, right: dict | None) -> float:
    left_tokens = _flatten_experience_tokens(left or {})
    right_tokens = _flatten_experience_tokens(right or {})
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_experience_abstraction(
        question_type: dict,
        execution_plan: list[dict],
        retrieval_decisions: list[dict],
        step_results: list[dict],
        retrieval_memory: dict,
        final_verification: dict,
        backtrack_attempts: int,
) -> dict:
    decision_by_step = {
        _safe_step_id(item.get("step_id")): item
        for item in retrieval_decisions or []
        if isinstance(item, dict)
    }
    result_by_step = {
        _safe_step_id(item.get("step_id")): item
        for item in step_results or []
        if isinstance(item, dict)
    }
    generalized_steps = []
    for index, step in enumerate(execution_plan or []):
        if not isinstance(step, dict):
            continue
        step_id = _safe_step_id(step.get("step_id"), index + 1)
        decision = decision_by_step.get(step_id, {})
        result = result_by_step.get(step_id, {})
        strategy = (
            result.get("strategy")
            or decision.get("retrieval_method")
            or decision.get("recommended_retrieval")
            or "hybrid"
        )
        generalized_steps.append({
            "order": index + 1,
            "step_type": _normalize_experience_token(step.get("step_type", "retrieval")),
            "triple": _canonicalize_triple_pattern(step.get("triple", {})),
            "depends_on": sorted([
                _safe_step_id(item)
                for item in (step.get("depends_on") or [])
                if item is not None
            ]),
            "strategy": _normalize_strategy_name(strategy),
            "modality": _infer_evidence_modality(strategy),
            "has_result": bool(str(result.get("context", "")).strip()),
        })

    repair_events = retrieval_memory.get("backtracking_events", []) or []
    repair_signatures = []
    for event in repair_events:
        if not isinstance(event, dict):
            continue
        repair_context = event.get("repair_context", {}) or {}
        repair_signatures.append({
            "failed_step_id": _safe_step_id(event.get("start_step_id") or repair_context.get("failed_step_id")),
            "recommended_strategy": _normalize_strategy_name(
                repair_context.get("recommended_strategy", "hybrid")
            ),
            "missing_information_count": len(repair_context.get("missing_information", []) or []),
            "repair_action_types": [
                _normalize_experience_token(action.get("stage", "repair"))
                for action in repair_context.get("repair_actions", []) or []
                if isinstance(action, dict)
            ],
        })

    return {
        "question_type": question_type.get("label", {}),
        "graph_topology": {
            "step_count": len(generalized_steps),
            "steps": generalized_steps,
        },
        "tool_schedule": [
            {
                "order": item.get("order"),
                "strategy": item.get("strategy"),
                "modality": item.get("modality"),
            }
            for item in generalized_steps
        ],
        "self_healing": {
            "backtrack_attempts": backtrack_attempts,
            "repair_signatures": repair_signatures,
        },
        "verification_outcome": {
            "overall_valid": bool(final_verification.get("verification_summary", {}).get("overall_valid")),
            "confidence": final_verification.get("verification_summary", {}).get("confidence", ""),
        },
        "generalization_hint": (
            "When applying this pattern, map concrete entity names to their roles, "
            "treat compatible predicates as a semantic axis, and re-verify evidence "
            "against the current corpus before generation."
        ),
    }


def _build_compact_experience_trace(
        question_type: dict,
        execution_plan: list[dict],
        retrieval_decisions: list[dict],
        step_results: list[dict],
        retrieval_memory: dict,
        final_verification: dict,
        backtrack_attempts: int,
) -> dict:
    decision_by_step = {
        _safe_step_id(item.get("step_id")): item
        for item in retrieval_decisions or []
        if isinstance(item, dict)
    }
    result_by_step = {
        _safe_step_id(item.get("step_id")): item
        for item in step_results or []
        if isinstance(item, dict)
    }
    compact_steps = []
    for index, step in enumerate(execution_plan or []):
        if not isinstance(step, dict):
            continue
        step_id = _safe_step_id(step.get("step_id"), index + 1)
        result = result_by_step.get(step_id, {})
        context = result.get("context", "")
        context_text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
        decision = decision_by_step.get(step_id, {})
        strategy = (
            result.get("strategy")
            or decision.get("retrieval_method")
            or decision.get("recommended_retrieval")
            or "graph"
        )
        compact_steps.append({
            "step_id": step_id,
            "step_type": step.get("step_type", "retrieval"),
            "sub_query_intent": step.get("sub_query", "")[:240],
            "triple_pattern": _canonicalize_triple_pattern(step.get("triple", {})),
            "depends_on": step.get("depends_on") or [],
            "strategy": _normalize_strategy_name(strategy),
            "modality": _infer_evidence_modality(strategy),
            "has_evidence": bool(context_text.strip()),
            "evidence_chars": len(context_text),
        })

    verification_summaries = []
    for item in retrieval_memory.get("verifications", []) or []:
        if not isinstance(item, dict):
            continue
        summary = item.get("verification_summary", {}) or {}
        answer_check = item.get("answer_verification", {}) or {}
        verification_summaries.append({
            "overall_valid": bool(summary.get("overall_valid")),
            "first_invalid_step": summary.get("first_invalid_step"),
            "last_valid_step_id": summary.get("last_valid_step_id"),
            "reason": (summary.get("reason") or answer_check.get("reason") or "")[:500],
            "missing_information": answer_check.get("missing_information", [])[:5],
        })

    repair_events = []
    for event in retrieval_memory.get("backtracking_events", []) or []:
        if not isinstance(event, dict):
            continue
        repair_context = event.get("repair_context", {}) or {}
        repair_events.append({
            "attempt": event.get("attempt"),
            "start_step_id": event.get("start_step_id"),
            "trusted_step_ids": event.get("trusted_step_ids", []),
            "discarded_step_ids": event.get("discarded_step_ids", []),
            "reason": (repair_context.get("reason") or "")[:500],
            "recommended_strategy": repair_context.get("recommended_strategy", "hybrid"),
            "missing_information": repair_context.get("missing_information", [])[:5],
            "repair_action_types": [
                action.get("stage", "repair")
                for action in repair_context.get("repair_actions", []) or []
                if isinstance(action, dict)
            ],
        })

    return {
        "question_type": question_type.get("label", {}),
        "sample_query_intent": retrieval_memory.get("original_query", "")[:300],
        "steps": compact_steps,
        "verification_summaries": verification_summaries[-3:],
        "repair_events": repair_events[-3:],
        "final_outcome": {
            "overall_valid": bool(final_verification.get("verification_summary", {}).get("overall_valid")),
            "backtrack_attempts": backtrack_attempts,
            "confidence": final_verification.get("verification_summary", {}).get("confidence", ""),
        },
    }


def _default_experience_compilation_prompt() -> str:
    return """You are the Meta-level Supervisor Agent for an Agentic RAG system.

Your task is to compile a successful retrieval trajectory into a compact reusable experience.
Do not store raw evidence, source text, retrieved context, entity-specific facts, or long execution logs.
Extract only transferable retrieval knowledge: topology, tool policy, validation signals, repair policy, evidence requirements, and anti-patterns.

Input is a compact trace, not the full retrieval content:
{compact_trace_json}

Return one valid JSON object only. Use this schema:
{{
  "name": "short reusable pattern name",
  "retrieval_pattern": "one or two sentences describing the generalized multi-step retrieval method",
  "applicability": "when this pattern should be used",
  "tool_policy": {{
    "preferred_schedule": [
      {{"step_role": "role of the step", "preferred_tool": "knowledge_graph|vector_text|hybrid|symbolic_operation", "reason": "short reason"}}
    ],
    "routing_rule": "short rule for selecting tools"
  }},
  "failure_diagnosis": "what failure mode was observed or should be watched for",
  "repair_policy": "how to repair from the earliest failed step while preserving the trusted prefix",
  "evidence_requirements": ["minimal evidence condition 1", "minimal evidence condition 2"],
  "anti_patterns": ["mistake to avoid 1", "mistake to avoid 2"],
  "abstraction": {{
    "graph_topology": "generalized topology without concrete entity values",
    "semantic_axes": ["generalized relation or modality axis"],
    "trusted_prefix_rule": "how to decide which prior steps can be reused"
  }},
  "prompt_hint": "short instruction that can be injected into future retrieval prompts"
}}
{completion_delimiter}"""


def _fallback_compiled_experience(
        compact_trace: dict,
        abstraction: dict,
        error_type: str,
        repair_context: dict,
) -> dict:
    tool_schedule = abstraction.get("tool_schedule", []) if isinstance(abstraction, dict) else []
    return {
        "name": "compiled_retrieval_pattern",
        "retrieval_pattern": "Reuse the generalized decomposition topology and tool schedule, then verify each step before generation.",
        "applicability": "Queries with a similar reasoning type, step count, and generalized triple topology.",
        "tool_policy": {
            "preferred_schedule": tool_schedule,
            "routing_rule": "Start with the prior successful modality for each step; switch to hybrid retrieval at the first missing or conflicting evidence step.",
        },
        "failure_diagnosis": error_type,
        "repair_policy": (
            "Preserve the trusted prefix, clear evidence from the first invalid step onward, "
            "and rerun the suffix with the recommended repair strategy."
        ),
        "evidence_requirements": [
            "Each planned step must produce non-empty evidence.",
            "Final evidence must answer the original query without unresolved missing information.",
        ],
        "anti_patterns": [
            "Do not restart the entire plan when a trusted prefix is available.",
            "Do not store or reuse raw retrieved context as experience.",
        ],
        "abstraction": {
            "graph_topology": abstraction.get("graph_topology", {}),
            "semantic_axes": [
                item.get("strategy", "")
                for item in tool_schedule
                if isinstance(item, dict)
            ],
            "trusted_prefix_rule": "Keep steps before the first invalid step locked as trusted evidence.",
        },
        "prompt_hint": (
            "Use the stored topology and tool schedule as hints; if verification fails, "
            f"repair with {repair_context.get('recommended_strategy', 'hybrid')} from the earliest failed step."
        ),
        "compile_source": "fallback",
        "compact_trace_summary": {
            "step_count": len(compact_trace.get("steps", [])),
            "backtrack_attempts": compact_trace.get("final_outcome", {}).get("backtrack_attempts", 0),
        },
    }


async def _compile_experience_with_llm(
        compact_trace: dict,
        abstraction: dict,
        error_type: str,
        repair_context: dict,
        global_config: dict[str, Any],
        query_param: QueryParam,
) -> dict:
    template = PROMPTS.get("experience_compilation") or _default_experience_compilation_prompt()
    prompt = template.format(
        compact_trace_json=json.dumps(compact_trace, ensure_ascii=False, indent=2),
        completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", ""),
    )
    try:
        use_model_func = query_param.model_func or global_config["llm_model_func"]
        result = await use_model_func(prompt, _priority=5)
        compiled = _extract_json_object_from_text(result)
        print(f"往经验库中存储的内容{compiled}")
        if isinstance(compiled, dict) and compiled:
            compiled["compile_source"] = "llm"
            return compiled
    except Exception as e:
        logger.warning(f"Experience compilation with LLM failed: {e}")

    return _fallback_compiled_experience(
        compact_trace=compact_trace,
        abstraction=abstraction,
        error_type=error_type,
        repair_context=repair_context,
    )


def _relax_experience_abstraction(existing: dict, candidate: dict) -> tuple[dict, list[dict]]:
    relaxed = dict(existing or candidate or {})
    existing_topology = (existing or {}).get("graph_topology", {})
    candidate_topology = (candidate or {}).get("graph_topology", {})
    existing_steps = existing_topology.get("steps", []) or []
    candidate_steps = candidate_topology.get("steps", []) or []
    relaxations = []
    merged_steps = []

    for index, candidate_step in enumerate(candidate_steps):
        old_step = existing_steps[index] if index < len(existing_steps) and isinstance(existing_steps[index], dict) else {}
        merged_step = dict(old_step or candidate_step)
        old_rel = ((old_step.get("triple") or {}).get("rel") or "").strip()
        new_rel = ((candidate_step.get("triple") or {}).get("rel") or "").strip()
        if old_rel and new_rel and old_rel != new_rel:
            axis = sorted({old_rel, new_rel})
            merged_step["semantic_axis"] = axis
            relaxations.append({
                "step_order": candidate_step.get("order", index + 1),
                "from_predicate": old_rel,
                "observed_predicate": new_rel,
                "prompt_instruction": (
                    "Relax this predicate as a semantic axis only when the current "
                    "query intent and verified evidence support the mapping."
                ),
            })
        if not old_step:
            merged_step = dict(candidate_step)
        merged_steps.append(merged_step)

    if merged_steps:
        relaxed["graph_topology"] = dict(existing_topology or candidate_topology)
        relaxed["graph_topology"]["steps"] = merged_steps
        relaxed["graph_topology"]["step_count"] = max(
            len(existing_steps),
            len(candidate_steps),
            relaxed["graph_topology"].get("step_count", 0),
        )
    relaxed["generalization_hint"] = (candidate or {}).get(
        "generalization_hint",
        relaxed.get("generalization_hint", ""),
    )
    return relaxed, relaxations


def _consolidate_experience_record(
        existing: dict,
        candidate: dict,
        similarity: float,
        now: float,
) -> tuple[dict, list[dict]]:
    updated = dict(existing or {})
    candidate_examples = candidate.get("examples", [])
    if not isinstance(candidate_examples, list):
        candidate_examples = []
    existing_examples = updated.get("examples", [])
    if not isinstance(existing_examples, list):
        existing_examples = []

    success_count = updated.get("success_count", 0) + 1
    activation_count = updated.get("activation_count", updated.get("success_count", 0)) + 1
    relaxed_abstraction, relaxations = _relax_experience_abstraction(
        updated.get("abstraction", {}),
        candidate.get("abstraction", {}),
    )
    updated.update({
        "updated_at": now,
        "success_count": success_count,
        "activation_count": activation_count,
        "confidence_weight": min(1.0, success_count / (success_count + 1)),
        "last_topology_similarity": similarity,
        "abstraction": relaxed_abstraction,
        "examples": (existing_examples + candidate_examples)[-5:],
    })
    if relaxations:
        prior_relaxations = updated.get("topology_relaxations", [])
        if not isinstance(prior_relaxations, list):
            prior_relaxations = []
        updated["topology_relaxations"] = (prior_relaxations + relaxations)[-10:]
    return updated, relaxations


async def _store_successful_experience_memory(
        experience_memory_kv: BaseKVStorage | None,
        question_type: dict,
        retrieval_memory: dict,
        execution_plan: list[dict],
        step_results: list[dict],
        final_verification: dict,
        backtrack_attempts: int,
        query_param: QueryParam,
        global_config: dict[str, Any],
) -> None:
    if experience_memory_kv is None or not question_type:
        return
    if not final_verification.get("verification_summary", {}).get("overall_valid"):
        return

    now = time.time()
    failed_verification = next(
        (
            item for item in retrieval_memory.get("verifications", [])
            if not item.get("verification_summary", {}).get("overall_valid")
        ),
        {},
    )
    repair_event = (retrieval_memory.get("backtracking_events") or [{}])[-1]
    repair_context = repair_event.get("repair_context", {}) or {}
    failed_step_id = _safe_step_id(
        repair_context.get("failed_step_id")
        or repair_event.get("start_step_id")
        or failed_verification.get("verification_summary", {}).get("first_invalid_step")
    )
    failed_step = _find_step_by_id(execution_plan, failed_step_id) if failed_step_id else {}
    successful_step_result = _find_step_by_id(step_results, failed_step_id) if failed_step_id else (
        step_results[-1] if step_results else {}
    )
    step_verification = _find_step_by_id(
        failed_verification.get("step_verifications", []),
        failed_step_id,
    ) if failed_step_id else {}
    error_type = (
        step_verification.get("error_type")
        or repair_context.get("reason")
        or failed_verification.get("verification_summary", {}).get("reason")
        or ("self_healed" if backtrack_attempts else "direct_success")
    )
    abstraction = _build_experience_abstraction(
        question_type=question_type,
        execution_plan=execution_plan,
        retrieval_decisions=retrieval_memory.get("retrieval_decisions", []),
        step_results=step_results,
        retrieval_memory=retrieval_memory,
        final_verification=final_verification,
        backtrack_attempts=backtrack_attempts,
    )
    compact_trace = _build_compact_experience_trace(
        question_type=question_type,
        execution_plan=execution_plan,
        retrieval_decisions=retrieval_memory.get("retrieval_decisions", []),
        step_results=step_results,
        retrieval_memory=retrieval_memory,
        final_verification=final_verification,
        backtrack_attempts=backtrack_attempts,
    )
    compiled_experience = await _compile_experience_with_llm(
        compact_trace=compact_trace,
        abstraction=abstraction,
        error_type=error_type,
        repair_context=repair_context,
        global_config=global_config,
        query_param=query_param,
    )
    compiled_abstraction = compiled_experience.get("abstraction", {})
    similarity_abstraction = {
        "compiled": compiled_abstraction,
        "structural": abstraction,
    }
    experience_id = compute_mdhash_id(
        json.dumps(similarity_abstraction, ensure_ascii=False, sort_keys=True),
        prefix="experience-",
    )

    new_experience = {
        "id": experience_id,
        "created_at": now,
        "updated_at": now,
        "success_count": 1,
        "activation_count": 1,
        "confidence_weight": 0.5,

        "structural_topology": abstraction,
        "compiled_experience": compiled_experience,
        # 3. 元数据与真实案例追踪（保持滑动窗口限制，利于 Debug 和人工审计）
        "problem": {
            "question_type_label": question_type.get("label", {}),
            "failed_step_id": failed_step_id,
        },

        "outcome": {
            "backtrack_attempts": backtrack_attempts,
            "verification_summary": {
                "confidence": final_verification.get("verification_summary", {}).get("confidence", "high"),
                "reason": final_verification.get("verification_summary", {}).get("reason", "")
            }
        },
        "examples": [{
            "query": retrieval_memory.get("original_query", ""),
            "created_at": now,
        }],
    }

    try:
        type_key = question_type.get("key", "")
        existing = await experience_memory_kv.get_by_id(type_key) or {
            "question_type": question_type.get("label", {}),
            "experiences": [],
            "created_at": now,
        }
        experiences = existing.get("experiences", [])
        if not isinstance(experiences, list):
            experiences = []

        novelty_threshold = _as_float(
            getattr(query_param, "experience_memory_topology_similarity_threshold", 0.82),
            0.82,
        )
        best_index = None
        best_similarity = -1.0
        for index, item in enumerate(experiences):
            if not isinstance(item, dict):
                continue
            #只是简单的拓扑相似度
            similarity = _topology_similarity(
                abstraction,
                item.get("abstraction", {}),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is None or best_similarity < novelty_threshold:
            experiences.append(new_experience)
            store_event = {
                "decision": "append_new_topology",
                "question_type_key": type_key,
                "experience_id": experience_id,
                "max_similarity": max(best_similarity, 0.0),
                "novelty_threshold": novelty_threshold,
            }
        else:
            consolidated, relaxations = _consolidate_experience_record(
                experiences[best_index],
                new_experience,
                best_similarity,
                now,
            )
            experiences[best_index] = consolidated
            experience_id = consolidated.get("id", experience_id)
            store_event = {
                "decision": "consolidate_existing_topology",
                "question_type_key": type_key,
                "experience_id": experience_id,
                "matched_experience_id": consolidated.get("id"),
                "max_similarity": best_similarity,
                "novelty_threshold": novelty_threshold,
                "relaxation_count": len(relaxations),
            }

        experiences = sorted(
            experiences,
            key=lambda item: (
                item.get("confidence_weight", 0) if isinstance(item, dict) else 0,
                item.get("activation_count", item.get("success_count", 0)) if isinstance(item, dict) else 0,
                item.get("updated_at", 0) if isinstance(item, dict) else 0,
            ),
            reverse=True,
        )
        max_records = max(
            1,
            _safe_step_id(
                getattr(query_param, "experience_memory_max_records_per_type", 20),
                5,
            ),
        )
        existing.update({
            "question_type": question_type.get("label", {}),
            "sample_query": question_type.get("sample_query", ""),
            "experiences": experiences[:max_records],
            "updated_at": now,
        })
        await experience_memory_kv.upsert({type_key: existing})
        _update_blackboard(retrieval_memory, Mexp={
            "last_store_event": store_event,
            "store_events": (
                _blackboard(retrieval_memory).setdefault("Mexp", {}).get("store_events", [])
                + [store_event]
            )[-10:],
        })
        _remember(retrieval_memory, "experience_memory_stored", {
            **store_event,
            "failed_step_id": failed_step_id,
            "error_type": error_type,
            "backtrack_attempts": backtrack_attempts,
        })
    except Exception as e:
        logger.warning(f"Experience memory store failed: {e}")


def _build_repair_context(
        verification: dict,
        backtracking_plan: dict | None = None,
) -> dict:
    summary = verification.get("verification_summary", {}) if verification else {}
    answer_verification = verification.get("answer_verification", {}) if verification else {}
    backtracking_plan = backtracking_plan or {}

    return {
        "failed_step_id": backtracking_plan.get("start_step_id") or summary.get("first_invalid_step"),
        "reason": (
            backtracking_plan.get("reason")
            or summary.get("reason")
            or answer_verification.get("reason")
        ),
        "missing_information": answer_verification.get("missing_information", []),
        "recommended_strategy": backtracking_plan.get("recommended_strategy", "hybrid"),
        "repair_query": backtracking_plan.get("query", ""),
        "repair_actions": backtracking_plan.get("repair_actions", []),
    }


def _format_repair_context_for_query(repair_context: dict | None) -> str:
    if not repair_context:
        return ""

    return (
        "Backtracking repair context:\n"
        f"- Previous failed step id: {repair_context.get('failed_step_id')}\n"
        f"- Failure reason: {repair_context.get('reason')}\n"
        f"- Missing information: {json.dumps(repair_context.get('missing_information', []), ensure_ascii=False)}\n"
        f"- Repair query: {repair_context.get('repair_query', '')}\n"
        "Use this context to avoid repeating the previous retrieval mistake."
    )


async def _execute_retrieval_step(
        query: str,
        step: dict,
        strategy: str,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage | None,
        global_config: dict[str, Any],
        query_param: QueryParam,
        previous_step_results: list[dict],
        use_model_func,
        repair_context: dict | None = None,
        experience_context: str = "",
) -> dict:
    step_id = step.get("step_id")
    sub_query = step.get("sub_query", "")
    hints = [
        item for item in [
            _format_repair_context_for_query(repair_context),
            experience_context,
        ]
        if item
    ]
    repaired_sub_query = f"{sub_query}\n\n" + "\n\n".join(hints) if hints else sub_query
    retrieval_step = dict(step)
    retrieval_step["sub_query"] = repaired_sub_query
    normalized_strategy = (strategy or "graph").lower()

    def unpack_kg_result(result):
        if len(result) == 5:
            return result
        if len(result) == 4:
            q1, q2, q3, entity_level_context = result
            return q1, q2, q3, entity_level_context, {}
        raise ValueError(f"Unexpected KG retrieval result length: {len(result)}")

    if normalized_strategy in {"vector", "vector_text"}:
        context = await vector_text_retrieval(
            sub_query=repaired_sub_query,
            entities_vdb=entities_vdb,
            top_k=3,
            query_param=query_param,
        )
        return {
            "step_id": step_id,
            "strategy": "Vector_Text",
            "context": context,
            "repair_context": repair_context,
        }

    if normalized_strategy in {"hybrid", "rerun_suffix"}:
        vec_ctx = await vector_text_retrieval(
            sub_query=repaired_sub_query,
            entities_vdb=entities_vdb,
            top_k=5,
            query_param=query_param,
        )
        chunk_ctx = ""
        if chunks_vdb is not None:
            chunk_ctx = await vector_chunk_retrieval(
                sub_query=repaired_sub_query,
                chunks_vdb=chunks_vdb,
                top_k=5,
                query_param=query_param,
            )
        q1, q2, q3, entity_level_context, unknown_map = unpack_kg_result(await kg_triple_retrieval_for_step(
            step=retrieval_step,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            global_config=global_config,
            query_param=query_param,
        ))
        kg_ctx = await extract_evidence_from_q3(q3, entities_vdb)
        if entity_level_context:
            kg_ctx = f"{kg_ctx}\n{entity_level_context}" if kg_ctx else entity_level_context
        context = _dedupe_context_blocks([vec_ctx, chunk_ctx, kg_ctx])
        return {
            "step_id": step_id,
            "strategy": "Hybrid",
            "context": context,
            "repair_context": repair_context,
            "debug": {
                "q1_size": len(q1),
                "q2_size": len(q2),
                "q3_size": len(q3),
                "unknown_map": unknown_map,
                "chunk_ctx_used": bool(chunk_ctx),
            },
        }

    if normalized_strategy in {"graph", "kg_triple", "kg"}:
        q1, q2, q3, entity_level_context, unknown_map = unpack_kg_result(await kg_triple_retrieval_for_step(
            step=retrieval_step,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            global_config=global_config,
            query_param=query_param,
        ))
        context = await extract_evidence_from_q3(q3, entities_vdb)
        if entity_level_context:
            context = f"{context}\n{entity_level_context}" if context else entity_level_context
        return {
            "step_id": step_id,
            "strategy": "KG_Triple",
            "context": context,
            "repair_context": repair_context,
            "debug": {"q1_size": len(q1), "q2_size": len(q2), "q3_size": len(q3), "unknown_map": unknown_map},
        }

    operation_context = await operation_reasoning_for_step(
        query=query,
        step=retrieval_step,
        previous_step_results=previous_step_results,
        original_query=query,
        global_config=global_config,
        query_param=query_param,
        model_func=use_model_func,
    )
    return {
        "step_id": step_id,
        "strategy": "Operation",
        "context": operation_context,
        "repair_context": repair_context,
    }


async def _run_retrieval_plan(
        query: str,
        execution_plan: list[dict],
        decision_map: dict,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage | None,
        global_config: dict[str, Any],
        query_param: QueryParam,
        use_model_func,
        memory: dict | None = None,
        start_step_id: int | None = None,
        trusted_step_results: list[dict] | None = None,
        repair_context: dict | None = None,
        experience_context: str = "",
        iteration: int = 0,
) -> tuple[list[dict], str]:
    step_results = list(trusted_step_results or [])
    all_contexts = [
        item.get("context", "")
        for item in step_results
        if isinstance(item, dict)
    ]

    for step in execution_plan or []:
        step_id = _safe_step_id(step.get("step_id"))
        #在回溯重新检测的时候可以定位到首次出错的地方
        if start_step_id is not None and step_id < start_step_id:
            continue

        sub_query = step.get("sub_query", "")
        decision = decision_map.get(step_id, {})
        strategy = decision.get("retrieval_method", "graph")

        if repair_context and step_id >= _safe_step_id(repair_context.get("failed_step_id"), step_id):
            strategy = repair_context.get("recommended_strategy") or "hybrid"

        print(f"Step {step_id} | Strategy: {strategy} | Query: {sub_query[:80]}...")
        step_result = await _execute_retrieval_step(
            query=query,
            step=step,
            strategy=strategy,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            chunks_vdb=chunks_vdb,
            global_config=global_config,
            query_param=query_param,
            previous_step_results=step_results,
            use_model_func=use_model_func,
            repair_context=repair_context if start_step_id is not None else None,
            experience_context=experience_context,
        )
        step_results.append(step_result)
        all_contexts.append(step_result.get("context", ""))

        if memory is not None:
            memory.setdefault("steps", []).append(step_result)
            _append_blackboard_evidence(memory, step_result, iteration=iteration)
            _remember(memory, "step_retrieved", {
                "step_id": step_id,
                "strategy": step_result.get("strategy"),
                "has_context": bool(step_result.get("context", "").strip()),
                "repair_context": step_result.get("repair_context"),
                "iteration": iteration,
            })

    return step_results, _dedupe_context_blocks(all_contexts)


async def operation_reasoning_for_step(
        query: str,
        step: dict,
        previous_step_results: list[dict],
        original_query: str,
        global_config: dict[str, Any],
        query_param: QueryParam | None = None,
        model_func=None,
) -> str:

    sub_query = step.get("sub_query", "")
    step_id = step.get("step_id", "")
    previous_results_text = _format_operation_step_results(previous_step_results)
    # response_type = getattr(query_param, "response_type", "Multiple Paragraphs") if query_param else "Multiple Paragraphs"
    language = global_config.get("addon_params", {}).get(
        "language", PROMPTS.get("DEFAULT_LANGUAGE", "English")
    )
    result = ""
    try:
        template_prompt = PROMPTS.get("Operation_Respond")
        if template_prompt:
            prompt = template_prompt.format(
                original_query = query,
                sub_query=sub_query,
                step_id=step_id,
                previous_results_text=previous_results_text,
                language=language,
                completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
            )

            use_model_func = global_config["llm_model_func"]
            respond_result = await use_model_func(prompt, _priority=5)
            result=respond_result
    except Exception as e:
        logger.warning(f"Strategy Operation failed: {e}")
    return (
        result.replace(prompt, "")
        .replace("<system>", "")
        .replace("</system>", "")
        .strip()
    )


async def kg_query(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        chunks_vdb: BaseVectorStorage = None,
        experience_memory_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        use_model_func = partial(use_model_func, _priority=5)

    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )


    retrieval_memory = _make_retrieval_memory(query)
    relation_list = await get_unique_relationship_types(relationships_vdb)
    type_list =await get_unique_entity_types(entities_vdb)
    # _update_blackboard(retrieval_memory, static_boundary={
    #     "entity_types": type_list,
    #     "relationship_types": relation_list,
    # })
    # _persist_blackboard_snapshot(
    #     retrieval_memory,
    #     global_config,
    #     status="initialized",
    # )
    # 1. 查询分解
    decomposition_result = await decompose_multi_hop_query(
        type_list=type_list,
        relation_list=relation_list,
        question=query,
        param=query_param,
        global_config=global_config,
        max_steps=5,  # 可根据需要调整或从 param 中获取
        hashing_kv=hashing_kv
    )
    # 安全提取（推荐写法）
    if not decomposition_result:
        # LLM 调用失败的情况
        question_analysis = {}
        execution_plan = []
    else:
        question_analysis = decomposition_result.get("question_analysis", {})
        execution_plan = decomposition_result.get("execution_plan", [])
    question_intent = (
        question_analysis.get("answer_type")
        or question_analysis.get("question_intent")
        or question_analysis.get("target_type")
        or ""
    )
    retrieval_memory["decomposition"] = decomposition_result or {}
    # _update_blackboard(
    #     retrieval_memory,
    #     P=execution_plan,
    #     static_boundary={
    #         "question_intent": question_intent,
    #         "reasoning_type": decomposition_result.get("reasoning_type", "multi_hop") if decomposition_result else "multi_hop",
    #     },
    # )
    # 记忆json文件
    _remember(retrieval_memory, "decomposed", {
        "reasoning_type": decomposition_result.get("reasoning_type", "multi_hop") if decomposition_result else "multi_hop",
        "step_count": len(execution_plan),
        "question_intent": question_intent,
    })
    # 记忆json文件
    experience_question_type = _build_experience_question_type(query, decomposition_result)

    experience_context = ""
    if getattr(query_param, "enable_experience_memory", True):
        #根据传入的问题类型（question_type），从键值存储（KV Storage）中读取相关的“经验记忆”（Experience Memories），
        # 并按照“成功次数”和“更新时间”进行排序，最后返回质量最高的前 top_k 条经验。
        retrieved_experiences = await _load_experience_memories(
            experience_memory_kv,
            experience_question_type,
            top_k=_safe_step_id(getattr(query_param, "experience_memory_top_k", 3), 3),
        )
        #将上一步筛选出来的优质“经验记忆”（experiences 列表）进行文本格式化，拼装成一段清晰的 Prompt（提示词）上下文，
        experience_context = _format_experience_context_for_query(retrieved_experiences)
        retrieval_memory["experience_memory"] = {
            "question_type": experience_question_type,
            "matched_count": len(retrieved_experiences),
            "matched_experiences": retrieved_experiences,
        }
        # _update_blackboard(retrieval_memory, Mexp={
        #     "enabled": True,
        #     "question_type": experience_question_type,
        #     "matched_count": len(retrieved_experiences),
        #     "matched_experiences": retrieved_experiences,
        # })
        if retrieved_experiences:
            _remember(retrieval_memory, "experience_memory_loaded", {
                "question_type_key": experience_question_type.get("key"),
                "matched_count": len(retrieved_experiences),
            })
    # else:
        # _update_blackboard(retrieval_memory, Mexp={"enabled": False})
    # _persist_blackboard_snapshot(
    #     retrieval_memory,
    #     global_config,
    #     status="experience_loaded",
    # )
    #进入检索的部分
    # ==================== 新增：Tool Calling 决定检索策略 ====================
    retrieval_decisions = await decide_retrieval_strategies(
        execution_plan=execution_plan,
        relation_list=relation_list,
        global_config=global_config,
        type_list=type_list,
        model_func=use_model_func,
        experience_context=experience_context,
    )
    # retrieval_decisions: List[dict]，每个元素包含 step_id 和 recommended_retrieval
    retrieval_memory["retrieval_decisions"] = retrieval_decisions or []
    # _update_blackboard(retrieval_memory, D=retrieval_decisions or [])
    # _persist_blackboard_snapshot(
    #     retrieval_memory,
    #     global_config,
    #     status="routed",
    # )
    decision_map = {
        _safe_step_id(d.get("step_id")): d
        for d in retrieval_decisions or []
        if isinstance(d, dict)
    }
    # ==================== 检索阶段（支持混合策略）===================
    step_results, context = await _run_retrieval_plan(
        query=query,
        execution_plan=execution_plan,
        decision_map=decision_map,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        chunks_vdb=chunks_vdb,
        global_config=global_config,
        query_param=query_param,
        use_model_func=use_model_func,
        memory=retrieval_memory,
        experience_context=experience_context,# ⭐ 【在这里真正生效】作为参考上下文传入具体的 RAG 执行器中！
        iteration=0,
    )
    # _persist_blackboard_snapshot(
    #     retrieval_memory,
    #     global_config,
    #     status="retrieved",
    # )

    print(f"第一步 context 长度: {len(context)}")

    verification = await verify_execution_plan_fine_grained(
        question=query,
        execution_plan=execution_plan,
        step_results=step_results,  # 你执行过程中收集的结果
        reasoning_type=decomposition_result.get("reasoning_type", "multi_hop") if decomposition_result else "multi_hop",
        global_config=global_config
    )
    retrieval_memory.setdefault("verifications", []).append(verification)
    # 向系统的“黑板”（共享内存/状态机）中追加一条最新的验证记录（Verification Record）
    # ，并自动附带当前的迭代轮数和时间戳
    # _append_blackboard_verification(retrieval_memory, verification)
    # _persist_blackboard_snapshot(
    #     retrieval_memory,
    #     global_config,
    #     status="verified",
    # )

    max_backtrack_retries = max(
        0,
        _safe_step_id(getattr(query_param, "max_backtrack_retries", 2), 2),
    )
    retrieval_memory["max_backtrack_retries"] = max_backtrack_retries
    backtrack_attempt = 0

    while not verification.get("verification_summary", {}).get("overall_valid"):
        last_valid = verification.get("verification_summary", {}).get("last_valid_step_id", 0)

        if backtrack_attempt >= max_backtrack_retries:
            print(
                f"⚠️ 达到最大回溯次数 {max_backtrack_retries}，"
                "将使用当前最完整上下文继续生成"
            )
            _remember(retrieval_memory, "backtracking_stopped", {
                "reason": "max_backtrack_retries_reached",
                "attempts_used": backtrack_attempt,
                "max_backtrack_retries": max_backtrack_retries,
                "last_valid_step_id": last_valid,
            })
            break

        backtrack_attempt += 1
        print(
            f"⚠️ 在第 {last_valid + 1} 步首次发现问题，"
            f"可信前缀为 {last_valid} 步，开始第 {backtrack_attempt}/"
            f"{max_backtrack_retries} 次回溯检索"
        )

        backtracking_plan = (verification.get("backtracking_plan") or [{}])[0]
        start_step_id = _safe_step_id(
            backtracking_plan.get("start_step_id"),
            verification.get("verification_summary", {}).get("first_invalid_step", last_valid + 1),
        )
        trusted_step_results = [
            item for item in step_results
            if _safe_step_id(item.get("step_id")) < start_step_id
        ]
        #对系统的“黑板”（共享内存）执行“截断”操作——
        # 清除所有不不可信的步骤，保留并“锁定”可信的步骤，同时切换系统的控制状态，并记录下一笔详细的审计日志（Event Log）
        discarded_suffix = _truncate_blackboard_for_backtrack(
            retrieval_memory,
            start_step_id=start_step_id,
            trusted_step_results=trusted_step_results,
        )
        repair_context = _build_repair_context(verification, backtracking_plan)
        retrieval_memory.setdefault("backtracking_events", []).append({
            "attempt": backtrack_attempt,
            "start_step_id": start_step_id,
            "trusted_step_ids": [item.get("step_id") for item in trusted_step_results],
            "discarded_step_ids": [item.get("step_id") for item in discarded_suffix],
            "repair_context": repair_context,
            "verification_before_retry": verification,
        })
        _remember(retrieval_memory, "backtracking_started", {
            "attempt": backtrack_attempt,
            "start_step_id": start_step_id,
            "repair_context": repair_context,
            "discarded_step_ids": [item.get("step_id") for item in discarded_suffix],
        })

        # _update_blackboard(
        #     retrieval_memory,
        #     iteration=backtrack_attempt,
        #     control_state="repairing",
        # )
        # _persist_blackboard_snapshot(
        #     retrieval_memory,
        #     global_config,
        #     status="backtracking",
        # )
        step_results, context = await _run_retrieval_plan(
            query=query,
            execution_plan=execution_plan,
            decision_map=decision_map,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            chunks_vdb=chunks_vdb,
            global_config=global_config,
            query_param=query_param,
            use_model_func=use_model_func,
            memory=retrieval_memory,
            start_step_id=start_step_id,
            trusted_step_results=trusted_step_results,
            repair_context=repair_context,
            experience_context=experience_context,# ⭐ 【回溯阶段继续参考】引导 Agent 在回溯重试时做出更聪明的工具调度或谓词映射决策
            iteration=backtrack_attempt,
        )
        # _persist_blackboard_snapshot(
        #     retrieval_memory,
        #     global_config,
        #     status="repair_retrieved",
        # )
        verification = await verify_execution_plan_fine_grained(
            question=query,
            execution_plan=execution_plan,
            step_results=step_results,
            reasoning_type=decomposition_result.get("reasoning_type", "multi_hop") if decomposition_result else "multi_hop",
            global_config=global_config
        )
        retrieval_memory.setdefault("verifications", []).append(verification)
        # _append_blackboard_verification(retrieval_memory, verification)
        # _persist_blackboard_snapshot(
        #     retrieval_memory,
        #     global_config,
        #     status="repair_verified",
        # )

    if verification.get("verification_summary", {}).get("overall_valid"):
        # _update_blackboard(retrieval_memory, control_state="accepted")
        if backtrack_attempt:
            print(f"✅ 第 {backtrack_attempt} 次回溯检索后验证通过")
        else:
            print("✅ 全流程验证通过")
            # 更新，写入记忆json
        await _store_successful_experience_memory(
            experience_memory_kv=experience_memory_kv,
            question_type=experience_question_type,
            retrieval_memory=retrieval_memory,
            execution_plan=execution_plan,
            step_results=step_results,
            final_verification=verification,
            backtrack_attempts=backtrack_attempt,
            query_param=query_param,
            global_config=global_config,
        )
        # _persist_blackboard_snapshot(
        #     retrieval_memory,
        #     global_config,
        #     status="accepted",
        # )
    else:
        _update_blackboard(retrieval_memory, control_state="fallback_generation")
        _remember(retrieval_memory, "experience_memory_pruned_failed_path", {
            "reason": "verification_failed_after_max_backtrack",
            "backtrack_attempts": backtrack_attempt,
            "max_backtrack_retries": max_backtrack_retries,
        })
        # _persist_blackboard_snapshot(
        #     retrieval_memory,
        #     global_config,
        #     status="fallback_generation",
        # )
    global_config["last_retrieval_memory"] = retrieval_memory

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        question_intent=question_intent,
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(

        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    return response,context


def _extract_json_object_from_text(text: str) -> dict:
    """Parse the first JSON object from an LLM response."""
    if not text:
        return {}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        fixed = force_fix_json(match.group(0))
        try:
            return json.loads(fixed)
        except Exception:
            logger.error("Failed to parse JSON object from verifier output")
            return {}


def _safe_step_id(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_step_execution_records(
        execution_plan: list,
        step_results: list,
        max_context_chars: int = 8000
) -> list[dict]:
    """Join each planned step with its retrieved or inferred result."""
    result_by_step = {
        _safe_step_id(item.get("step_id"), index + 1): item
        for index, item in enumerate(step_results or [])
        if isinstance(item, dict)
    }

    records = []
    for index, step in enumerate(execution_plan or []):
        if not isinstance(step, dict):
            continue

        step_id = _safe_step_id(step.get("step_id"), index + 1)
        result = result_by_step.get(step_id, {})
        context = result.get("context", "")
        context_text = context if isinstance(context, str) else json.dumps(
            context, ensure_ascii=False
        )

        records.append({
            "step_id": step_id,
            "step_type": step.get("step_type", "retrieval"),
            "sub_query": step.get("sub_query", ""),
            "triple": step.get("triple", {}),
            "depends_on": step.get("depends_on", []),
            "strategy": result.get("strategy", ""),
            "context": context_text[:max_context_chars],
            "context_length": len(context_text),
            "has_result": bool(context_text.strip()),
        })

    return records


def _make_default_accept_verification(
        step_records: list[dict],
        reason: str = "Current evidence is sufficient to answer the question."
) -> dict:
    last_step_id = max([item.get("step_id", 0) for item in step_records], default=0)
    return {
        "verification_summary": {
            "overall_valid": True,
            "decision": "accept",
            "confidence": "medium",
            "reason": reason,
            "first_invalid_step": None,
            "last_valid_step_id": last_step_id,
            "trusted_prefix_length": last_step_id,
        },
        "answer_verification": {
            "is_reliable": True,
            "is_answerable": True,
            "supported_answer": None,
            "supporting_step_ids": [item.get("step_id") for item in step_records],
            "missing_information": [],
            "reason": reason,
        },
        "step_verifications": [],
        "backtracking_plan": [],
    }


def _make_default_backtrack_verification(
        step_records: list[dict],
        reason: str = "Current evidence is not sufficient to answer the question."
) -> dict:
    first_invalid_step = next(
        (item.get("step_id") for item in step_records if not item.get("has_result")),
        step_records[-1].get("step_id") if step_records else 1,
    )
    last_valid_step_id = max(first_invalid_step - 1, 0)
    return {
        "verification_summary": {
            "overall_valid": False,
            "decision": "backtrack",
            "confidence": "low",
            "reason": reason,
            "first_invalid_step": first_invalid_step,
            "last_valid_step_id": last_valid_step_id,
            "trusted_prefix_length": last_valid_step_id,
        },
        "answer_verification": {
            "is_reliable": False,
            "is_answerable": False,
            "supported_answer": None,
            "supporting_step_ids": [],
            "missing_information": [reason],
            "reason": reason,
        },
        "step_verifications": [],
        "backtracking_plan": [{
            "start_step_id": first_invalid_step,
            "reason": "insufficient_evidence",
            "recommended_strategy": "hybrid",
            "query": "",
            "reuse_trusted_steps": list(range(1, last_valid_step_id + 1)),
        }],
    }


def _normalize_reliability_decision(raw_data: dict, step_records: list[dict]) -> dict:
    answer_verification = raw_data.get("answer_verification", raw_data)
    is_reliable = bool(
        answer_verification.get("is_reliable")
        or answer_verification.get("is_answerable")
        or raw_data.get("overall_valid")
    )
    confidence = answer_verification.get("confidence", raw_data.get("confidence", "medium"))
    reason = answer_verification.get("reason", raw_data.get("reason", ""))

    if is_reliable:
        data = _make_default_accept_verification(step_records, reason or "Evidence verified.")
        data["verification_summary"]["confidence"] = confidence
        data["answer_verification"].update(answer_verification)
        data["answer_verification"]["is_reliable"] = True
        data["answer_verification"]["is_answerable"] = True
        return data

    data = _make_default_backtrack_verification(step_records, reason or "Evidence failed verification.")
    data["verification_summary"]["confidence"] = confidence
    data["answer_verification"].update(answer_verification)
    data["answer_verification"]["is_reliable"] = False
    return data


async def verify_current_evidence_reliability(
        question: str,
        execution_plan: list,
        step_results: list,
        reasoning_type: str,
        global_config: dict,
        max_context_chars: int = 8000
) -> dict:
    """
    First-pass gatekeeper using traditional heuristics (no LLM).
    Decides whether current retrieved/inferred evidence can be safely sent to final answer generator.
    
    Strategy:
    1. Check if all steps have results
    2. Verify dependency chains are intact
    3. Validate result quality (non-empty context)
    4. Accept if all steps have valid results, else backtrack
    """
    step_records = _normalize_step_execution_records(
        execution_plan=execution_plan,
        step_results=step_results,
        max_context_chars=max_context_chars,
    )

    if not step_records:
        return _make_default_backtrack_verification(
            step_records,
            "No execution records were available for verification."
        )

    if all(not item.get("has_result") for item in step_records):
        return _make_default_backtrack_verification(
            step_records,
            "All retrieved or inferred step results are empty."
        )
    
    # Traditional heuristic verification
    result_by_step = {item.get("step_id"): item for item in step_records}
    
    # Check dependencies and result validity
    all_valid = True
    first_invalid_step = None
    missing_info = []
    
    for record in step_records:
        step_id = record.get("step_id")
        has_result = record.get("has_result", False)
        depends_on = record.get("depends_on", [])
        
        # Check if this step has result
        if not has_result:
            all_valid = False
            if first_invalid_step is None:
                first_invalid_step = step_id
            missing_info.append(f"Step {step_id}: No result for query '{record.get('sub_query', '')}'")
            continue
        
        # Check if all dependencies are satisfied
        for dep_step_id in depends_on:
            dep_record = result_by_step.get(dep_step_id)
            if not dep_record or not dep_record.get("has_result"):
                all_valid = False
                if first_invalid_step is None:
                    first_invalid_step = step_id
                missing_info.append(f"Step {step_id}: Dependency on step {dep_step_id} failed")
    
    # If all steps are valid, accept
    if all_valid:
        return _make_default_accept_verification(
            step_records,
            "All steps have valid results and dependencies are satisfied."
        )
    
    # Otherwise, backtrack
    if first_invalid_step is None:
        first_invalid_step = step_records[-1].get("step_id", 1)
    
    return {
        "verification_summary": {
            "overall_valid": False,
            "decision": "backtrack",
            "confidence": "medium",
            "reason": "Some steps lack valid results or have unmet dependencies.",
            "first_invalid_step": first_invalid_step,
            "last_valid_step_id": max(first_invalid_step - 1, 0),
            "trusted_prefix_length": max(first_invalid_step - 1, 0),
        },
        "answer_verification": {
            "is_reliable": False,
            "is_answerable": False,
            "supported_answer": None,
            "supporting_step_ids": [r.get("step_id") for r in step_records if r.get("has_result")],
            "missing_information": missing_info,
            "reason": "Some steps lack valid results.",
        },
        "step_verifications": [],
        "backtracking_plan": [],
    }


def _build_backtracking_plan_from_diagnosis(
        question: str,
        execution_plan: list,
        step_records: list[dict],
        step_verifications: list[dict],
        answer_verification: dict,
        reasoning_type: str
) -> list[dict]:
    plan_by_id = {
        _safe_step_id(step.get("step_id"), index + 1): step
        for index, step in enumerate(execution_plan or [])
        if isinstance(step, dict)
    }
    verification_by_id = {
        _safe_step_id(item.get("step_id")): item
        for item in step_verifications or []
        if isinstance(item, dict)
    }

    invalid_ids = [
        step_id for step_id in sorted(plan_by_id)
        if verification_by_id.get(step_id, {}).get("status") != "valid"
    ]
    if invalid_ids:
        first_invalid_step = invalid_ids[0]
    else:
        first_invalid_step = step_records[-1].get("step_id") if step_records else 1

    failed_step = plan_by_id.get(first_invalid_step, {})
    failed_check = verification_by_id.get(first_invalid_step, {})
    error_type = failed_check.get("error_type") or "insufficient_evidence"
    missing_info = answer_verification.get("missing_information", [])

    repair_query = (
        f"Original question: {question}\n"
        f"Failed step id: {first_invalid_step}\n"
        f"Failed sub-query: {failed_step.get('sub_query', '')}\n"
        f"Failed triple: {json.dumps(failed_step.get('triple', {}), ensure_ascii=False)}\n"
        f"Missing information: {json.dumps(missing_info, ensure_ascii=False)}\n"
        f"Failure reason: {failed_check.get('error_reason') or answer_verification.get('reason', '')}"
    )

    if error_type in {"wrong_entity", "dependency_error"}:
        recommended_strategy = "rerun_suffix"
    elif error_type == "operation_error":
        recommended_strategy = "operation"
    elif error_type in {"empty_result", "wrong_relation", "insufficient_evidence", "unsupported_fact"}:
        recommended_strategy = "hybrid"
    else:
        recommended_strategy = "hybrid"

    backtracking_plan = [{
        "start_step_id": first_invalid_step,
        "reason": error_type,
        "recommended_strategy": recommended_strategy,
        "query": repair_query,
        "reuse_trusted_steps": [
            step_id for step_id in sorted(plan_by_id)
            if step_id < first_invalid_step
            and verification_by_id.get(step_id, {}).get("status") == "valid"
        ],
        "repair_actions": [],
    }]

    if recommended_strategy in {"hybrid", "rerun_suffix"}:
        backtracking_plan[0]["repair_actions"].extend([
            {
                "stage": "graph_backtrack",
                "method": "graph",
                "description": "Retry KG relation matching and allow path fallback around the failed triple.",
            },
            {
                "stage": "vector_backtrack",
                "method": "vector",
                "description": "Search text chunks with both the failed sub-query and the original question.",
            },
        ])

    if recommended_strategy in {"operation", "rerun_suffix"} or reasoning_type == "comparison":
        backtracking_plan[0]["repair_actions"].append({
            "stage": "operation_recompute",
            "method": "operation",
            "description": "Recompute the derived answer using only trusted previous results.",
        })

    return backtracking_plan


async def diagnose_unreliable_steps(
        question: str,
        execution_plan: list,
        step_results: list,
        reasoning_type: str,
        global_config: dict,
        answer_verification: dict,
        max_context_chars: int = 8000
) -> dict:
    """
    Second-pass verifier using traditional heuristics (no LLM).
    Only called when current evidence is not reliable.
    Finds the earliest bad step and creates a staged backtracking plan.
    
    Strategy:
    1. Scan step_records to find first step without result
    2. Check dependency chain to identify root cause
    3. Classify error type based on dependency/result status
    4. Build repair strategy accordingly
    """
    step_records = _normalize_step_execution_records(
        execution_plan=execution_plan,
        step_results=step_results,
        max_context_chars=max_context_chars,
    )
    
    # Build result and plan lookup tables
    result_by_step = {item.get("step_id"): item for item in step_records}
    plan_by_id = {
        _safe_step_id(step.get("step_id"), index + 1): step
        for index, step in enumerate(execution_plan or [])
        if isinstance(step, dict)
    }
    
    # Find first step without result or with unmet dependencies
    first_invalid_step = None
    step_verifications = []
    
    for record in step_records:
        step_id = record.get("step_id")
        has_result = record.get("has_result", False)
        depends_on = record.get("depends_on", [])
        
        # Determine this step's status
        status = "valid"
        error_type = None
        error_reason = None
        
        if not has_result:
            status = "invalid"
            error_type = "empty_result"
            error_reason = f"No results retrieved for query: '{record.get('sub_query', '')}'"
        elif depends_on:
            # Check if all dependencies have results
            unmet_deps = [
                dep_id for dep_id in depends_on 
                if not result_by_step.get(dep_id, {}).get("has_result")
            ]
            if unmet_deps:
                status = "invalid"
                error_type = "dependency_error"
                error_reason = f"Dependency steps {unmet_deps} have no results"
        
        step_verifications.append({
            "step_id": step_id,
            "status": status,
            "confidence": "high" if status == "invalid" else "medium",
            "error_type": error_type,
            "error_reason": error_reason,
            "repair_hint": "Rerun this step with hybrid retrieval." if status == "invalid" else None,
        })
        
        if status == "invalid" and first_invalid_step is None:
            first_invalid_step = step_id
    
    # Fallback if no invalid step found explicitly but answer_verification says unreliable
    if first_invalid_step is None:
        first_invalid_step = step_records[-1].get("step_id") if step_records else 1
    
    last_valid_step_id = max(first_invalid_step - 1, 0)
    
    # Build backtracking plan based on failure type
    backtracking_plan = _build_backtracking_plan_from_diagnosis(
        question=question,
        execution_plan=execution_plan,
        step_records=step_records,
        step_verifications=step_verifications,
        answer_verification=answer_verification,
        reasoning_type=reasoning_type,
    )

    return {
        "verification_summary": {
            "overall_valid": False,
            "decision": "backtrack",
            "confidence": answer_verification.get("confidence", "medium"),
            "reason": answer_verification.get("reason", "Current evidence failed reliability verification."),
            "first_invalid_step": first_invalid_step,
            "last_valid_step_id": last_valid_step_id,
            "trusted_prefix_length": last_valid_step_id,
        },
        "answer_verification": answer_verification,
        "step_verifications": step_verifications,
        "backtracking_plan": backtracking_plan,
    }


async def verify_execution_plan_fine_grained(
        question: str,
        execution_plan: list,
        step_results: list,
        reasoning_type: str,
        global_config: dict,
        max_context_chars: int = 8000
) -> dict:
    """
    Verification gate for staged backtracking.

    Flow:
    1. Verify whether current retrieved text / inferred answers are reliable.
    2. If reliable, accept immediately and let the final generation agent respond.
    3. If unreliable, diagnose the failed step and return a backtracking plan.
    """
    reliability = await verify_current_evidence_reliability(
        question=question,
        execution_plan=execution_plan,
        step_results=step_results,
        reasoning_type=reasoning_type,
        global_config=global_config,
        max_context_chars=max_context_chars,
    )

    if reliability.get("verification_summary", {}).get("overall_valid"):
        return reliability

    answer_verification = reliability.get("answer_verification", {})
    diagnosis = await diagnose_unreliable_steps(
        question=question,
        execution_plan=execution_plan,
        step_results=step_results,
        reasoning_type=reasoning_type,
        global_config=global_config,
        answer_verification=answer_verification,
        max_context_chars=max_context_chars,
    )

    return diagnosis


async def verify_execution_plan(
        question: str,
        execution_plan: list,
        step_results: dict,  # { "?r1": result_dict, "?r2": ... }
        reasoning_type: str,
        global_config: dict,
        type_list: list = None,
        relation_list: list = None
) -> dict:
    template = PROMPTS.get("verification_backtracking")
    if not template:
        logger.error("Verification prompt not found")
        return {}

    # 准备 step_results 的字符串表示（让 LLM 容易阅读）
    results_str = json.dumps(step_results, ensure_ascii=False, indent=2)

    full_prompt = template.format(
        question=question,
        reasoning_type=reasoning_type,
        execution_plan=json.dumps(execution_plan, ensure_ascii=False, indent=2),
        step_results=results_str,
        completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    )

    use_model_func = global_config["llm_model_func"]

    try:
        result = await use_model_func(full_prompt, _priority=6)  # 验证优先级较高
        match = re.search(r"\{.*\}", result, re.DOTALL)

        if match:
            verification_data = json.loads(match.group(0))
            return verification_data
        else:
            logger.error("No JSON in verification output")
            return {}

    except Exception as e:
        logger.error(f"Verification agent failed: {e}")
        return {}

def force_fix_json(raw_json):
    # 移除可能存在的注释
    content = re.sub(r'//.*$', '', raw_json, flags=re.MULTILINE)

    # 核心修复逻辑：
    # 匹配 { 后面直接跟着一个字符串，且后面是逗号的情况（即漏掉了键名）
    # 它会将 { "unknown_1", 替换为 { "name": "unknown_1",
    # 它会将 { "The Office", 替换为 { "name": "The Office",
    fixed = re.sub(r'\{\s*"([^"]+)"\s*,', r'{"name": "\1",', content)

    # 针对你给出的错误样本，这个正则会精准命中并在引号前插入 "name":
    return fixed
async def extract_question_semantics_only_musique(
        relation_list: list[str],
        type_list: list[str],
        question: str,
        param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None
) -> dict:
    # 1. 获取配置与分词器
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    tokenizer = global_config["tokenizer"]
    use_model_func = global_config["llm_model_func"]

    # --- 第一步：实体提取 (Entity Extraction) ---
    entity_template = PROMPTS.get("question_entity_extraction")
    entity_prompt = entity_template.format(
        question=question,
        language=language,
        type_list=type_list,#entity_vdb中的所有实体类型名汇总
        completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    )

    logger.debug(f"[Entity Step] Tokens: {len(tokenizer.encode(entity_prompt))}")

    try:
        # 调用 LLM 提取实体
        entity_result_raw = await use_model_func(entity_prompt, _priority=5)
        # 简单清洗实体结果（假设返回的是逗号分隔或JSON，这里示例按JSON处理）
        # entities = self._parse_json_from_text(entity_result_raw)
        entity_match = re.search(r"\{.*\}", entity_result_raw, re.DOTALL)
        if entity_match:
            raw_json = entity_match.group(0)
            clean_json = force_fix_json(raw_json)
            entities_dict = json.loads(clean_json)
        else:
            entities_dict = {"entities": []}
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {}

    triple_template = PROMPTS.get("triple_generation_with_entities")
    triple_prompt = triple_template.format(
        question=question,
        type_list=type_list,
        entities=json.dumps(entities_dict, ensure_ascii=False),  # 结合实体
        language=language,
        relation_list=relation_list,#relation_vdb中所有的关系名汇总
        completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    )

    logger.debug(f"[Triple Step] Tokens: {len(tokenizer.encode(triple_prompt))}")

    try:
        # 第二次调用 LLM
        final_result_raw = await use_model_func(triple_prompt, _priority=5)

        # 解析最终的 JSON 结果
        match = re.search(r"\{.*\}", final_result_raw, re.DOTALL)
        if not match:
            logger.error("❌ No JSON-like section in Triple Generation output")
            return {}

        data = json.loads(match.group(0))
        return data

    except Exception as e:
        logger.error(f"Triple generation failed: {e}")
        return {}


async def decompose_multi_hop_query(
    type_list: list[str],
    relation_list: list[str],
    question: str,
    param: QueryParam,
    global_config: dict[str, str],
    max_steps: int = 6,
    hashing_kv: BaseKVStorage | None = None
) -> dict:
    language = global_config.get("addon_params", {}).get(
        "language", PROMPTS.get("DEFAULT_LANGUAGE", "English")
    )
    #
    # # ====================== 1. 先提取实体 ======================
    # entities = []
    # try:
    #     template_entity = PROMPTS.get("entity_extraction")
    #     if template_entity:
    #         entity_prompt = template_entity.format(
    #             question=question,
    #             language=language,
    #             completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    #         )
    #         use_model_func = global_config["llm_model_func"]
    #         entity_result = await use_model_func(entity_prompt, _priority=5)
    #
    #         match = re.search(r"\{.*\}", entity_result, re.DOTALL)
    #         if match:
    #             entities_data = json.loads(match.group(0))
    #             entities = entities_data.get("entities", [])
    # except Exception as e:
    #     logger.warning(f"Entity extraction failed: {e}")
    #
    # logger.debug(f"[decomposition] Extracted entities: {entities}")

    # ====================== 2. 类型判断 (Router) - 从 md 文件加载 ======================
    try:
        router_template = PROMPTS.get("query_type_classifier")
        if not router_template:
            logger.error("query_type_classifier prompt not found!")
            reasoning_type = "multi_hop"
        else:
            router_prompt = router_template.format(
                question=question,
                completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
            )

            use_model_func = global_config["llm_model_func"]
            router_result = await use_model_func(router_prompt, _priority=5)

            match = re.search(r"\{.*\}", router_result, re.DOTALL)
            if match:
                router_data = json.loads(match.group(0))
                reasoning_type = router_data.get("reasoning_type", "multi_hop").lower().strip()
            else:
                reasoning_type = "multi_hop"
    except Exception as e:
        logger.warning(f"Query type classification failed: {e}, defaulting to multi_hop")
        reasoning_type = "multi_hop"

    logger.debug(f"[decomposition] Detected reasoning type: {reasoning_type}")

    # ====================== 3. 根据类型选择提示词 ======================
    if reasoning_type == "comparison" or "compare" in reasoning_type or any(
        word in question.lower() for word in ["older", "younger", "larger", "smaller", "more", "less", "first", "last"]
    ):
        template_key = "multi_hop_query_decomposition_comparative"
        logger.info("🔀 Using Comparative Decomposition Prompt")
    else:
        template_key = "multi_hop_query_decomposition_triple"
        logger.info("🔀 Using Triple Execution Plan Decomposition Prompt")

    template = PROMPTS.get(template_key)
    if not template:
        logger.error(f"Prompt template {template_key} not found!")
        return {}

    full_prompt = template.format(
        type_list=type_list,
        language=language,
        max_steps=max_steps,
        question=question,
        completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    )

    # ====================== 4. 执行分解 ======================
    tokenizer = global_config["tokenizer"]
    logger.debug(f"[decomposition] Tokens: {len(tokenizer.encode(full_prompt))}")

    try:
        result = await use_model_func(full_prompt, _priority=5)
    except Exception as e:
        logger.error(f"LLM decomposition failed: {e}")
        return {}

    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("❌ No JSON-like section in LLM output")
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.error("❌ JSON parsing error")
        return {}


# ====================== 5. 合并校验优化 ======================
#     execution_plan = data.get("execution_plan", [])
#     if execution_plan and len(execution_plan) > 2:
#         try:
#             optimizer_template = PROMPTS.get("execution_plan_optimizer")
#             if not optimizer_template:
#                 logger.warning("execution_plan_optimizer prompt not found, skipping merge pass")
#             else:
#                 optimizer_prompt = optimizer_template.format(
#                     question=question,
#                     execution_plan=json.dumps(execution_plan, ensure_ascii=False, indent=2),
#                     completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
#                 )
#
#                 optimizer_result = await use_model_func(optimizer_prompt, _priority=5)
#
#                 opt_match = re.search(r"\{.*\}", optimizer_result, re.DOTALL)
#                 if opt_match:
#                     opt_data = json.loads(opt_match.group(0))
#                     if opt_data.get("optimized", False):
#                         changes = opt_data.get("changes", [])
#                         logger.info(
#                             f"[optimizer] Plan merged: {len(execution_plan)} → "
#                             f"{len(opt_data['execution_plan'])} steps | "
#                             f"changes: {changes}"
#                         )
#                         data["execution_plan"] = opt_data["execution_plan"]
#                         data["optimizer_changes"] = changes
#                     else:
#                         logger.debug("[optimizer] Plan unchanged after merge pass")
#                 else:
#                     logger.warning("[optimizer] No valid JSON from optimizer, keeping original plan")
#
#         except json.JSONDecodeError as e:
#             logger.warning(f"[optimizer] JSON parse error: {e}, keeping original plan")
#         except Exception as e:
#             logger.warning(f"[optimizer] Optimization failed: {e}, keeping original plan")

    if isinstance(data, dict):
        data["reasoning_type"] = reasoning_type

    return data


async def extract_question_semantics_only(
        type_list: list[str],
        question: str,
        param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None

) -> dict:

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    # type_list = param.type_list if hasattr(param, "type_list") else []

    template = PROMPTS.get("question_triple_extraction_with_unknowns")
    full_prompt = template.format(
        question=question,
        language=language,
        type_list=type_list,
        completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
    )

    tokenizer = global_config["tokenizer"]
    logger.debug(f"[question_semantics] Tokens: {len(tokenizer.encode(full_prompt))}")
    print(param.model_func)

    # 3️⃣ Select model executor
    use_model_func = global_config["llm_model_func"]
    # Apply higher priority (5) to query relation LLM function
    use_model_func = partial(use_model_func)  # must expose acomplete()

    try:

        result = await use_model_func(full_prompt, _priority=5)

    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return {}

    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("❌ No JSON-like section in LLM output")
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.error("❌ JSON parsing error")
        return {}


    return data


async def get_keywords_from_query(
        query: str,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
        text: str,
        param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=param.mode,
                    cache_type="keywords",
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        tokenizer: Tokenizer,
) -> tuple[list, list, list] | None:
    """
    Retrieve vector context from the vector database.

    This function performs vector search to find relevant text chunks for a query,
    formats them with file path and creation time information.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including top_k and ids
        tokenizer: Tokenizer for counting tokens

    Returns:
        Tuple (empty_entities, empty_relations, text_units) for combine_contexts,
        compatible with _get_edge_data and _get_node_data format
    """
    try:
        results = await chunks_vdb.query(
            query, top_k=query_param.top_k, ids=query_param.ids
        )
        if not results:
            return [], [], []

        valid_chunks = []
        for result in results:
            if "content" in result:
                # Directly use content from chunks_vdb.query result
                chunk_with_time = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                }
                valid_chunks.append(chunk_with_time)

        if not valid_chunks:
            return [], [], []

        maybe_trun_chunks = truncate_list_by_token_size(
            valid_chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.max_token_for_text_unit,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"Truncate chunks from {len(valid_chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
        )
        logger.info(
            f"Vector query: {len(maybe_trun_chunks)} chunks, top_k: {query_param.top_k}"
        )

        if not maybe_trun_chunks:
            return [], [], []

        # Create empty entities and relations contexts
        entities_context = []
        relations_context = []

        # Create text_units_context directly as a list of dictionaries
        text_units_context = []
        for i, chunk in enumerate(maybe_trun_chunks):
            text_units_context.append(
                {
                    "id": i + 1,
                    "content": chunk["content"],
                    "file_path": chunk["file_path"],
                }
            )

        return entities_context, relations_context, text_units_context
    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return [], [], []

async def vector_text_retrieval(
        sub_query: str,
        entities_vdb: BaseVectorStorage,
        top_k: int = 3,
        query_param: QueryParam | None = None,
) -> str:
    ids = getattr(query_param, "ids", None) if query_param else None
    top_k = top_k or getattr(query_param, "top_k", 3) or 3

    try:
        logger.info(
            f"Vector entity query: {sub_query}, top_k: {top_k}, "
            f"cosine: {entities_vdb.cosine_better_than_threshold}"
        )
        entity_results = await entities_vdb.query(sub_query, top_k=top_k, ids=ids)
    except Exception as e:
        logger.error(f"Entity similarity retrieval failed: {e}")
        return ""

    if not entity_results:
        logger.warning(f"No similar entities found for vector step: {sub_query}")
        return ""

    sentences = []
    for i, entity in enumerate(entity_results):
        content = extract_text_safe(entity.get("content", ""))
        if not content:
            continue
        sentences.append(
            {
                "id": i + 1,
                "entity": entity.get("entity_name") or entity.get("entity") or entity.get("id", "UNKNOWN"),
                "content": content,
                "similarity": entity.get("distance", entity.get("score", "UNKNOWN")),
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    if not sentences:
        logger.warning(f"No valid content in retrieved entities for: {sub_query}")
        return ""

    logger.info(f"Entity-vector retrieval returns {len(sentences)} sentences for: {sub_query}")

    return f"""-----Sentences-----

```json
{json.dumps(sentences, ensure_ascii=False, indent=2)}
```

"""

async def vector_chunk_retrieval(
        sub_query: str,
        chunks_vdb: BaseVectorStorage | None,
        top_k: int = 3,
        query_param: QueryParam | None = None,
) -> str:
    if chunks_vdb is None:
        return ""

    ids = getattr(query_param, "ids", None) if query_param else None
    top_k = top_k or getattr(query_param, "top_k", 3) or 3

    try:
        logger.info(
            f"Vector chunk query: {sub_query}, top_k: {top_k}, "
            f"cosine: {chunks_vdb.cosine_better_than_threshold}"
        )
        chunk_results = await chunks_vdb.query(sub_query, top_k=top_k, ids=ids)
    except Exception as e:
        logger.error(f"Chunk similarity retrieval failed: {e}")
        return ""

    if not chunk_results:
        logger.warning(f"No similar chunks found for vector step: {sub_query}")
        return ""

    chunks = []
    for i, chunk in enumerate(chunk_results):
        raw_content = extract_text_safe(chunk.get("content", ""))
        if not raw_content:
            continue

        keyword_part, separator, sentence_part = raw_content.partition("\n")
        keywords = extract_text_safe(keyword_part)
        sentence = extract_text_safe(sentence_part if separator else raw_content)

        if not sentence:
            continue

        chunks.append(
            {
                "id": i + 1,
                "keywords": keywords,
                "content": sentence,
                "similarity": chunk.get("distance", chunk.get("score", "UNKNOWN")),
                "file_path": chunk.get("file_path", "unknown_source"),
            }
        )

    if not chunks:
        logger.warning(f"No valid content in retrieved chunks for: {sub_query}")
        return ""

    logger.info(f"Chunk-vector retrieval returns {len(chunks)} chunks for: {sub_query}")

    return f"""-----Chunks-----

```json
{json.dumps(chunks, ensure_ascii=False, indent=2)}
```

"""

async def _build_query_context(
        ll_keywords: str,
        hl_keywords: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        chunks_vdb: BaseVectorStorage = None,  # Add chunks_vdb parameter for mix mode
):
    logger.info(f"Process {os.getpid()} building query context...")

    # Handle local and global modes as before
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid or mix mode
        ll_data = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        hl_data = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        # Initialize vector data with empty lists
        vector_entities_context, vector_relations_context, vector_text_units_context = (
            [],
            [],
            [],
        )

        # Only get vector data if in mix mode
        if query_param.mode == "mix" and hasattr(query_param, "original_query"):
            # Get tokenizer from text_chunks_db
            tokenizer = text_chunks_db.global_config.get("tokenizer")

            # Get vector context in triple format
            vector_data = await _get_vector_context(
                query_param.original_query,  # We need to pass the original query
                chunks_vdb,
                query_param,
                tokenizer,
            )

            # If vector_data is not None, unpack it
            if vector_data is not None:
                (
                    vector_entities_context,
                    vector_relations_context,
                    vector_text_units_context,
                ) = vector_data

        # Combine and deduplicate the entities, relationships, and sources
        entities_context = process_combine_contexts(
            hl_entities_context, ll_entities_context, vector_entities_context
        )
        relations_context = process_combine_contexts(
            hl_relations_context, ll_relations_context, vector_relations_context
        )
        text_units_context = process_combine_contexts(
            hl_text_units_context, ll_text_units_context, vector_text_units_context
        )
    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context:
        return None

    # 转换为 JSON 字符串
    entities_str = json.dumps(entities_context, ensure_ascii=False)
    relations_str = json.dumps(relations_context, ensure_ascii=False)
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)

    result = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

"""
    return result


async def _get_node_data(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )

    results = await entities_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", ""

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas,
        query_param,
        text_chunks_db,
        knowledge_graph_inst,
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
        tokenizer=tokenizer,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # build prompt
    entities_context = []
    for i, n in enumerate(node_datas):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from node data
        file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["entity_name"],
                "type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", "UNKNOWN"),
                "rank": n["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    relations_context = []
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from edge data
        file_path = e.get("file_path", "unknown_source")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": e["src_tgt"][0],
                "entity2": e["src_tgt"][1],
                "description": e["description"],
                "keywords": e["keywords"],
                "weight": e["weight"],
                "rank": e["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    text_units_context = []
    for i, t in enumerate(use_text_units):
        text_units_context.append(
            {
                "id": i + 1,
                "content": t["content"],
                "file_path": t.get("file_path", "unknown_source"),
            }
        )
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage,
        knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
        if dp["source_id"] is not None
    ]

    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
    # Build the edges list in the same order as node_datas.
    edges = [batch_edges_dict.get(name, []) for name in node_names]

    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)

    # Batch retrieve one-hop node data using get_nodes_batch
    all_one_hop_nodes_data_dict = await knowledge_graph_inst.get_nodes_batch(
        all_one_hop_nodes
    )
    all_one_hop_nodes_data = [
        all_one_hop_nodes_data_dict.get(e) for e in all_one_hop_nodes
    ]

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []

    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = index
                tasks.append((c_id, index, this_edges))

    # Process in batches tasks at a time to avoid overwhelming resources
    batch_size = 5
    results = []

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i: i + batch_size]
        batch_results = await asyncio.gather(
            *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in batch_tasks]
        )
        results.extend(batch_results)

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 0.0"
                )
                edge_props["weight"] = 0.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    tokenizer: Tokenizer = knowledge_graph_inst.global_config.get("tokenizer")
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return all_edges_data
async def get_unique_entity_types(entities_vdb) -> set:
    """
    从 relationships_vdb 中提取所有关系类型，并返回一个去重后的集合
    """
    # storage = await entity_vdb.client_storage
    # all_data = storage.get("data", [])
    all_entity_raw = await entities_vdb.client_storage
    all_data = all_entity_raw.get("data", []) if isinstance(all_entity_raw, dict) else []

    entity_types = set()

    for item in all_data:
        entity_type = item.get("entity_type", "")
        if entity_type:
            entity_types.add(entity_type)

    return entity_types

async def get_unique_relationship_types(relationships_vdb) -> set:
    """
    从 relationships_vdb 中提取所有关系类型，并返回一个去重后的集合
    """
    storage = await relationships_vdb.client_storage
    all_data = storage.get("data", [])

    rel_types = set()

    for item in all_data:
        content = item.get("content", "")
        if not content:
            continue

        lines = content.split("\n")
        if len(lines) < 2:
            continue

        # 第二行是关系类型
        rel_type = lines[1].strip()
        if rel_type:
            rel_types.add(rel_type)

    return rel_types


async def _get_edge_data(
        keywords,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )
    ##提取关系vdb中的所有关系类型
    # storage = await relationships_vdb.client_storage
    # all_data = storage["data"]
    # rel_types = set()
    # for item in all_data:
    #     content = item.get("content", "")
    #     lines = content.split("\n")
    #     if len(lines) < 2:
    #         continue
    #
    #     # 第二行是关系
    #     rel_type = lines[1].strip()
    #     rel_types.add(rel_type)
    #     #print(rel_type)
    # type_list = await get_unique_relationship_types(relationships_vdb)
    # print(type_list)

    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )

    if not len(results):
        return "", "", ""

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    # For edge degrees, use tuples.
    edge_pairs_tuples = [(r["src_id"], r["tgt_id"]) for r in results]

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 0.0"
                )
                edge_props["weight"] = 0.0

            # Use edge degree from the batch as rank.
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "rank": edge_degrees_dict.get(pair, k.get("rank", 0)),
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_global_context,
        tokenizer=tokenizer,
    )
    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas,
            query_param,
            knowledge_graph_inst,
        ),
        _find_related_text_unit_from_relationships(
            edge_datas,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    relations_context = []
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from edge data
        file_path = e.get("file_path", "unknown_source")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": e["src_id"],
                "entity2": e["tgt_id"],
                "description": e["description"],
                "keywords": e["keywords"],
                "weight": e["weight"],
                "rank": e["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    entities_context = []
    for i, n in enumerate(use_entities):
        created_at = n.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from node data
        file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["entity_name"],
                "type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", "UNKNOWN"),
                "rank": n["rank"],
                "created_at": created_at,
                "file_path": file_path,
            }
        )

    text_units_context = []
    for i, t in enumerate(use_text_units):
        text_units_context.append(
            {
                "id": i + 1,
                "content": t["content"],
                "file_path": t.get("file_path", "unknown"),
            }
        )
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Batch approach: Retrieve nodes and their degrees concurrently with one query each.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(entity_names),
        knowledge_graph_inst.node_degrees_batch(entity_names),
    )

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        degree = degrees_dict.get(entity_name, 0)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name and computed degree (as rank)
        combined = {**node, "entity_name": entity_name, "rank": degree}
        node_datas.append(combined)

    tokenizer: Tokenizer = knowledge_graph_inst.global_config.get("tokenizer")
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"] if x["description"] is not None else "",
        max_token_size=query_param.max_token_for_local_context,
        tokenizer=tokenizer,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage,
        knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
        if dp["source_id"] is not None
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    tokenizer: Tokenizer = text_chunks_db.global_config.get("tokenizer")
    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
        tokenizer=tokenizer,
    )

    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


async def naive_query(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Handle cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    tokenizer: Tokenizer = global_config["tokenizer"]

    _, _, text_units_context = await _get_vector_context(
        query, chunks_vdb, query_param, tokenizer
    )

    if text_units_context is None or len(text_units_context) == 0:
        return PROMPTS["fail_response"]

    text_units_str = json.dumps(text_units_context, ensure_ascii=False)
    if query_param.only_need_context:
        return f"""
---Document Chunks---

```json
{text_units_str}
```

"""
    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=text_units_str,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[naive_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    return response


# TODO: Deprecated, use user_prompt in QueryParam instead
async def kg_query_with_keywords(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        ll_keywords: list[str] = [],
        hl_keywords: list[str] = [],
        chunks_vdb: BaseVectorStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb=chunks_vdb,
    )
    if not context:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return context

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # Clean up response content
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        if hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=query_param.mode,
                    cache_type="query",
                ),
            )

    return response


# TODO: Deprecated, use user_prompt in QueryParam instead
async def query_with_keywords(
        query: str,
        prompt: str,
        param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Extract keywords from the query and then use them for retrieving information.

    1. Extracts high-level and low-level keywords from the query
    2. Formats the query with the extracted keywords and prompt
    3. Uses the appropriate query method based on param.mode

    Args:
        query: The user's query
        prompt: Additional prompt to prepend to the query
        param: Query parameters
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entities vector database
        relationships_vdb: Relationships vector database
        chunks_vdb: Document chunks vector database
        text_chunks_db: Text chunks storage
        global_config: Global configuration
        hashing_kv: Cache storage

    Returns:
        Query response or async iterator
    """
    # Extract keywords

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query=query,
        query_param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    # Create a new string with the prompt and the keywords
    keywords_str = ", ".join(ll_keywords + hl_keywords)
    formatted_question = (
        f"{prompt}\n\n### Keywords\n\n{keywords_str}\n\n### Query\n\n{query}"
    )

    param.original_query = query

    # Use appropriate query method based on mode
    if param.mode in ["local", "global", "hybrid", "mix"]:
        return await kg_query_with_keywords(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            chunks_vdb=chunks_vdb,
        )
    elif param.mode == "naive":
        return await naive_query(
            formatted_question,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")

async def decide_retrieval_strategies(
    execution_plan: list[dict],
    relation_list: list[str],
    global_config: dict[str, str],
    type_list: list[str],
    model_func: callable,
    experience_context: str = "",
) -> list[dict]:

    language = global_config.get("addon_params", {}).get(
        "language", PROMPTS.get("DEFAULT_LANGUAGE", "English")
    )

    def fallback_decisions(reason: str) -> list[dict]:
        return [
            {
                "step_id": step.get("step_id"),
                "retrieval_method": "graph",
                "confidence": "low",
                "reason": reason,
            }
            for step in execution_plan or []
            if isinstance(step, dict)
        ]

    template = PROMPTS.get("Retrieval Strategy Router")
    if not template:
        return fallback_decisions(
            "Retrieval strategy router prompt not found; fallback to graph retrieval."
        )

    try:
        formatted_prompt = template.format(
            execution_plan=json.dumps(execution_plan, ensure_ascii=False, indent=2),
            experience_context=experience_context.strip() if experience_context else "No relevant experience memory retrieved.",
            language=language,
            completion_delimiter=PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "")
        )
        use_model_func = model_func or global_config["llm_model_func"]
        response = await use_model_func(formatted_prompt, _priority=5)
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # 再尝试匹配对象 {...}
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response.strip())

        return result


    except json.JSONDecodeError as e:

        logger.error(f"JSON解析失败, response was: {locals().get('response', '')}")

        return fallback_decisions(f"JSON解析失败，降级为graph检索: {e}")
    except Exception as e:
        logger.error(f"Retrieval strategy routing failed: {e}")
        return fallback_decisions(f"Router failed，降级为graph检索: {e}")


from collections import Counter, deque
from typing import Optional


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _is_unknown_type(val: str) -> bool:
    """判断一个字符串是否是 unknownN(Type) 格式"""
    if not isinstance(val, str):
        return False
    import re
    return bool(re.match(r'^unknown\d+\(', val))


def build_shared_state(execution_plan: list[dict]) -> tuple[dict, dict]:
    """
    自动扫描 execution_plan，推导哪些 unknown 存在 AND 约束。

    判断规则：
      - 某个 unknownN 在多个 step 中作为 result_var 出现，说明多个 triple 都在
        解析它，即它需要同时满足这些 triple 的约束。
      - 若 unknownN 只是作为某 step 的输入 anchor（不是该 step 的 result_var），
        不计入约束计数。

    返回：
      shared_candidates: dict  unknown_key -> set，运行时动态填充，初始为空
      shared_constraints: dict  unknown_key -> int，该 unknown 还有几个约束 triple 待处理
    """
    constraint_counter = Counter()
    for step in execution_plan:
        t = step.get("triple", {})
        result_var = step.get("result_var", "")
        for val in [t.get("head"), t.get("tail")]:
            if val and _is_unknown_type(val) and val == result_var:
                constraint_counter[val] += 1

    shared_constraints = {k: v for k, v in constraint_counter.items() if v > 1}
    shared_candidates: dict = {}
    return shared_candidates, shared_constraints


# ─────────────────────────────────────────────
# 核心检索函数
# ─────────────────────────────────────────────

async def kg_triple_retrieval_for_step(
        step: dict,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        query_param: QueryParam,
        shared_candidates: Optional[dict] = None,
        shared_constraints: Optional[dict] = None,
) -> tuple[deque, deque, list, str, dict]:
    """
    对单个 execution plan step 执行 KG 检索。

    原有的链式检索、实体链接、关系验证逻辑完全不变。
    新增逻辑：
      - 若 step 的 result_var 在 shared_constraints 中，说明该 unknown 有 AND 约束，
        走交集收窄路径，不立即写入 unknown_map。
      - 倒计时归零后才将交集结果冻结进 unknown_map，供后续 step 消费。

    参数：
      shared_candidates: 跨 step 共享的候选集，AND 约束 unknown 专用
      shared_constraints: 跨 step 共享的倒计时，AND 约束 unknown 专用

    返回：
      q1:            未能完成实体链接的剩余 triple
      q2:            关系验证队列（本轮最终状态）
      q3:            已验证通过的 item 列表
      final_sentences: 实体级别的文本片段
      unknown_map:   本 step 产出的 unknown -> entity_id 绑定（已冻结的）
    """
    # 降级兼容：未传入共享状态时，退化为原始行为
    if shared_candidates is None:
        shared_candidates = {}
    if shared_constraints is None:
        shared_constraints = {}

    triple = step.get("triple", {})
    result_var = step.get("result_var", "")
    initial_triples = [{
        "subject": triple.get("head"),
        "predicate": triple.get("rel"),
        "object": triple.get("tail")
    }]
    query = step.get("sub_query", "")

    q1 = deque(initial_triples)
    q3 = []
    q2 = deque()
    unknown_map = {}
    final_sentences = ""
    iteration = 0

    while q1 and iteration < 5:
        iteration += 1
        print(f"\n=== 第 {iteration} 轮迭代 ===")

        # ── 实体链接阶段（原逻辑不变）──
        remaining_q1 = deque()
        while q1:
            raw_triple = q1.popleft()
            candidate_triples = await _expand_triples_with_map(raw_triple, unknown_map)
            for current_triple in candidate_triples:
                temp_analysis = {'triples': [current_triple]}
                linked_results, entity_level_sentence = await link_entities_from_kg(
                    temp_analysis, entities_vdb, query_param
                )
                if linked_results:
                    q2.extend(linked_results)
                else:
                    print(f"{current_triple} 实体链接失败")
                    remaining_q1.append(current_triple)
                if entity_level_sentence:
                    final_sentences += f" {entity_level_sentence}"
        q1 = remaining_q1

        # ── 关系验证阶段（原逻辑不变）──
        verified_queue = deque()
        failed_tasks = await verify_relationships_with_fallback(
            query=query,
            matched_queue=q2,
            verified_queue=verified_queue,
            relationships_vdb=relationships_vdb,
            global_config=global_config
        )

        # ── 收集本轮所有 resolved_id，再统一处理绑定 ──
        # 用临时集合先聚合，避免同一 unknown 在一轮内多次命中时过早做交集
        this_round_resolved: dict[str, set] = {}

        while verified_queue:
            item = verified_queue.popleft()
            real_t = item['triple']
            orig_t = item['original_triple']
            seed = item['seed_entity']
            anchor_node = item.get('via') or seed
            q3.append(item)

            unknown_key = None
            resolved_id = None
            if _is_unknown_type(orig_t.get('object')):
                unknown_key = orig_t['object']
                resolved_id = (
                    real_t['tgt_id'] if real_t['src_id'] == anchor_node
                    else real_t['src_id']
                )
            elif _is_unknown_type(orig_t.get('subject')):
                unknown_key = orig_t['subject']
                resolved_id = (
                    real_t['src_id'] if real_t['tgt_id'] == anchor_node
                    else real_t['tgt_id']
                )

            if unknown_key and resolved_id and resolved_id != anchor_node:
                this_round_resolved.setdefault(unknown_key, set()).add(resolved_id)

        # ── 绑定决策：AND 约束路径 vs 原始路径 ──
        for unknown_key, resolved_ids in this_round_resolved.items():
            if unknown_key in shared_constraints:
                # AND 约束路径：交集收窄，不立即冻结
                if unknown_key not in shared_candidates:
                    # 第一次命中，直接赋值
                    shared_candidates[unknown_key] = set(resolved_ids)
                    print(f"  [候选初始化] {unknown_key} -> {shared_candidates[unknown_key]}")
                else:
                    # 后续命中，与已有候选集取交集
                    before = shared_candidates[unknown_key]
                    shared_candidates[unknown_key] &= resolved_ids
                    print(f"  [候选收窄] {unknown_key}: {before} ∩ {resolved_ids} = {shared_candidates[unknown_key]}")
            else:
                # 原始路径：直接写入 unknown_map（行为与原代码完全一致）
                if unknown_key not in unknown_map:
                    unknown_map[unknown_key] = set()
                unknown_map[unknown_key] |= resolved_ids
                print(f"  [发现映射] {unknown_key} -> {resolved_ids} (锚点: {anchor_node})")

    # ── AND 约束收尾：倒计时减一，归零则冻结 ──
    if result_var in shared_constraints:
        shared_constraints[result_var] -= 1
        print(f"  [倒计时] {result_var} 剩余约束数: {shared_constraints[result_var]}")

        if shared_constraints[result_var] == 0:
            candidates = shared_candidates.get(result_var, set())
            if candidates:
                unknown_map[result_var] = candidates
                print(f"  [冻结绑定] {result_var} 最终绑定为: {candidates}")
            else:
                print(f"  [警告] {result_var} AND 约束交集为空，无法绑定")

    unlinked_entity_sentences = await collect_unlinked_entity_contents(
        q3=q3,
        entities_vdb=entities_vdb
    )

    return q1, q2, q3, final_sentences, unknown_map


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

async def execute_plan(
        execution_plan: list[dict],
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        query_param: QueryParam,
) -> dict:
    """
    按顺序执行 execution_plan 中的每个 step。

    - 自动推导 AND 约束，无需手动配置。
    - shared_candidates / shared_constraints 在所有 step 间共享，
      保证交集收窄逻辑跨 step 生效。
    - 每个 step 产出的 unknown_map 合并进 accumulated_unknown_map，
      供后续 step 通过 _expand_triples_with_map 消费已冻结的绑定。

    返回：
      accumulated_unknown_map: 所有 step 最终产出的 unknown -> entity_id 绑定
    """
    # 自动推导 AND 约束，shared 状态在所有 step 间传递
    shared_candidates, shared_constraints = build_shared_state(execution_plan)

    if shared_constraints:
        print(f"[执行计划] 检测到 AND 约束 unknown: {shared_constraints}")
    else:
        print("[执行计划] 无 AND 约束，全部走原始链式路径")

    accumulated_unknown_map: dict = {}
    all_q3: list = []
    all_sentences: str = ""

    for step in execution_plan:
        step_id = step.get("step_id")
        print(f"\n{'=' * 40}")
        print(f"[Step {step_id}] {step.get('sub_query', '')}")
        print(f"{'=' * 40}")

        # 将已冻结的绑定注入当前 step 的 triple，实现跨 step 的链式传播
        # 注意：_expand_triples_with_map 在函数内部被调用，
        # 这里需要把 accumulated_unknown_map 预先合并进 step 的上下文。
        # 最简单的做法是把已知绑定注入 triple（替换已冻结的 unknown）：
        step = _inject_known_bindings(step, accumulated_unknown_map)

        _, _, q3, sentences, step_unknown_map = await kg_triple_retrieval_for_step(
            step=step,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            global_config=global_config,
            query_param=query_param,
            shared_candidates=shared_candidates,
            shared_constraints=shared_constraints,
        )

        accumulated_unknown_map.update(step_unknown_map)
        all_q3.extend(q3)
        all_sentences += sentences

    print(f"\n[执行完成] 最终绑定: {accumulated_unknown_map}")
    return accumulated_unknown_map


def _inject_known_bindings(step: dict, unknown_map: dict) -> dict:
    """
    将已冻结的 unknown 绑定替换进 step 的 triple，
    使后续 step 能以具体实体为 anchor 继续检索。
    只替换集合大小为 1 的绑定（已唯一确定的实体）。
    """
    import copy
    step = copy.deepcopy(step)
    triple = step.get("triple", {})
    for pos in ["head", "tail"]:
        val = triple.get(pos)
        if val and _is_unknown_type(val) and val in unknown_map:
            candidates = unknown_map[val]
            if isinstance(candidates, set) and len(candidates) == 1:
                triple[pos] = next(iter(candidates))
            elif isinstance(candidates, str):
                triple[pos] = candidates
    step["triple"] = triple
    return step

