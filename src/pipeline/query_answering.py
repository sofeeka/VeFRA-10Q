from loguru import logger
from qdrant_client.conversions.common_types import ScoredPoint

from src.data_models.retrieval import DocumentMetadata
from src.debug.debug import DebugData_Chunk, DebugData_Document, debug_manager
from src.generation.async_generator import AsyncGenerator
from src.pipeline.query_expansion import expand_query
from src.pipeline.relevant_docs_extractor import get_relevant_docs
from src.processing.document_processor import process_chunk_after_retrieval
from src.retrieval.database import UserKnowledgeBase
from src.utils.dependency import get_reranking_model
from src.utils.exceptions import VeFRA_GenerationError


class QueryAnsweringConfig:
    def __init__(
        self,
        document_extraction: bool = True,
        query_expansion: bool = True,
        reranking: bool = True,
    ):
        self.document_extraction = document_extraction
        self.query_expansion = query_expansion
        self.reranking = reranking


class QueryAnsweringPipeline:
    def __init__(
        self,
        db: UserKnowledgeBase,
        generator: AsyncGenerator,
        config: QueryAnsweringConfig = QueryAnsweringConfig(),
    ):
        self.db = db
        self.generator = generator
        self.config = config

    async def _extract_metadata(self, query: str) -> list[DocumentMetadata]:
        try:
            logger.info("Extracting relevant document metadata...", query=query)

            relevant_docs_metadata: list[DocumentMetadata] = await get_relevant_docs(
                question=query, user_id=self.db.user_id
            )

            if debug_manager.is_enabled():
                for doc in relevant_docs_metadata:
                    debug_manager.get_data().add_document(
                        quarter=doc.quarter, year=doc.year
                    )

            if not relevant_docs_metadata:
                logger.warning("No documents identified. Searching without filter.")
                return []
            else:
                serializable = [doc.model_dump() for doc in relevant_docs_metadata]
                logger.info(
                    f"{len(relevant_docs_metadata)} relevant document metadata extracted.",
                    metadata=serializable,
                )
            return relevant_docs_metadata
        except VeFRA_GenerationError as e:
            logger.warning(
                f"Document extraction failed with a generation error: {e.message}"
            )
            return []
        except Exception:
            logger.error(
                "An unexpected error occurred during document metadata extraction. Proceeding without filter."
            )
            return []

    async def _expand_query(self, query: str) -> list[str]:
        queries: list[str] = await expand_query(query=query)
        return queries

    async def _retrieve(
        self,
        queries: list[str],
        document_metadata: list[DocumentMetadata],
    ) -> list[ScoredPoint]:
        all_retrieved_points: list[ScoredPoint] = []

        for query in queries:
            retrieved_points = await self.db.get_related_points(
                query=query,
                doc_metadata_filter=document_metadata,
            )

            all_retrieved_points.extend(retrieved_points)

            if debug_manager.is_enabled():
                chunks_per_query: list[DebugData_Chunk] = []
                for point in retrieved_points:
                    metadata = point.payload.get(
                        "metadata", {"year": "N/A", "quarter": "N/A"}
                    )
                    year = metadata["year"]
                    quarter = metadata["quarter"]
                    chunk = DebugData_Chunk(
                        text=point.payload["text"],
                        document=DebugData_Document(year=year, quarter=quarter),
                    )
                    chunks_per_query.append(chunk)

                debug_manager.get_data().add_expanded_query(
                    query=query, chunks=chunks_per_query
                )

        logger.info(
            "Retrieved {chunk_count} chunks from database.",
            chunk_count=len(all_retrieved_points),
            query_count=len(queries),
        )

        if not all_retrieved_points:
            logger.warning(
                "No chunks found after retrieval, possibly due to document filtering.",
                relevant_docs_metadata=metadata,
            )
            return []

        unique_points_dict = {point.id: point for point in all_retrieved_points}
        unique_retrieved_points = list(unique_points_dict.values())

        return unique_retrieved_points

    async def _rerank(self, query: str, chunks: list[str]) -> list[str]:
        reranker = get_reranking_model()
        try:
            reranked_chunks = reranker.rerank(
                query=query,
                chunks=chunks,
                top_n=10,
            )
        except Exception:
            logger.error(
                "Re-ranking failed. Proceeding with original chunk order.",
                user_id=self.db.user_id,
                exc_info=True,
            )
            reranked_chunks = chunks[:10]
        return reranked_chunks

    async def _generate(self, chunks: list[str], query: str) -> str:
        if debug_manager.is_enabled():
            debug_manager.get_data().all_retrieved_chunks = chunks

        context = format_for_prompt(chunks)
        user_prompt: str = f"""
Context from 10-Q form:
---
{context}
---
Question: {query}
"""

        try:
            parsed_response = await self.generator.generate_response(prompt=user_prompt)
            response = parsed_response.output_parsed.response
            logger.success("Successfully generated and parsed response from LLM.")

            return response
        except Exception:
            logger.error(
                "Failed to generate or parse response from LLM.",
                user_id=self.db.user_id,
                exc_info=True,
            )
        raise

    async def run(self, query: str) -> str:
        logger.info(f"Running pipeline for query: {query}")

        metadata = []
        if self.config.document_extraction:
            metadata = await self._extract_metadata(query=query)

        queries = [query]
        if self.config.query_expansion:
            expanded = await self._expand_query(query=query)
            queries.extend(expanded)

        points = await self._retrieve(queries=queries, document_metadata=metadata)
        if not points:
            return (
                "I could not find any relevant information relevant to your question. "
                "Please try being more specific with the period you are interested in."
            )

        rebuilt_chunks = create_chunks(points=points, user_id=self.db.user_id)

        if self.config.reranking:
            reranked_chunks = await self._rerank(query=query, chunks=rebuilt_chunks)
        else:
            reranked_chunks = rebuilt_chunks

        response = await self._generate(chunks=reranked_chunks, query=query)
        return response


def create_chunks(
    points: list[ScoredPoint],
    user_id: str,
) -> list[str]:
    context_parts = []
    rebuilt_chunks = []

    for _, point in enumerate(points):
        payload = point.payload
        raw_chunk_text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        year = metadata.get("year", "N/A")
        quarter = metadata.get("quarter", "N/A")

        rebuilt_chunk = process_chunk_after_retrieval(
            chunk=raw_chunk_text,
            user_id=user_id,
        )

        if debug_manager.is_enabled():
            debug_manager.get_data().add_unique_retrieved_chunk(
                rebuilt_chunk, quarter, year
            )

        rebuilt_chunks.append(rebuilt_chunk)

        context_block = f"""
Document: {year} {quarter}
Content:
{rebuilt_chunk}
"""
        context_parts.append(context_block)

    logger.debug("Reconstructed context parts.", context_parts=context_parts)

    return rebuilt_chunks


def format_for_prompt(ordered_chunks: list[str]) -> str:
    """
    Helper to join chunks into a single string for the LLM prompt.
    Adds clear separators so the LLM knows where one chunk ends.
    """
    return "\n\n---\n\n".join(ordered_chunks)
