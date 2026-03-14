from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docqa_app.config import Settings
from docqa_app.memory import update_memory
from docqa_app.retrieval import build_chunks, doc_refs_from_question, is_overview_query, page_refs_from_question
from docqa_app.types import ChunkRecord, ConversationMemory, DocumentRecord, PageRecord, SearchHit


class RetrievalMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            work_dir=Path(self.tmpdir.name),
            history_file=Path(self.tmpdir.name) / "history.json",
            warn_after_turns=3,
            rollover_after_turns=4,
        )
        self.docs = [
            DocumentRecord(
                doc_id="paper-a",
                name="Paper A.pdf",
                kind="pdf",
                path=Path("/tmp/Paper A.pdf"),
                page_count=2,
                summary="Research paper about retrieval pipelines.",
                pages=[
                    PageRecord(
                        doc_id="paper-a",
                        doc_name="Paper A.pdf",
                        page_num=1,
                        text="Introduction section about document question answering.",
                        source="digital-text",
                        confidence=0.92,
                        summary="Paper A page 1 introduction.",
                    ),
                    PageRecord(
                        doc_id="paper-a",
                        doc_name="Paper A.pdf",
                        page_num=2,
                        text="Results section with latency discussion and evaluation.",
                        source="digital-text",
                        confidence=0.91,
                        summary="Paper A page 2 results.",
                    ),
                ],
            ),
            DocumentRecord(
                doc_id="form-b",
                name="Handwritten Form.png",
                kind="image",
                path=Path("/tmp/Handwritten Form.png"),
                page_count=1,
                summary="Handwritten intake form with name, phone number, and notes.",
                pages=[
                    PageRecord(
                        doc_id="form-b",
                        doc_name="Handwritten Form.png",
                        page_num=1,
                        text="Name | Phone\nAli | 0300-0000000",
                        source="ocr-primary",
                        confidence=0.71,
                        summary="Handwritten form page 1.",
                    )
                ],
            ),
        ]

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_page_refs_are_detected(self) -> None:
        self.assertEqual(page_refs_from_question("Compare pages 2 and 4"), [2, 4])
        self.assertEqual(page_refs_from_question("Look at pg 3"), [3])

    def test_doc_refs_support_ordinals_and_all(self) -> None:
        self.assertEqual(doc_refs_from_question("Summarize the first document", self.docs), ["paper-a"])
        self.assertEqual(
            doc_refs_from_question("Compare both files", self.docs),
            ["paper-a", "form-b"],
        )

    def test_overview_queries_are_detected(self) -> None:
        self.assertTrue(is_overview_query("What is this document about?"))
        self.assertFalse(is_overview_query("What is the phone number on page 1?"))

    def test_build_chunks_creates_summary_and_text_entries(self) -> None:
        chunks = build_chunks(self.docs, self.settings)
        kinds = {chunk.kind for chunk in chunks}
        self.assertIn("doc-summary", kinds)
        self.assertIn("page-summary", kinds)
        self.assertIn("page-text", kinds)

    def test_memory_rollover_preserves_summary_and_resets_visible_context(self) -> None:
        memory = ConversationMemory()
        hit = SearchHit(
            chunk=ChunkRecord(
                chunk_id="paper-a:page:2:text:1",
                doc_id="paper-a",
                doc_name="Paper A.pdf",
                page_num=2,
                kind="page-text",
                text="Results section with latency discussion and evaluation.",
                token_set={"results", "latency"},
            ),
            score=0.88,
        )

        reset = False
        for turn in range(4):
            memory, reset = update_memory(
                memory,
                question=f"Question {turn}",
                answer=f"Answer {turn}",
                hits=[hit],
                llm_complete=lambda prompt: "summary: active doc paper-a page 2",
                settings=self.settings,
            )

        self.assertTrue(reset)
        self.assertEqual(memory.user_turns, 0)
        self.assertEqual(memory.recent_turns, [])
        self.assertTrue(memory.rolling_summary)
        self.assertEqual(memory.active_doc_ids, ["paper-a"])
        self.assertEqual(memory.active_pages, [2])


if __name__ == "__main__":
    unittest.main()
