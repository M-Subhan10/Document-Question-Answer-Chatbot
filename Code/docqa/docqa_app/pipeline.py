from __future__ import annotations

from pathlib import Path

from .backends import select_backends
from .config import Settings
from .extraction import DocumentExtractor, make_doc_id
from .memory import format_recent_turns, update_memory
from .prompts import load_prompt
from .retrieval import HybridRetriever, build_chunks
from .state import SessionState, append_history
from .types import DocumentRecord, SearchHit


class DocumentPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_dirs()
        self.primary_backend, self.backup_backend = select_backends(settings)
        self.extractor = DocumentExtractor(settings, self.primary_backend, self.backup_backend)
        self.retriever = HybridRetriever(settings)
        self.state = SessionState()

    @property
    def backend_label(self) -> str:
        if self.backup_backend:
            return f"{self.primary_backend.name} + {self.backup_backend.name} fallback"
        return self.primary_backend.name

    def ingest(self, files) -> tuple[list[DocumentRecord], list[str]]:
        if not files:
            raise ValueError("No file selected.")
        paths = [Path(getattr(file, "name", file)) for file in files]
        if len(paths) > self.settings.max_files:
            raise ValueError(f"Upload at most {self.settings.max_files} files at a time.")

        seen_doc_ids: set[str] = set()
        docs: list[DocumentRecord] = []
        logs: list[str] = []
        remaining_page_budget = self.settings.max_total_pages

        for path in paths:
            try:
                if remaining_page_budget <= 0:
                    raise ValueError("Total page budget reached for this batch.")
                if not path.exists():
                    raise FileNotFoundError(f"Missing file: {path}")
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.settings.max_file_size_mb:
                    raise ValueError(
                        f"{path.name} is {size_mb:.1f} MB. Max allowed is {self.settings.max_file_size_mb} MB."
                    )

                doc_id = make_doc_id(path.name, seen_doc_ids)
                doc = self.extractor.extract(doc_id, path, remaining_page_budget)
                remaining_page_budget -= doc.page_count

                doc.summary = self._summarize_document(doc)
                for page in doc.pages:
                    page.summary = self._summarize_page(page.doc_name, page.page_num, page.text)
                docs.append(doc)
                warnings = sum(len(page.warnings) for page in doc.pages) + len(doc.warnings)
                logs.append(
                    f"{doc.name}: {doc.page_count} page(s), {sum(len(page.tables) for page in doc.pages)} table(s), "
                    f"sources={', '.join(sorted({page.source for page in doc.pages}))}, warnings={warnings}"
                )
            except Exception as exc:
                logs.append(f"{path.name}: skipped ({exc})")

        if not docs:
            raise ValueError("No files could be processed. Check file type, size, page limits, or API setup.")

        chunks = build_chunks(docs, self.settings)
        vectors = self.retriever.build_vectors(chunks)
        self.state.docs = docs
        self.state.chunks = chunks
        self.state.vectors = vectors
        self.state.reset_for_documents()
        self.state.zip_path = self._build_zip(docs)
        return docs, logs

    def _summarize_document(self, doc: DocumentRecord) -> str:
        excerpt = "\n\n".join(page.text for page in doc.pages[: min(4, len(doc.pages))])
        prompt = load_prompt("document_summary.txt").format(doc_name=doc.name, text=excerpt[:12000])
        try:
            return self.primary_backend.complete(prompt).strip()
        except Exception:
            first_page = doc.pages[0].text[:800] if doc.pages else ""
            return f"{doc.name}: extracted document. Preview: {first_page}"

    def _summarize_page(self, doc_name: str, page_num: int, text: str) -> str:
        head = " ".join(text.split()[:70]).strip()
        return f"{doc_name} page {page_num}: {head}"

    def _build_zip(self, docs: list[DocumentRecord]) -> str | None:
        import zipfile

        table_paths = [table.path for doc in docs for page in doc.pages for table in page.tables]
        if not table_paths:
            return None
        zip_path = self.settings.work_dir / "tables.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in table_paths:
                if path.exists():
                    archive.write(path, arcname=path.name)
        return str(zip_path)

    def answer(self, question: str) -> tuple[str, list[SearchHit], bool]:
        if not question.strip():
            return "", [], False

        if not self.state.has_docs:
            prompt = load_prompt("general_answer.txt").format(question=question)
            answer = self.primary_backend.complete(prompt)
            return answer, [], False

        scope, hits = self.retriever.search(
            question,
            self.state.docs,
            self.state.chunks,
            self.state.vectors,
            self.state.memory,
        )

        if scope.external_query:
            prompt = load_prompt("general_answer.txt").format(question=question.replace("/web", "", 1).strip())
            answer = self.primary_backend.complete(prompt)
            return answer, [], False

        context = self._build_context(question, hits, scope.overview_query)
        prompt = load_prompt("document_answer.txt").format(
            memory_summary=self.state.memory.rolling_summary or "(none)",
            recent_turns=format_recent_turns(self.state.memory),
            context=context,
            question=question,
        )
        answer = self.primary_backend.complete(prompt)
        self.state.memory, reset_chat = update_memory(
            self.state.memory,
            question,
            answer,
            hits,
            self.primary_backend.complete,
            self.settings,
        )
        append_history(self.settings, [doc.name for doc in self.state.docs], question, answer)
        return answer, hits, reset_chat

    def stream_answer(self, question: str):
        if not question.strip():
            yield "", [], False
            return

        if not self.state.has_docs:
            prompt = load_prompt("general_answer.txt").format(question=question)
            response = ""
            for token in self.primary_backend.stream(prompt):
                response += token
                yield response, [], False
            return

        scope, hits = self.retriever.search(
            question,
            self.state.docs,
            self.state.chunks,
            self.state.vectors,
            self.state.memory,
        )

        if scope.external_query:
            prompt = load_prompt("general_answer.txt").format(question=question.replace("/web", "", 1).strip())
            response = ""
            for token in self.primary_backend.stream(prompt):
                response += token
                yield response, [], False
            return

        context = self._build_context(question, hits, scope.overview_query)
        prompt = load_prompt("document_answer.txt").format(
            memory_summary=self.state.memory.rolling_summary or "(none)",
            recent_turns=format_recent_turns(self.state.memory),
            context=context,
            question=question,
        )

        response = ""
        for token in self.primary_backend.stream(prompt):
            response += token
            yield response, hits, False

        self.state.memory, reset_chat = update_memory(
            self.state.memory,
            question,
            response,
            hits,
            self.primary_backend.complete,
            self.settings,
        )
        append_history(self.settings, [doc.name for doc in self.state.docs], question, response)
        yield response, hits, reset_chat

    def _build_context(self, question: str, hits: list[SearchHit], overview_query: bool) -> str:
        if overview_query:
            return "\n\n".join(
                f"[Document {doc.name}]\n{doc.summary}\n" for doc in self.state.docs
            )

        blocks: list[str] = []
        seen = set()
        for hit in hits[: self.settings.retrieval_k]:
            page = hit.chunk.page_num
            key = (hit.chunk.doc_id, page, hit.chunk.kind, hit.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            header = f"[{hit.chunk.doc_name} | page {page or 'summary'} | {hit.chunk.kind} | score={hit.score:.2f}]"
            blocks.append(f"{header}\n{hit.chunk.text}")
        if not blocks:
            blocks.append(
                "\n\n".join(f"[Document {doc.name}]\n{doc.summary}" for doc in self.state.docs)
            )
        return "\n\n".join(blocks)
