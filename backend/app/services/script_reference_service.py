from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import ScriptReferenceDocument, ScriptReferenceFragment, ScriptReferenceLibraryState


class ScriptReferenceService:
    SUPPORTED_SUFFIXES = {
        ".md",
        ".markdown",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".doc",
        ".docx",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".xls",
        ".xlsx",
    }
    DEFAULT_LIBRARY_PATH = Path(settings.redis_url or r"E:\topdesk\毕业设计\剧本杀")

    @staticmethod
    def _default_rebuild_status(root: str = "") -> dict[str, Any]:
        return {
            "running": False,
            "completed": False,
            "root": root,
            "documents": 0,
            "fragments": 0,
            "processed_files": 0,
            "total_files": 0,
            "skipped": 0,
            "last_error": "",
            "started_at": "",
            "finished_at": "",
        }

    def ensure_library_ready(self, force: bool = False) -> None:
        self._load_or_create_state(SessionLocal(), self.DEFAULT_LIBRARY_PATH)

    def _ensure_library_ready_sync(self) -> None:
        self.ensure_library_ready()

    def ingest_library(self, db: Session, library_path: str | Path | None = None) -> dict[str, Any]:
        root = Path(library_path or self.DEFAULT_LIBRARY_PATH)
        state = self._load_or_create_state(db, root)
        documents = db.execute(select(ScriptReferenceDocument)).scalars().all()
        fragments = db.execute(select(ScriptReferenceFragment)).scalars().all()
        return {
            "root_path": str(root),
            "documents": len(documents),
            "fragments": len(fragments),
            "processed_files": state.processed_files,
            "total_files": state.total_files,
            "skipped": state.skipped,
            "running": state.running,
            "completed": state.completed,
            "last_error": state.last_error,
        }

    def get_rebuild_status(self) -> dict[str, Any]:
        db = SessionLocal()
        try:
            try:
                state = self._load_or_create_state(db, self.DEFAULT_LIBRARY_PATH)
                return self._sync_status_from_state(state)
            except SQLAlchemyError:
                db.rollback()
                return self._default_rebuild_status(str(self.DEFAULT_LIBRARY_PATH))
        finally:
            db.close()

    def get_library_stats(self, db: Session) -> dict[str, Any]:
        try:
            snapshot = self.ingest_library(db)
            return {
                "documents": int(snapshot.get("documents", 0)),
                "fragments": int(snapshot.get("fragments", 0)),
                "documents_by_category": {},
                "fragments_by_type": {},
            }
        except SQLAlchemyError:
            db.rollback()
            return {
                "documents": 0,
                "fragments": 0,
                "documents_by_category": {},
                "fragments_by_type": {},
            }

    def search_fragments(self, db: Session, query: str, limit: int = 5, category: str = "", fragment_type: str = "") -> list[dict[str, Any]]:
        rows = db.execute(select(ScriptReferenceFragment)).scalars().all()
        query_text = (query or "").strip().lower()
        results: list[dict[str, Any]] = []
        for row in rows:
            haystack = " ".join([row.title or "", row.content or "", " ".join(row.keywords or [])]).lower()
            if query_text and query_text not in haystack:
                continue
            if category and category not in (row.metadata_json or {}).get("category", ""):
                continue
            if fragment_type and fragment_type != row.fragment_type:
                continue
            results.append(
                {
                    "id": row.id,
                    "document_id": row.document_id,
                    "source_path": row.source_path,
                    "title": row.title,
                    "content": row.content,
                    "fragment_type": row.fragment_type,
                    "keywords": row.keywords or [],
                    "metadata": row.metadata_json or {},
                }
            )
        return results[: max(1, limit)]

    def build_reference_context(self, fragments: list[dict[str, Any]]) -> str:
        return "\n\n".join(f"[{item.get('title', '')}]\n{item.get('content', '')}" for item in fragments if item.get("content"))

    def get_or_create_default_state(self, db: Session) -> ScriptReferenceLibraryState:
        return self._load_or_create_state(db, self.DEFAULT_LIBRARY_PATH)

    def _load_or_create_state(self, db: Session, root: Path) -> ScriptReferenceLibraryState:
        state = db.execute(select(ScriptReferenceLibraryState).where(ScriptReferenceLibraryState.root_path == str(root))).scalar_one_or_none()
        if state:
            return state
        state = ScriptReferenceLibraryState(root_path=str(root))
        db.add(state)
        db.commit()
        db.refresh(state)
        return state

    def _sync_status_from_state(self, state: ScriptReferenceLibraryState) -> dict[str, Any]:
        return {
            "running": bool(state.running),
            "completed": bool(state.completed),
            "root": state.root_path,
            "documents": int(state.documents),
            "fragments": int(state.fragments),
            "processed_files": int(state.processed_files),
            "total_files": int(state.total_files),
            "skipped": int(state.skipped),
            "last_error": state.last_error,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
        }
