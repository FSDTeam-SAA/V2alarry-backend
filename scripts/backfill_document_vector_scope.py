"""Operator-run Qdrant payload backfill for existing document vectors.

The command is a dry run unless --apply is supplied.
"""

import argparse

from qdrant_client.http import models

from app.core.vector_store import VectorStore
from app.db.database import SessionLocal
from app.models.document import Document


def backfill(*, apply: bool) -> None:
    store = VectorStore()
    with SessionLocal() as db:
        documents = db.query(Document).order_by(Document.uploaded_at.asc()).all()
        for document in documents:
            print(
                f"document={document.id} global={document.is_global} "
                f"target_user_id={document.target_user_id} apply={apply}"
            )
            if not apply:
                continue
            store.client.set_payload(
                collection_name=store.collection_name,
                payload={
                    "is_global": document.is_global,
                    "target_user_id": document.target_user_id,
                },
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document.id)),
                        )
                    ]
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    backfill(apply=args.apply)


if __name__ == "__main__":
    main()
