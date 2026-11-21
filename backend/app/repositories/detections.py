from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models.detection import Detection


def create(db: Session, **values) -> Detection:
    record = Detection(**values)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get(db: Session, detection_id: UUID) -> Detection | None:
    return db.get(Detection, detection_id)


def list_records(
    db: Session, search: str | None, offset: int, limit: int
) -> tuple[list[Detection], int]:
    query = select(Detection)
    count_query = select(func.count()).select_from(Detection)
    if search:
        predicate = Detection.filename.ilike(f"%{search}%")
        query = query.where(predicate)
        count_query = count_query.where(predicate)
    records = list(
        db.scalars(
            query.order_by(Detection.created_at.desc()).offset(offset).limit(limit)
        )
    )
    return records, int(db.scalar(count_query) or 0)


def delete(db: Session, record: Detection) -> None:
    db.delete(record)
    db.commit()
