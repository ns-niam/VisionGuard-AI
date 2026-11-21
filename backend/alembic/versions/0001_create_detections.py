import sqlalchemy as sa
from alembic import op

revision = "0001_create_detections"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "detections",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False),
        sa.Column("image_width", sa.Integer, nullable=False),
        sa.Column("image_height", sa.Integer, nullable=False),
        sa.Column("detection_count", sa.Integer, nullable=False),
        sa.Column("confidence_threshold", sa.Float, nullable=False),
        sa.Column("iou_threshold", sa.Float, nullable=False),
        sa.Column("image_size", sa.Integer, nullable=False),
        sa.Column("inference_latency_ms", sa.Float, nullable=False),
        sa.Column("detections", sa.JSON, nullable=False),
        sa.Column("annotated_image", sa.LargeBinary, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_detections_created_at", "detections", ["created_at"])


def downgrade():
    op.drop_index("ix_detections_created_at", table_name="detections")
    op.drop_table("detections")
