"""Background worker. Postgres is the queue: `FOR UPDATE SKIP LOCKED` lets several workers share it safely.

Run: python -m app.worker
"""

import logging
import time
import traceback

from sqlalchemy import select, update

from . import pipeline
from .db import Meeting, Session

log = logging.getLogger("worker")


def claim() -> str | None:
    with Session.begin() as db:
        m = db.execute(
            select(Meeting).where(Meeting.status == "queued").order_by(Meeting.created_at).limit(1).with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if m:
            m.status, m.error = "processing", None
            return m.id
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # ponytail: single-worker recovery — jobs left "processing" by a crash are re-queued at start.
    # With several workers, add a heartbeat column and re-queue only stale jobs.
    while True:
        try:
            with Session.begin() as db:
                db.execute(update(Meeting).where(Meeting.status == "processing").values(status="queued"))
            break
        except Exception as e:  # DB not up yet (e.g. right after boot): wait for it
            log.warning("database unavailable, retrying in 5 s: %s", str(e).splitlines()[0])
            time.sleep(5)
    log.info("worker ready")
    while True:
        try:
            mid = claim()
        except Exception:  # DB restarting or migrating: wait and try again
            log.exception("queue unavailable")
            time.sleep(5)
            continue
        if not mid:
            time.sleep(1)
            continue
        log.info("processing %s", mid)
        try:
            pipeline.run(mid)
            log.info("done %s", mid)
        except Exception as e:  # keep the worker alive; the error is shown in the UI and the stage can be retried
            log.error("failed %s\n%s", mid, traceback.format_exc())
            with Session.begin() as db:
                db.execute(update(Meeting).where(Meeting.id == mid).values(status="error", error=str(e)[:2000]))


if __name__ == "__main__":
    main()
