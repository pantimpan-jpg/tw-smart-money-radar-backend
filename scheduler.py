from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from .scanner import run_scan


def main() -> None:
    scheduler = BlockingScheduler(timezone="Asia/Taipei")
    scheduler.add_job(run_scan, "cron", day_of_week="mon-fri", hour=14, minute=35, kwargs={"save": True})
    scheduler.add_job(run_scan, "cron", day_of_week="mon-fri", hour=20, minute=0, kwargs={"save": True})
    print("Scheduler started. Jobs: weekdays 14:35 and 20:00 Asia/Taipei")
    scheduler.start()


if __name__ == "__main__":
    main()
