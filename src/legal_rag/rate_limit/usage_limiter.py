from datetime import datetime, timezone
from fastapi import HTTPException, status
from src.legal_rag.rate_limit.redis_client import reserve_quota, adjust_quota

SECONDS_UNTIL_MIDNIGHT_BUFFER = 90_000        
SECONDS_UNTIL_MONTH_END_BUFFER = 33 * 86_400  


def _day_key(user_id: str, resource: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"quota:{user_id}:{resource}:day:{day}"


def _month_key(user_id: str, resource: str) -> str:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    return f"quota:{user_id}:{resource}:month:{month}"


def check_and_reserve(user_id: str, resource: str, amount: int, daily_limit: int, monthly_limit: int):
    """Atomically checks BOTH daily and monthly limits. Rolls back daily if monthly fails."""
    day_key = _day_key(user_id, resource)
    month_key = _month_key(user_id, resource)

    ok_day, used_day, _ = reserve_quota(keys=[day_key], args=[amount, daily_limit, SECONDS_UNTIL_MIDNIGHT_BUFFER])
    if not ok_day:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily limit reached for {resource} ({used_day}/{daily_limit}). Remaining quota is insufficient to process another request. Resets at midnight UTC.",
        )

    ok_month, used_month, _ = reserve_quota(keys=[month_key], args=[amount, monthly_limit, SECONDS_UNTIL_MONTH_END_BUFFER])
    if not ok_month:
        # roll back the daily increment since monthly failed
        adjust_quota(keys=[day_key], args=[-amount, SECONDS_UNTIL_MIDNIGHT_BUFFER])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly limit reached for {resource} ({used_month}/{monthly_limit}). Resets next month.",
        )


def adjust_actual_usage(user_id: str, resource: str, delta: int):
    """Correct daily+monthly counters once the real cost (e.g. actual tokens) is known."""
    day_key = _day_key(user_id, resource)
    month_key = _month_key(user_id, resource)
    day_adjusted_token = adjust_quota(keys=[day_key], args=[delta, SECONDS_UNTIL_MIDNIGHT_BUFFER])
    month_adjusted_token = adjust_quota(keys=[month_key], args=[delta, SECONDS_UNTIL_MONTH_END_BUFFER])

    return day_adjusted_token, month_adjusted_token