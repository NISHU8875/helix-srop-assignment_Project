"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.

TODO for candidate: implement these tools.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Get the user's most recent builds, ordered by date descending.
    
    Args:
        user_id: The user identifier
        limit: Maximum number of builds to return (default 5)
        
    Returns:
        List of BuildSummary objects with build metadata
    """
    # Mock data for demonstration
    now = datetime.utcnow()
    mock_builds = [
        BuildSummary(
            build_id=f"build_{i:04d}",
            pipeline="main",
            status=["passed", "failed", "cancelled"][i % 3],
            branch="main" if i % 2 == 0 else "develop",
            started_at=now - timedelta(hours=i),
            duration_seconds=300 + (i * 30),
        )
        for i in range(1, limit + 1)
    ]
    return mock_builds


async def get_account_status(user_id: str) -> AccountStatus:
    """
    Get the user's current account status including plan tier and usage metrics.
    
    Args:
        user_id: The user identifier
        
    Returns:
        AccountStatus object with current plan and usage information
    """
    # Mock status — in reality this would query the DB
    return AccountStatus(
        user_id=user_id,
        plan_tier="pro",  # Would fetch from sessions.user.plan_tier
        concurrent_builds_used=2,
        concurrent_builds_limit=5,
        storage_used_gb=12.5,
        storage_limit_gb=100.0,
    )


