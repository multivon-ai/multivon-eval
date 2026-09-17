"""Native runner input checks and ownership of asynchronous child tasks."""
from __future__ import annotations

import asyncio
import math


def validate_run_options(values):
    """Reject invalid controls before preparation or any target/judge work."""
    for name in ('runs', 'workers', 'concurrency', 'evaluator_concurrency'):
        if name not in values:
            continue
        value = values[name]
        if value is None and name in ('workers', 'evaluator_concurrency'):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('fail_threshold', 'max_error_rate'):
        value = values.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not 0 <= value <= 1 or not math.isfinite(value)):
            raise ValueError(f'{name} must be a finite number between 0 and 1')


async def gather_owned(*coroutines):
    """Cancel and drain our children on any escaping error, including cancellation.

    asyncio.gather alone leaves siblings running when one child cancels itself
    or raises. Cancellation cannot terminate synchronous work in a worker thread.
    """
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
