import time
import traceback
import asyncio
import inspect
from functools import wraps
from typing import List, Callable, Optional


def post_iter_action(
    post_actions: Optional[List[Callable]] = None,
    timing=False,
    select_from_kwargs=None,
    catch_exceptions=True,
    exception_handler=None,
):
    post_actions = post_actions or []
    select_from_kwargs = select_from_kwargs or []

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time() if timing else None

            result = await func(*args, **kwargs)

            if timing:
                elapsed = time.time() - start_time
                print(f"[Timer] Iteration took {elapsed:.4f} seconds.")

            try:
                async with asyncio.TaskGroup() as tg:
                    for action in post_actions:
                        selected_kwargs = {
                            k: kwargs[k] for k in select_from_kwargs if k in kwargs
                        }
                        if inspect.iscoroutinefunction(action):
                            tg.create_task(action(*args, **selected_kwargs))
                        else:
                            # Wrap sync action into async task
                            async def sync_wrapper(fn, *args, **kwargs):
                                fn(*args, **kwargs)

                            tg.create_task(
                                sync_wrapper(action, *args, **selected_kwargs)
                            )
            except Exception as e:
                if catch_exceptions:
                    print(f"[Warning] Post-actions failed:")
                    traceback.print_exc()
                    if exception_handler:
                        exception_handler(e)
                else:
                    raise

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if inspect.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            else:
                return asyncio.run(async_wrapper(*args, **kwargs))

        return sync_wrapper

    return decorator
