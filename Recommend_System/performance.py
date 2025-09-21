import time
import functools
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='performance.log'
)
performance_logger = logging.getLogger('performance')

def time_function(func):
    """
    Decorator to time the execution of functions and log the results
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        performance_logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        print(f"Completed {func.__name__} in {execution_time:.2f} seconds")
        return result
    return wrapper
