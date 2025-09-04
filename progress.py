import sys
import time
import threading

class ProgressIndicator:
    def __init__(self, message="Working"):
        self.message = message
        self.active = False
        self.thread = None
    
    def start(self, message=None):
        if message:
            self.message = message
        self.active = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join()
        # Clear the progress line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
    
    def _animate(self):
        chars = "|/-\\"
        idx = 0
        while self.active:
            sys.stdout.write('\r' + self.message + ' ' + chars[idx % len(chars)])
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

def with_progress(message):
    """Decorator to show progress during function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            progress = ProgressIndicator(message)
            progress.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                progress.stop()
        return wrapper
    return decorator
