from functools import wraps
import cProfile
import pstats
import io


def time(file=''):
    def decorator(func):
        wraps(func)
        def wrapper(*args, **kwargs):        
            pr = cProfile.Profile()
            pr.enable()
            func(*args, **kwargs)
            
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
            ps.print_stats()
        
            with open(f"timings/timing-{file}.txt", 'w+') as f:
                f.write(s.getvalue())
        return wrapper
    return decorator
