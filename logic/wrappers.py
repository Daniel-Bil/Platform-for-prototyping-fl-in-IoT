import time
from colorama import Fore
def time_wrapper(func):

    def wrap(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()

        print(f"{Fore.BLUE}{func.__name__}{Fore.GREEN} {(end-start):^3.3f}s{Fore.RESET}")
        return result
    return wrap
