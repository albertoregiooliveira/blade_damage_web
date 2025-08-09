import os
import psutil

def get_memory_info():
    process = psutil.Process(os.getpid())
    print(f"Memória usada: {process.memory_info().rss / 1024 ** 2:.2f} MB")