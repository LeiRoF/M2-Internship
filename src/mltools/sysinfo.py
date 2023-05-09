import psutil
import GPUtil
import numpy as np

def get():
    try:
        return f"CPU: {psutil.cpu_percent()}%"\
        + f", GPU: {np.mean([i.load for i in GPUtil.getGPUs()])*100:.1f}%"\
        + f", RAM: {psutil.virtual_memory().percent}%"\
        + f" ({psutil.virtual_memory().used/1024**3:.1f}GB"\
        + f"/{psutil.virtual_memory().total/1024**3:.1f}GB)"
    except:
        return f"CPU: {psutil.cpu_percent()}%"
        + f", RAM: {psutil.virtual_memory().percent}%"\
        + f" ({psutil.virtual_memory().used/1024**3:.1f}GB"\
        + f"/{psutil.virtual_memory().total/1024**3:.1f}GB)"
