import gc
from torch.cuda import empty_cache


def utils_cuda_clear():
    """ Clear the leftover memory from COLAB training model
    """

    print("Clearning memory...", end="")
    empty_cache()
    gc.collect()
    empty_cache()
    print("done.")
