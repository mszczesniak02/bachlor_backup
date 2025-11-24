# autopep8: off

import sys
import os
original_sys_path = sys.path.copy()

# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# importing commons
from segmentation.common.dataloader import *
from segmentation.common.model import *
from segmentation.common.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

# normal imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
# autopep8: off

#--------------------------------------------------------------------------

def time_it(func):
    def foo(*args, **kw):
        t_start = time.time()
        result = func(*args, **kw)
        t_stop = time.time() - t_start
        return t_stop, result
    return foo


def evaluate_init(model_name:str="segformer"):
    model = model_load(model_name=model_name, device="cpu")
    model.eval()
    model.to("cpu")

    dataset = dataset_get(img_path=IMG_TEST_PATH,
                          mask_path=MASK_TEST_PATH, transform=transform_val)

    return model, dataset


@time_it
def predict_single(model, img):
    return model(img)


def evaluate_amount(model, dataset, amount=100):
    times = []
    imgs = []
    msks = []
    measurements = {}

    for idx, (i, m) in enumerate(iter(dataset)):
        if idx == amount-1:
            break
        imgs.append(i)
        msks.append(m)
    jdx = 0
    max_len = len(imgs)
    for measurement in tqdm(range(max_len - 1), desc="Measurement", leave=False):
        t, out = predict_single(model, imgs[jdx].unsqueeze(0))
        times.append(t)

        # out = cleanup(ten2np(out))
        msk = ten2np(msks[jdx])
        out = out.flatten()
        msk = msk.flatten()

        convergence = []

        for idx, _ in enumerate(out):
            if (out[idx] == msk[idx]):
                convergence.append(1)
            else:
                convergence.append(0)
        conv_sum = np.array(convergence).sum()
        conv_len = len(convergence)
        measurements[measurement] = {
            "0": conv_len - 1 - conv_sum,
            "1": conv_sum,
            "%": conv_sum/(conv_len-1)*100.0
        }
        jdx += 1
    return times, measurements


def main():
    print(f"Testing model: {MODEL_INFERENCE_PATH} ")

    model, dataset = evaluate_init(model_name="segformer")
    times, measurements = evaluate_amount(model, dataset)
    percentages = [i["%"] for i in measurements.values()]

    print(
        f"model prediction times\navg={(np.mean(times) * 1000):.3f} ms\tmedian={(np.median(times)*1000):.3f} ms\tmin={(np.min(times)*1000):.3f} ms\tmax={(np.max(times)*1000):3f} ms")
    print("\n\n")

    print(f"convergence:\navg={(np.mean(percentages)):.3f}%\tmedian={(np.median(percentages)):.3f}%\tmin={(np.min(percentages)):.3f}%\tmax={(np.max(percentages)):3f}%")


if __name__ == "__main__":
    main()
