from hparams import *
from model import *
from dataloader import *
from plots import *
from post_seg_cleanup import cleanup


from tqdm import tqdm
import time
import os


def time_it(func):
    def foo(*args, **kw):
        t_start = time.time()
        result = func(*args, **kw)
        t_stop = time.time() - t_start
        return t_stop, result
    return foo


def evaluate_init():
    model = model_load(device="cpu")
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

        out = cleanup(ten2np(out))
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

    model, dataset = evaluate_init()
    times, measurements = evaluate_amount(model, dataset)
    percentages = [i["%"] for i in measurements.values()]

    # print(f"t_avg={(np.mean(times) * 1000):.3f} ; t_median={(np.median(times)*1000):.3f} ; t_min={(np.min(times)*1000):.3f}; t_max={(np.max(times)*1000):3f}")

    print(
        f"model prediction times\navg={(np.mean(times) * 1000):.3f} ms\tmedian={(np.median(times)*1000):.3f} ms\tmin={(np.min(times)*1000):.3f} ms\tmax={(np.max(times)*1000):3f} ms")
    print("\n\n")

    print(f"convergence:\navg={(np.mean(percentages)):.3f}%\tmedian={(np.median(percentages)):.3f}%\tmin={(np.min(percentages)):.3f}%\tmax={(np.max(percentages)):3f}%")


if __name__ == "__main__":
    main()
