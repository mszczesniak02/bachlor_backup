#autopep8: off
import sys
import os
import time
import psutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

original_sys_path = sys.path.copy()

# moving to "classification/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# importing commons
from classification.common.dataloader import *
from classification.common.model import *
from classification.common.hparams import *

# importing utils
from utils.utils import *

# go back to the origin path
sys.path = original_sys_path

CLASS_NAMES_DISPLAY = ["Włosowate", "Małe", "Średnie", "Duże"]

class Benchmark:
    def __init__(self, device):
        self.device = device
        self.results = []

    def get_resource_usage(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpu_usage = 0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / 1024**2  # MB

        return cpu_usage, ram_usage, gpu_usage

    def evaluate_model(self, model_name, models, dataloader):
        print(f"\n--- Benchmarking {model_name} ---")

        # Move models to device and eval mode
        for model in models:
            model.to(self.device).eval()

        predictions = []
        ground_truth = []

        total_time = 0
        cpu_usages = []
        ram_usages = []
        gpu_usages = []

        total_samples = 0

        # Pre-warm GPU
        if torch.cuda.is_available():
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
            with torch.no_grad():
                for model in models:
                    model(dummy_input)

        np.random.seed(42) # Ensure consistency

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Testing {model_name}")):
                images = images.to(self.device)

                # Resource measurement start
                start_time = time.time()

                # Ensemble logic inside
                all_probs = []
                for model in models:
                    outputs = model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probs.append(probabilities)

                avg_probs = torch.stack(all_probs).mean(dim=0)
                preds = avg_probs.argmax(dim=1).cpu().numpy()

                end_time = time.time()
                total_time += (end_time - start_time)

                predictions.extend(preds)
                ground_truth.extend(labels.numpy())
                total_samples += images.size(0)

                # Resource sampling
                c, r, g = self.get_resource_usage()
                cpu_usages.append(c)
                ram_usages.append(r)
                gpu_usages.append(g)

        # Metrics
        accuracy = accuracy_score(ground_truth, predictions)
        report = classification_report(ground_truth, predictions, target_names=CLASS_NAMES_DISPLAY, output_dict=True)
        conf_matrix = confusion_matrix(ground_truth, predictions)

        avg_inference_time = (total_time / total_samples) * 1000 # ms per image

        # Store comprehensive metrics
        result = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Avg Inference Time (ms/img)": avg_inference_time,
            "Total Time (s)": total_time,
            "Avg CPU (%)": np.mean(cpu_usages),
            "Max RAM (%)": np.max(ram_usages),
            "Max GPU Mem (MB)": np.max(gpu_usages)
        }

        # Add per-class precision/recall
        for cls_name in CLASS_NAMES_DISPLAY:
            if cls_name in report:
                result[f"{cls_name} Precision"] = report[cls_name]['precision']
                result[f"{cls_name} Recall"] = report[cls_name]['recall']
                result[f"{cls_name} F1-Score"] = report[cls_name]['f1-score']

        self.results.append(result)

        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Avg Inference Time: {avg_inference_time:.2f} ms/sample")
        print("Confusion Matrix:")
        print(pd.DataFrame(conf_matrix, index=[f"True {c}" for c in CLASS_NAMES_DISPLAY], columns=[f"Pred {c}" for c in CLASS_NAMES_DISPLAY]))

        return result

    def save_results(self, filename="benchmark_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nBenchmark results saved to {filename}")
        print("\nSummary Table:")
        print(df[["Model", "Accuracy", "Avg Inference Time (ms/img)", "Max GPU Mem (MB)"]].to_string(index=False))


def main():
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on: {device}")

    # Initialize Benchmark
    benchmark = Benchmark(device)

    # 1. Load Models
    print("Loading models...")

    path_eff = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/newer/efficientnet/efficientnet_f1_0.9134_epoch15.pth"
    path_conv = "/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/classification/newer/convnext/convnext.pth"

    try:
        model_eff = model_load("efficienet", filepath=path_eff, device=device)
        model_conv = model_load("convnet", filepath=path_conv, device=device)
    except Exception as e:
        print(f"Critical Error loading models: {e}")
        return

    # 2. Prepare Valid Dataloader
    # Using dataloader_get from common.dataloader which returns DataLoader directly
    valid_dl = dataloader_get(TEST_DIR, batch_size=16, image_size=IMAGE_SIZE, is_training=False, num_workers=4)

    # 3. Run Benchmarks

    # EfficientNet
    benchmark.evaluate_model("EfficientNet", [model_eff], valid_dl)

    # ConvNeXt
    benchmark.evaluate_model("ConvNeXt", [model_conv], valid_dl)

    # Ensemble
    benchmark.evaluate_model("Ensemble (Eff+Conv)", [model_eff, model_conv], valid_dl)

    # 4. Save and Print
    benchmark.save_results()

if __name__ == "__main__":
    main()
