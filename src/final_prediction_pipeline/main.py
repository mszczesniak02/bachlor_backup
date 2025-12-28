
from final_prediction_pipeline.prediction import FinalPipeline
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))


def main():
    print("--- Final Prediction Pipeline Test ---")

    # 1. Initialize Pipeline
    pipeline = FinalPipeline()

    # 2. Get Sample Image
    # Try to find a sample image
    sample_image_path = os.path.join(os.path.dirname(
        __file__), "../segmentation/tiny.png")  # Adjust if needed

    if not os.path.exists(sample_image_path):
        sample_image_path = os.path.join(os.path.dirname(
            __file__), "../autoencoder/figure_1.png")

    if not os.path.exists(sample_image_path):
        print("No sample image found to test.")
        return

    print(f"Testing on image: {sample_image_path}")

    # 3. Run Prediction
    is_in_domain, category, mask = pipeline.predict(sample_image_path)

    # 4. Show Results
    print(f"\n--- Result ---")
    print(f"In Domain: {is_in_domain}")

    if is_in_domain:
        print(f"Category: {category}")
        print(f"Mask Shape: {mask.shape if mask is not None else 'None'}")

        # Save results if possible
        if mask is not None:
            # Visualize
            img = cv2.imread(sample_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Prediction Mask")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

            out_path = "prediction_test_result.png"
            plt.savefig(out_path)
            print(f"Visual result saved to {out_path}")
    else:
        print(f"Failure Reason: {category}")


if __name__ == "__main__":
    main()
