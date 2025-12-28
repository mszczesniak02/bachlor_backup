#autopep8: off

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json

# Add src to path for imports
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

import geometric_analysis.analytics as analytics
from final_prediction_pipeline.hparams import *
from final_prediction_pipeline import prediction


class CrackAnalysisPipeline:
    def __init__(self, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Pipeline on {self.device}...")

        # Load models using prediction module
        (self.segformer,
         self.unet,
         self.yolo,
         self.efficientnet,
         self.convnext,
         self.domain_controller) = prediction.load_all_models(device=self.device)

        # Move models to device if not already (load_all_models usually handles this)
        # But let's ensure domain controller is in eval mode
        if self.domain_controller:
            self.domain_controller.to(self.device).eval()

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses image for models.
        Returns: img_tensor, img_numpy, original_shape
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Use prediction's load_image logic which resizes to OUTPUT_IMAGE_SIZE
        # But we might want to keep original size for final visualization?
        # The models need 512x512 usually.
        img_tensor, img_resized = prediction.load_image(image_path)
        img_tensor = img_tensor.to(self.device)

        # Also load original for analytics overlay if we want high res?
        # For now, working with the resized version used for inference is consistent.

        return img_tensor, img_resized

    def run_domain_controller(self, img_tensor):
        """
        Determines if the image contains a crack or not.
        Returns: is_crack (bool), confidence (float), label (str)
        """
        if self.domain_controller is None:
            return None, 0.0, "N/A"

        try:
            with torch.no_grad():
                # Entry model is SimpleClassificator (Binary)
                # Output shape [1, 2] -> [No Crack, Crack]
                logits = self.domain_controller(img_tensor)
                probs = F.softmax(logits, dim=1)

                # Assuming index 1 is Crack, 0 is No Crack (based on class_names ["no_crack", "crack"])
                crack_prob = probs[0, 1].item()
                prediction_idx = torch.argmax(probs, dim=1).item()

                is_crack = (prediction_idx == 1)
                label = "Crack" if is_crack else "No Crack"

                return is_crack, crack_prob, label
        except Exception as e:
            print(f"Domain Controller Error: {e}")
            return None, 0.0, "Error"

    def run_segmentation_ensemble(self, img_tensor, img_numpy):
        """
        Runs U-Net, SegFormer, YOLO and averages results.
        Returns: binary_mask, final_mask (heatmap), masks_dict
        """
        # We can reuse logic from prediction.predict but tailored to return values

        # Weights (hardcoded as in prediction.py)
        weights = {"unet": 1.0, "segformer": 1.0, "yolo": 1.0}

        # Normalize weights based on loaded models
        active_sum = 0
        if self.unet:
            active_sum += weights["unet"]
        if self.segformer:
            active_sum += weights["segformer"]
        if self.yolo:
            active_sum += weights["yolo"]

        norm_weights = {}
        if active_sum > 0:
            if self.unet:
                norm_weights["unet"] = weights["unet"] / active_sum
            if self.segformer:
                norm_weights["segformer"] = weights["segformer"] / active_sum
            if self.yolo:
                norm_weights["yolo"] = weights["yolo"] / active_sum

        mask_accum = np.zeros(
            (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.float32)
        masks_dict = {}

        # 1. U-Net
        if self.unet:
            try:
                with torch.no_grad():
                    out = self.unet(img_tensor)
                    mask = torch.sigmoid(out).squeeze().cpu().numpy()
                    mask_accum += mask * norm_weights["unet"]
                    masks_dict["unet"] = mask
            except Exception as e:
                print(f"U-Net Error: {e}")

        # 2. SegFormer
        if self.segformer:
            try:
                with torch.no_grad():
                    out = self.segformer(img_tensor)
                    out = F.interpolate(out, size=(OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE),
                                        mode='bilinear', align_corners=False)
                    mask = torch.sigmoid(out).squeeze().cpu().numpy()
                    mask_accum += mask * norm_weights["segformer"]
                    masks_dict["segformer"] = mask
            except Exception as e:
                print(f"SegFormer Error: {e}")

        # 3. YOLO
        if self.yolo:
            try:
                # YOLO expects path or numpy array.
                # img_numpy is RGB, 0-255. YOLO works with that.
                results = self.yolo.predict(img_numpy, imgsz=OUTPUT_IMAGE_SIZE, verbose=False,
                                            device=0 if self.device == 'cuda' else 'cpu')

                mask_yolo = np.zeros(
                    (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.float32)
                if results and results[0].masks is not None:
                    data = results[0].masks.data
                    # Handle resize if output is smaller
                    if data.shape[1:] != (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE):
                        data = F.interpolate(data.unsqueeze(1), size=(OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE),
                                             mode='bilinear', align_corners=False).squeeze(1)

                    # Combine instances
                    mask_yolo = torch.any(
                        data > 0.5, dim=0).float().cpu().numpy()

                mask_accum += mask_yolo * norm_weights["yolo"]
                masks_dict["yolo"] = mask_yolo

            except Exception as e:
                print(f"YOLO Error: {e}")
                masks_dict["yolo"] = None

        final_mask = mask_accum
        binary_mask = (final_mask > 0.5).astype(np.uint8)  # 0 or 1

        return binary_mask, final_mask, masks_dict

    def run_classification_ensemble(self, img_tensor, binary_mask):
        """
        Runs EfficientNet and ConvNeXt on the masked image.
        Returns: category_idx, category_name, confidence
        """
        # If mask is empty, skip
        if np.sum(binary_mask) == 0:
            return -1, "No Crack Detected", 0.0

        try:
            # Create masked image input
            mask_tensor = torch.from_numpy(binary_mask).to(
                self.device).float().unsqueeze(0).unsqueeze(0)
            masked_img = img_tensor * mask_tensor

            # EfficientNet
            eff_out = None
            if self.efficientnet:
                with torch.no_grad():
                    eff_out = self.efficientnet(masked_img)

            # ConvNeXt
            conv_out = None
            if self.convnext:
                with torch.no_grad():
                    conv_out = self.convnext(masked_img)

            # Combine
            final_out = None
            if eff_out is not None and conv_out is not None:
                final_out = (eff_out + conv_out) / 2
            elif eff_out is not None:
                final_out = eff_out
            elif conv_out is not None:
                final_out = conv_out

            if final_out is not None:
                probs = F.softmax(final_out, dim=1)
                conf, pred_idx = torch.max(probs, 1)

                category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]
                pred_idx = pred_idx.item()

                class_name = category_names[pred_idx] if pred_idx < len(
                    category_names) else str(pred_idx)

                return pred_idx, class_name, conf.item()

        except Exception as e:
            print(f"Classification Ensemble Error: {e}")

        return -1, "Error", 0.0

    def run_pipeline(self, image_path, output_dir=None):
        """
        Executes the full pipeline for a single image.
        """
        print(f"Processing {image_path}...")
        results = {}

        # 1. Preprocess
        img_tensor, img_numpy = self.preprocess_image(image_path)

        # 2. Domain Controller
        is_crack, domain_conf, domain_label = self.run_domain_controller(
            img_tensor)
        results['domain_controller'] = {
            "is_crack": is_crack,
            "confidence": domain_conf,
            "label": domain_label
        }

        # 3. Segmentation
        binary_mask, heatmap, masks_dict = self.run_segmentation_ensemble(
            img_tensor, img_numpy)
        results['segmentation_completed'] = True

        # 4. Classification
        class_idx, class_name, class_conf = self.run_classification_ensemble(
            img_tensor, binary_mask)
        results['classification'] = {
            "class_index": class_idx,
            "class_name": class_name,
            "confidence": class_conf
        }

        # 5. Geometric Analysis
        # Assuming pixel_size_mm is unknown or standard, passing None for now
        # Creating a boolean mask for analytics
        bool_mask = binary_mask > 0
        geo_results, skeleton, dist_map = analytics.analyze_crack_mask(
            bool_mask)
        results['geometric_analysis'] = geo_results

        # 6. Visualization & Saving
        images = {}
        images['original'] = img_numpy
        images['binary_mask'] = binary_mask * 255  # uint8 for saving
        images['heatmap'] = (heatmap * 255).astype(np.uint8)

        # Create Overlay
        # Red overlay for crack
        overlay = img_numpy.copy()
        overlay[binary_mask == 1] = [255, 0, 0]  # simple red paint
        # improved overlay: alpha blend
        alpha = 0.4
        overlay_vis = cv2.addWeighted(img_numpy, 1-alpha, overlay, alpha, 0)
        images['overlay'] = overlay_vis

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Save Json
            import json
            # Helper to convert numpy/float32 types to python native

            def default_converter(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)

            with open(os.path.join(output_dir, f"{base_name}_results.json"), 'w') as f:
                json.dump(results, f, indent=4, default=default_converter)

            # Save Images
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), cv2.cvtColor(
                images['original'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(
                output_dir, f"{base_name}_mask.png"), images['binary_mask'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.png"), cv2.applyColorMap(
                images['heatmap'], cv2.COLORMAP_JET))
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), cv2.cvtColor(
                images['overlay'], cv2.COLOR_RGB2BGR))

            # Save Analytics Visualization (using existing function but saving to file)
            analytics_save_path = os.path.join(
                output_dir, f"{base_name}_analytics.png")
            analytics.visualize_analysis(
                img_numpy, bool_mask, skeleton, dist_map, geo_results, save_path=analytics_save_path)

        return results, images


if __name__ == "__main__":
    # Example Usage
    pipeline = CrackAnalysisPipeline()

    # Test on a file
    test_image = IMAGE_PATH_1  # From hparams
    if os.path.exists(test_image):
        pipeline.run_pipeline(test_image, output_dir="pipeline_output")
    else:
        print(f"Test image {test_image} not found.")
