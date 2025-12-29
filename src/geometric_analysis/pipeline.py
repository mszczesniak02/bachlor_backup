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

        # 3. Check for Crack
        if not is_crack:
            print(f"No crack detected in {image_path}. Stopping pipeline.")
            results['segmentation_completed'] = False
            results['classification'] = None
            results['geometric_analysis'] = None

            images = {}
            images['original'] = img_numpy

            return results, images

        # 3. Segmentation
        binary_mask, heatmap, masks_dict = self.run_segmentation_ensemble(
            img_tensor, img_numpy)
        results['segmentation_completed'] = True

        # VERIFICATION: If segmentation found nothing, override Domain Controller
        if np.sum(binary_mask) == 0:
            print(f"Segmentation found 0 pixels. Overriding DC result for {image_path}.")
            results['domain_controller']['is_crack'] = False
            results['domain_controller']['label'] = "No Crack"
            # Confidence remains high that it Was a crack, but we override decision? 
            # Or effectively say confidence of being a crack is now 0? 
            # Let's simple keep it false so UI handles it.

            results['segmentation_completed'] = False
            results['classification'] = None
            results['geometric_analysis'] = None

            images = {}
            images['original'] = img_numpy
            # Return early
            return results, images

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
        # Load original image for resizing results back to original resolution
        original_img_cv2 = cv2.imread(image_path)
        if original_img_cv2 is not None:
             # Convert BGR to RGB
            original_img_cv2 = cv2.cvtColor(original_img_cv2, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = original_img_cv2.shape[:2]
        else:
            # Fallback if load fails (shouldn't happen if exists)
            # Use img_numpy (from preprocess) which might be resized or original depending on implementation
            # In preprocess_image we return img_resized (512x512 usually) as second arg if load_image does that.
            # But let's assume img_numpy here is what we have.
            original_img_cv2 = img_numpy
            orig_h, orig_w = original_img_cv2.shape[:2]

        # --- UPSCALE FOR VISUALIZATION IF TOO SMALL ---
        MIN_VIS_SIZE = 1200
        max_dim = max(orig_h, orig_w)
        if max_dim < MIN_VIS_SIZE:
            scale_factor = MIN_VIS_SIZE / float(max_dim)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            print(f"Upscaling image for visualization from {orig_w}x{orig_h} to {new_w}x{new_h}")
            original_img_cv2 = cv2.resize(original_img_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            orig_h, orig_w = new_h, new_w  # Update dims for mask resizing logic below

        # Resize masks back to original
        # binary_mask is 0/1, use Nearest
        binary_mask_orig = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # heatmap is float or uint8, use Linear
        heatmap_orig = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        images = {}
        images['original'] = original_img_cv2
        images['binary_mask'] = binary_mask_orig * 255  # uint8 for saving
        images['heatmap'] = (heatmap_orig * 255).astype(np.uint8)

        # Create Overlay on ORIGINAL image
        # Red overlay for crack
        overlay = original_img_cv2.copy()
        overlay[binary_mask_orig == 1] = [255, 0, 0]  # simple red paint
        # improved overlay: alpha blend
        alpha = 0.4
        overlay_vis = cv2.addWeighted(original_img_cv2, 1-alpha, overlay, alpha, 0)

        # Draw Annotations
        try:
            if geo_results:
                # --- 1. Smart Info Box ---
                # Find corner with least crack pixels in binary_mask_orig
                h, w = binary_mask_orig.shape
                mid_h, mid_w = h // 2, w // 2

                # Define corners: (y_slice, x_slice, corner_name, top_left_coords)
                corners = [
                    (slice(0, mid_h), slice(0, mid_w), "TL", (0, 0)),
                    (slice(0, mid_h), slice(mid_w, w), "TR", (mid_w, 0)),
                    (slice(mid_h, h), slice(0, mid_w), "BL", (0, mid_h)),
                    (slice(mid_h, h), slice(mid_w, w), "BR", (mid_w, mid_h))
                ]

                best_corner = min(corners, key=lambda c: np.sum(binary_mask_orig[c[0], c[1]]))
                corner_name = best_corner[2]

                # Info Box Content
                basic = geo_results.get('basic', {})
                width_stats = geo_results.get('width_stats', {})
                adv = geo_results.get('advanced', {})

                # Prepare Classification Info
                # class_name and class_conf are available from scope (line 282)
                cat_info = f"Kat: {class_name} ({class_conf*100:.1f}%)"

                lines = [
                    cat_info,
                    f"Dlugosc: {geo_results.get('length_pixels', 0):.1f} px",
                    f"Powierzchnia: {basic.get('area_pixels', 0):.0f} px2",
                    f"Szerokosc (Sr.): {width_stats.get('mean_width', 0):.2f} px",
                    f"Kretosc: {adv.get('tortuosity', 1.0):.3f}"
                ]

                # --- Dynamic Scaling Factor ---
                # Base reference: 1000px.
                # If image is small (e.g. 400px), scale should not be huge relative to it.
                # Let's dampen the scale for smaller images.
                dim_max = max(h, w)

                # Sqrt scaling gives better balance between small (400px) and large (4000px)
                # scale = sqrt(dim_max / 1000)
                # But simple linear with larger base also works.
                # Previous 800 was producing huge text on small images?
                # User said "on small images need smaller fonts".
                # If image is 400px, 400/800 = 0.5. Font 0.3. Box 340*0.5 = 170.
                # Maybe the issue is absolute size is okay but relative to image it blocks too much?
                # Let's reduce base scale significantly.

                viz_scale = dim_max / 1200.0
                viz_scale = max(viz_scale, 0.35) # Allow smaller minimum scale

                # Sizing config
                font_scale = 0.5 * viz_scale # Reduced base font
                thickness = max(1, int(1.5 * viz_scale))
                box_margin = int(10 * viz_scale) # Reduced margin
                line_height = int(25 * viz_scale) # Reduced line height
                box_w = int(320 * viz_scale) # Slightly narrower
                box_h = len(lines) * line_height + int(15 * viz_scale)

                # Determine Box Position based on corner
                if "TL" in corner_name:
                    bx, by = box_margin, box_margin
                elif "TR" in corner_name:
                    bx, by = w - box_w - box_margin, box_margin
                elif "BL" in corner_name:
                    bx, by = box_margin, h - box_h - box_margin
                else: # BR
                    bx, by = w - box_w - box_margin, h - box_h - box_margin

                # Draw Box with Opacity
                # Ensure coords are within bounds
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = min(box_w, w - bx)
                bh = min(box_h, h - by)

                if bw > 0 and bh > 0:
                    sub_img = overlay_vis[by:by+bh, bx:bx+bw]
                    rect = np.full(sub_img.shape, (20, 20, 20), dtype=np.uint8) 
                    res = cv2.addWeighted(sub_img, 0.4, rect, 0.6, 0)
                    overlay_vis[by:by+bh, bx:bx+bw] = res

                    # Draw Text
                    text_x = bx + int(10 * viz_scale)
                    text_y_start = by + int(25 * viz_scale)

                    for i, line in enumerate(lines):
                        y_pos = text_y_start + i * line_height
                        if y_pos < by + bh:
                            cv2.putText(overlay_vis, line, (text_x, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # --- 2. Max/Min Width Annotations ---
                scale_x = orig_w / OUTPUT_IMAGE_SIZE
                scale_y = orig_h / OUTPUT_IMAGE_SIZE

                def draw_labeled_point(img, loc, label, sublabel, color_bg, color_text=(255,255,255)):
                    if not loc: return
                    # Scale loc (y, x) -> (x, y)
                    x, y = int(loc[1] * scale_x), int(loc[0] * scale_y)

                    radius_outer = int(8 * viz_scale)
                    radius_inner = int(4 * viz_scale)
                    circle_thick = max(1, int(2 * viz_scale))

                    # 1. Draw Point with Frame
                    # Outer ring
                    cv2.circle(img, (x, y), radius_outer, (255, 255, 255), circle_thick)
                    # Inner dot
                    cv2.circle(img, (x, y), radius_inner, color_bg, -1)

                    # 2. Draw Label with opacity background
                    label_font_scale = 0.5 * viz_scale
                    label_thickness = max(1, int(1 * viz_scale))

                    text = f"{label}: {sublabel}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)

                    # Smart label positioning (avoid edges)
                    offset_x = int(15 * viz_scale)
                    offset_y = int(10 * viz_scale)

                    lx = x + offset_x
                    ly = y - offset_y
                    if lx + tw > w: lx = x - offset_x - tw
                    if ly - th < 0: ly = y + offset_y * 2

                    pad = int(5 * viz_scale)
                    # ROI for label bg
                    rx1, ry1 = lx - pad, ly - th - pad
                    rx2, ry2 = lx + tw + pad, ly + pad

                    # Clip to image bounds
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(w, rx2), min(h, ry2)

                    if rx2 > rx1 and ry2 > ry1:
                        roi = img[ry1:ry2, rx1:rx2]
                        # Use color_bg for box background but darker/muted? Or just gray?
                        # User asked for "frame with opacity"
                        bg_rect = np.full(roi.shape, (50, 50, 50), dtype=np.uint8)
                        blended = cv2.addWeighted(roi, 0.3, bg_rect, 0.7, 0)
                        img[ry1:ry2, rx1:rx2] = blended

                        cv2.putText(img, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, color_text, label_thickness, cv2.LINE_AA)


                max_loc = width_stats.get('max_width_loc')
                min_loc = width_stats.get('min_width_loc')
                max_val = width_stats.get('max_width', 0)
                min_val = width_stats.get('min_width', 0)

                # Draw Max (Cyan-ish)
                draw_labeled_point(overlay_vis, max_loc, "Maks", f"{max_val:.1f}px", (0, 255, 255)) 
                draw_labeled_point(overlay_vis, min_loc, "Min", f"{min_val:.1f}px", (255, 255, 0)) 

        except Exception as e:
            print(f"Error drawing annotations: {e}")
            import traceback
            traceback.print_exc()

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
