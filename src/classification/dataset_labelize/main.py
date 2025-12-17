import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool, cpu_count
from functools import partial


# msk_path = r"/content/datasets/multi/train_lab"
# img_path = r"/content/datasets/multi/train_img"

# msk_path_test = r"/content/datasets/multi/test_lab"
# img_path_test = r"/content/datasets/multi/test_img"

# msk_path_train = r"../../../../datasets/dataset_classification/train_lab"
# msk_path_test = r"../../../../datasets/dataset_classification/test_lab"


msk_path_train = "/content/datasets/multi/train_lab"
msk_path_test = "/content/datasets/multi/test_lab"


def image_analyze(mask_path, show=False) -> tuple:
    with Image.open(mask_path) as img:
        mask = np.array(img)

        total_px = mask.size

        cracked = np.sum(mask > 127)
        crack_percentage = (cracked/total_px) * 100

        # print(f"total:{total_px}, cracked:{cracked}, %:{crack_percentage:.3f}")
        if show:
            plt.imshow(mask)
            plt.show()
        return (total_px, cracked, crack_percentage)


def analyze_dir_old(mask_dir, show=True):
    """Analizuje wszystkie maski w katalogu i tworzy histogramy"""

    if not os.path.isdir(mask_dir):
        print(f"ERROR: {mask_dir} is not a directory")
        return None

    # Zbierz dane ze wszystkich masek
    crack_pixels = []
    crack_percentages = []

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(mask_files) == 0:
        print(f"No mask files found in {mask_dir}")
        return None

    print(f"Analyzing {len(mask_files)} masks from {mask_dir}...")

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            total_px, cracked, percentage = image_analyze(
                mask_path, show=False)
            crack_pixels.append(cracked)
            crack_percentages.append(percentage)
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
            continue

    # Statystyki
    print(f"\nStatistics:")
    print(f"  Total masks analyzed: {len(crack_pixels)}")
    print(
        f"  Crack pixels - min: {np.min(crack_pixels)}, max: {np.max(crack_pixels)}, avg: {np.mean(crack_pixels):.1f}")
    print("0_brak",
          f"  Crack % - min: {np.min(crack_percentages):.3f}%, max: {np.max(crack_percentages):.3f}%, avg: {np.mean(crack_percentages):.3f}%")

    if show:
        # Ogranicz zakres do 50% max
        max_percent = 50.0
        filtered_percentages = [
            p for p in crack_percentages if p <= max_percent]
        outliers = len(crack_percentages) - len(filtered_percentages)

        # Stwórz figure z dwoma histogramami
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram 1: Piksele pęknięć - więcej binów dla małych wartości
        # Niestandardowe biny: gęste dla małych wartości, rzadsze dla dużych
        max_pixels = np.max(crack_pixels)
        pixel_bins = np.concatenate([
            # 20 binów dla pierwszych 10%
            np.linspace(0, max_pixels * 0.1, 20),
            np.linspace(max_pixels * 0.1, max_pixels *
                        0.5, 10),  # 10 binów dla 10-50%
            np.linspace(max_pixels * 0.5, max_pixels, 5)  # 5 binów dla reszty
        ])

        ax1.hist(crack_pixels, bins=pixel_bins,
                 color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Liczba pikseli pęknięć')
        ax1.set_ylabel('Liczba masek')
        ax1.set_title('Rozkład: Piksele pęknięć')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axvline(np.mean(crack_pixels), color='red', linestyle='--',
                    linewidth=2, label=f'Średnia: {np.mean(crack_pixels):.0f}')
        ax1.legend()

        # Histogram 2: Procent powierzchni - niestandardowe biny
        # Gęste biny dla 0-5%, 5-10%, potem rzadsze
        percent_bins = np.concatenate([
            np.linspace(0, 0.5, 11),      # 10 binów dla 0-0.5% (bardzo małe)
            np.linspace(0.5, 1.0, 11),    # 10 binów dla 0.5-1%
            np.linspace(1, 2, 11),        # 10 binów dla 1-2%
            np.linspace(2, 5, 11),        # 10 binów dla 2-5%
            np.linspace(5, 10, 11),       # 10 binów dla 5-10%
            np.linspace(10, 20, 6),       # 5 binów dla 10-20%
            np.linspace(20, max_percent, 6)  # 5 binów dla 20-50%
        ])

        ax2.hist(filtered_percentages, bins=percent_bins,
                 color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Procent powierzchni pęknięć (%)')
        ax2.set_ylabel('Liczba masek')
        ax2.set_title(
            f'Rozkład: Procent powierzchni pęknięć (max {max_percent}%)')
        ax2.set_xlim(0, max_percent)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axvline(np.mean(filtered_percentages), color='red', linestyle='--', linewidth=2,
                    label=f'Średnia: {np.mean(filtered_percentages):.2f}%')

        if outliers > 0:
            ax2.text(0.98, 0.98, f'Wartości >{max_percent}%: {outliers}',
                     transform=ax2.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.legend()

        plt.tight_layout()
        plt.show()

    return {
        'crack_pixels': crack_pixels,
        'crack_percentages': crack_percentages,
        'num_masks': len(crack_pixels)
    }


def analyze_dir(mask_dir, show=False):
    """Analizuje wszystkie maski w katalogu (bez pokazywania)"""
    if not os.path.isdir(mask_dir):
        print(f"ERROR: {mask_dir} is not a directory")
        return None

    crack_pixels = []
    crack_percentages = []

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(mask_files) == 0:
        print(f"No mask files found in {mask_dir}")
        return None

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            total_px, cracked, percentage = image_analyze(
                mask_path, show=False)
            crack_pixels.append(cracked)
            crack_percentages.append(percentage)
        except:
            continue

    return {
        'crack_pixels': crack_pixels,
        'crack_percentages': crack_percentages,
        'num_masks': len(crack_pixels)
    }


def categorize_crack(percentage, thresholds=None):
    """
    Kategoryzuje pęknięcie na podstawie procentu powierzchni

    Kategorie:
    1: Włosowe 
    2: Małe 
    3: Średnie 
    4: Duże 

    Progi są dynamiczne - dzielą dane na 4 równe grupy na podstawie percentyli
    """
    # Domyślne progi będą nadpisane przez compute_optimal_thresholds()
    if thresholds is None:
        thresholds = [1.0, 3.0, 8.0]

    if percentage < thresholds[0]:
        return 0, "wlosowe"
    elif percentage < thresholds[1]:
        return 1, "male"
    elif percentage < thresholds[2]:
        return 2, "srednie"
    else:
        return 3, "duze"


def compute_optimal_thresholds(mask_dir, method='original'):
    """
    Oblicza progi kategoryzacji szybko (bez verbose)
    """
    percentages = []
    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for mask_file in mask_files:
        try:
            mask_path = os.path.join(mask_dir, mask_file)
            _, _, percentage = image_analyze(mask_path, show=False)
            percentages.append(percentage)
        except:
            continue

    percentages = np.array(percentages)

    if method == 'original':
        thresholds = [5.0, 10.0, 20.0]
    else:
        p20 = np.percentile(percentages, 20)
        p45 = np.percentile(percentages, 45)
        p70 = np.percentile(percentages, 70)
        thresholds = [p20, p45, p70]

    return thresholds


def show_category_samples(image_dir, mask_dir, num_samples=2):
    """
    Analizuje maski, kategoryzuje je i pokazuje losowe przykłady z każdej kategorii

    Args:
        image_dir: Katalog ze zdjęciami
        mask_dir: Katalog z maskami
        num_samples: Liczba przykładów na kategorię (domyślnie 2)
    """
    # Zbierz wszystkie maski i ich kategorie
    category_files = {i: [] for i in range(5)}

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Analizowanie {len(mask_files)} masek...")

    for mask_file in tqdm(mask_files, desc="Kategoryzowanie"):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, mask_file)

        if not os.path.exists(image_path):
            continue

        try:
            # Analizuj maskę
            mask = np.array(Image.open(mask_path).convert('L'))
            total_pixels = mask.size
            crack_pixels = np.sum(mask > 127)
            percentage = (crack_pixels / total_pixels) * 100

            # Kategoryzuj
            cat_id, cat_name = categorize_crack(percentage)

            # Zapisz do odpowiedniej kategorii
            category_files[cat_id].append({
                'filename': mask_file,
                'percentage': percentage,
                'image_path': image_path,
                'mask_path': mask_path
            })

        except Exception as e:
            print(f"Błąd przy {mask_file}: {e}")
            continue

    # Wyświetl statystyki
    print("\n" + "="*70)
    print("STATYSTYKI KATEGORII")
    print("="*70)

    category_names = ["Brak", "Włosowe", "Małe", "Średnie", "Duże"]
    ranges = ["0%", "0.01-2%", "2-5%", "5-12%", ">12%"]

    for cat_id in range(5):
        count = len(category_files[cat_id])
        if count > 0:
            percentages = [item['percentage']
                           for item in category_files[cat_id]]
            avg = np.mean(percentages)
            min_val = np.min(percentages)
            max_val = np.max(percentages)

            print(f"\n{cat_id}. {category_names[cat_id]} ({ranges[cat_id]})")
            print(
                f"   Liczba masek: {count} ({count/len(mask_files)*100:.1f}%)")
            print(f"   Średni %: {avg:.2f}%")
            print(f"   Zakres: {min_val:.2f}% - {max_val:.2f}%")
        else:
            print(f"\n{cat_id}. {category_names[cat_id]} ({ranges[cat_id]})")
            print(f"   Liczba masek: 0")

    print("\n" + "="*70)

    # Przygotuj figure do wizualizacji    category_files = show_category_samples(IMAGE_DIR, MASK_DIR, num_samples=2)
    fig, axes = plt.subplots(5, num_samples * 2, figsize=(num_samples * 6, 15))
    fig.suptitle('Przykłady z każdej kategorii (Obraz | Maska)',
                 fontsize=16, fontweight='bold', y=0.995)

    for cat_id in range(5):
        files = category_files[cat_id]

        if len(files) == 0:
            # Brak plików w kategorii
            for sample_idx in range(num_samples * 2):
                ax = axes[cat_id, sample_idx]
                ax.axis('off')
                if sample_idx == 0:
                    ax.text(0.5, 0.5, 'Brak danych', ha='center',
                            va='center', fontsize=12)
            continue

        # Wybierz losowe próbki
        selected = random.sample(files, min(num_samples, len(files)))

        for sample_idx, item in enumerate(selected):
            # Wczytaj obrazy
            img = Image.open(item['image_path'])
            mask = Image.open(item['mask_path'])

            # Pokaż obraz
            ax_img = axes[cat_id, sample_idx * 2]
            ax_img.imshow(img)
            ax_img.axis('off')
            if sample_idx == 0:
                ax_img.set_title(f'{category_names[cat_id]} ({ranges[cat_id]})',
                                 fontsize=11, fontweight='bold', pad=10)

            # Pokaż maskę
            ax_mask = axes[cat_id, sample_idx * 2 + 1]
            ax_mask.imshow(mask, cmap='gray')
            ax_mask.axis('off')
            ax_mask.set_title(
                f'{item["percentage"]:.2f}%', fontsize=10, color='red')

        # Wypełnij puste miejsca jeśli mniej niż num_samples
        for sample_idx in range(len(selected), num_samples):
            axes[cat_id, sample_idx * 2].axis('off')
            axes[cat_id, sample_idx * 2 + 1].axis('off')

    plt.tight_layout()
    plt.show()

    return category_files


def create_categorized_dataset(mask_dir, output_base_dir, thresholds=None, method='original'):
    """
    Kopiuje TYLKO MASKI do folderów według kategorii
    """
    if thresholds is None:
        thresholds = compute_optimal_thresholds(mask_dir, method=method)

    category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]

    # Stwórz strukturę folderów - TYLKO MASKS (bez images!)
    for cat_name in category_names:
        os.makedirs(os.path.join(output_base_dir, cat_name), exist_ok=True)

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    category_counts = {i: 0 for i in range(4)}
    skipped = 0

    # Pętla z progress bar
    for mask_file in tqdm(mask_files, desc="Kategoryzowanie"):
        mask_path = os.path.join(mask_dir, mask_file)

        try:
            _, _, percentage = image_analyze(mask_path, show=False)
            cat_id, _ = categorize_crack(percentage, thresholds=thresholds)

            if cat_id < 0 or cat_id >= len(category_names):
                skipped += 1
                continue

            dest_mask = os.path.join(
                output_base_dir, category_names[cat_id], mask_file)
            shutil.copy2(mask_path, dest_mask)
            category_counts[cat_id] += 1

        except:
            skipped += 1
            continue

    print(f"✓ Kategoryzacja: {sum(category_counts.values())} masek")
    return category_counts


def plot_category_histogram(category_counts, category_names, filename=None, title=None):
    """Zapisuje histogram bez pokazywania"""
    categories = list(range(len(category_names)))
    counts = [category_counts[i] for i in categories]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(category_names)), counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Kategoria pęknięć', fontsize=12, fontweight='bold')
    ax.set_ylabel('Liczba plików', fontsize=12, fontweight='bold')
    ax.set_title(title if title else 'Rozkład plików po kategoriach pęknięć',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(category_names)))
    ax.set_xticklabels(category_names, fontsize=11)

    # Dodaj wartości
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        percentage = (count / sum(counts)) * 100 if sum(counts) > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)

    plt.tight_layout()
    save_filename = filename if filename else 'category_distribution.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f"✓ Histogram: {save_filename}")


def augment_single_image(args):
    """
    Augmentuje TYLKO maskę z DUŻĄ augmentacją (+ random crop).
    Zwraca True jeśli sukces, False jeśli błąd.
    """
    from PIL import ImageOps, ImageEnhance

    mask_folder, original_name, aug_variant_idx = args

    try:
        original_mask_path = os.path.join(mask_folder, original_name)
        mask = Image.open(original_mask_path).convert('L')
        mask_width, mask_height = mask.size

        # Random crop (80-100% oryginalnego rozmiaru)
        if random.random() > 0.3:
            crop_factor = random.uniform(0.8, 1.0)
            crop_width = int(mask_width * crop_factor)
            crop_height = int(mask_height * crop_factor)
            left = random.randint(0, mask_width - crop_width)
            top = random.randint(0, mask_height - crop_height)
            mask = mask.crop((left, top, left + crop_width, top + crop_height))
            mask = mask.resize((mask_width, mask_height),
                               Image.Resampling.LANCZOS)

        # Kombinuj różne augmentacje losowo
        if random.random() > 0.2:  # 80% szansa
            mask = ImageOps.mirror(mask)

        if random.random() > 0.2:
            mask = ImageOps.flip(mask)

        if random.random() > 0.4:  # 60% szansa
            angle = random.choice([90, 180, 270])
            mask = mask.rotate(angle, expand=False)

        if random.random() > 0.5:  # 50% szansa - losowe przesunięcie
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            if offset_x != 0 or offset_y != 0:
                mask = ImageOps.expand(mask, (max(0, offset_x), max(0, offset_y),
                                              max(0, -offset_x), max(0, -offset_y)), fill=0)
                mask = mask.crop((abs(min(0, offset_x)), abs(min(0, offset_y)),
                                 mask.width + min(0, offset_x), mask.height + min(0, offset_y)))
                mask = mask.resize((mask_width, mask_height),
                                   Image.Resampling.LANCZOS)

        # Zapisz augmentowaną maskę
        base_name = original_name.split('.')[0]
        aug_name = f"{base_name}_aug_{aug_variant_idx:05d}.png"
        mask.save(os.path.join(mask_folder, aug_name))

        return True
    except Exception as e:
        return False


def augment_unbalanced_dataset(dataset_dir, augmentation_factor=None):
    """
    Augmentuje MASKI - wyrównuje kategorie do maxcount
    Jeśli augmentation_factor podany, multiplies by that factor (dla dodatkowej augmentacji)
    """
    from PIL import ImageOps, ImageEnhance
    from multiprocessing import Pool, cpu_count

    category_dirs = sorted([d for d in os.listdir(dataset_dir)
                            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith(('1_', '2_', '3_', '4_'))])

    if not category_dirs:
        return

    # Policz maski w każdej kategorii
    category_counts = {}
    for cat_dir in category_dirs:
        cat_path = os.path.join(dataset_dir, cat_dir)
        if os.path.exists(cat_path):
            count = len([f for f in os.listdir(cat_path)
                        if f.endswith(('.png', '.jpg', '.jpeg')) and '_aug_' not in f])
            category_counts[cat_dir] = count

    # Docelowa liczba = max kategoria (wyrównaj do największej)
    max_count = max(category_counts.values()) if category_counts else 1000
    target_count = max_count

    if augmentation_factor and augmentation_factor > 1:
        target_count = max_count * augmentation_factor

    num_processes = cpu_count()
    all_tasks = []
    global_aug_idx = 0

    # Przygotuj taskami - tyle augmentacji ile potrzeba
    for cat_dir in category_dirs:
        cat_path = os.path.join(dataset_dir, cat_dir)

        if not os.path.exists(cat_path):
            continue

        current_count = category_counts.get(cat_dir, 0)
        needed = target_count - current_count

        if needed <= 0:
            continue

        mask_files = sorted([f for f in os.listdir(cat_path)
                             if f.endswith(('.png', '.jpg', '.jpeg')) and '_aug_' not in f])

        # Rozprowadź augmentacje równomiernie między istniejące obrazy
        for aug_idx in range(needed):
            mask_idx = aug_idx % len(mask_files)
            original_name = mask_files[mask_idx]
            task = (cat_path, original_name, global_aug_idx)
            all_tasks.append(task)
            global_aug_idx += 1

    # Augmentuj z progress bar
    if all_tasks:
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(augment_single_image, all_tasks),
                      total=len(all_tasks), desc="Augmentacja"))

    print(f"✓ Augmentacja: {len(all_tasks)} masek")


def main():
    OUTPUT_DIR_FINAL_TRAIN = r"/content/datasets/classification/train_img"
    OUTPUT_DIR_FINAL_TEST = r"/content/datasets/classification/test_img"
    # OUTPUT_DIR_FINAL_TRAIN = r"/home/krzeslaav/Projects/datasets/multi/train_img"
    # OUTPUT_DIR_FINAL_TEST = r"/home/krzeslaav/Projects/datasets/multi/test_img"

    print("🔄 Kategoryzacja i augmentacja danych treningowych...")
    create_categorized_dataset(
        msk_path_train, OUTPUT_DIR_FINAL_TRAIN, method='balanced')
    augment_unbalanced_dataset(OUTPUT_DIR_FINAL_TRAIN)
    print("✅ Dane treningowe gotowe: " + OUTPUT_DIR_FINAL_TRAIN)

    print("🔄 Kategoryzacja danych testowych (BEZ augmentacji)...")
    create_categorized_dataset(
        msk_path_test, OUTPUT_DIR_FINAL_TEST, method='balanced')
    print("✅ Dane testowe gotowe: " + OUTPUT_DIR_FINAL_TEST)


if __name__ == "__main__":
    main()
