import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool, cpu_count
from functools import partial


# msk_path = r"../../../../datasets/dataset_classification/test_lab"
# img_path = r"../../../../datasets/dataset_classification/test_img"


msk_path = r"../../../../datasets/dataset_classification/train_lab"
img_path = r"../../../../datasets/dataset_classification/train_img"


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


def analyze_dir(mask_dir, show=True):
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
    print(
        f"  Crack % - min: {np.min(crack_percentages):.3f}%, max: {np.max(crack_percentages):.3f}%, avg: {np.mean(crack_percentages):.3f}%")

    if show:
        # Ogranicz zakres do 40% max i pomiń zerowe
        max_percent = 40.0
        min_percent = 0.01  # Pomiń wartości < 0.01%

        filtered_percentages = [
            p for p in crack_percentages if min_percent <= p <= max_percent]
        zero_count = len([p for p in crack_percentages if p < min_percent])
        outliers = len(crack_percentages) - \
            len(filtered_percentages) - zero_count

        # Stwórz figure z jednym większym histogramem
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Histogram: Procent powierzchni - szczegółowe biny
        percent_bins = np.concatenate([
            np.linspace(min_percent, 0.5, 21),    # 20 binów dla 0.01-0.5%
            np.linspace(0.5, 1.0, 21),            # 20 binów dla 0.5-1%
            np.linspace(1, 2, 21),                # 20 binów dla 1-2%
            np.linspace(2, 5, 31),                # 30 binów dla 2-5%
            np.linspace(5, 10, 26),               # 25 binów dla 5-10%
            np.linspace(10, 20, 11),              # 10 binów dla 10-20%
            np.linspace(20, max_percent, 7)       # 6 binów dla 20-40%
        ])

        ax.hist(filtered_percentages, bins=percent_bins, color='coral',
                edgecolor='black', alpha=0.75, linewidth=0.8)
        ax.set_xlabel('Procent powierzchni pęknięć (%)',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('Liczba masek', fontsize=12, fontweight='bold')
        ax.set_title(f'Rozkład: Procent powierzchni pęknięć (max {max_percent}%, pominięto zerowe)',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, max_percent)
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
        ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)

        # Linia średniej
        mean_val = np.mean(filtered_percentages)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                   label=f'Średnia: {mean_val:.2f}%', zorder=5)

        # Mediana
        median_val = np.median(filtered_percentages)
        ax.axvline(median_val, color='blue', linestyle=':', linewidth=2.5,
                   label=f'Mediana: {median_val:.2f}%', zorder=5)

        # Info o pominiętych wartościach
        info_text = []
        if zero_count > 0:
            info_text.append(
                f'Zerowe (<{min_percent}%): {zero_count} ({zero_count/len(crack_percentages)*100:.1f}%)')
        if outliers > 0:
            info_text.append(
                f'>{max_percent}%: {outliers} ({outliers/len(crack_percentages)*100:.1f}%)')

        if info_text:
            ax.text(0.98, 0.98, '\n'.join(info_text),
                    transform=ax.transAxes, ha='right', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'))

        # Statystyki w narożniku
        stats_text = f'n = {len(filtered_percentages)}\nMin: {np.min(filtered_percentages):.2f}%\nMax: {np.max(filtered_percentages):.2f}%\nStd: {np.std(filtered_percentages):.2f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='black'))

        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

        plt.tight_layout()
        plt.show()

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
    Oblicza progi kategoryzacji.

    method='original': Progi oryginalne (0-5%, 5-10%, 10-20%, >20%)
    method='balanced': Progi zbalansowane aby max różnica była 15%
    """
    print(f"Obliczam progi kategoryzacji (metoda: {method})...")
    percentages = []

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for mask_file in tqdm(mask_files, desc="Analiza obrazów", leave=False):
        try:
            mask_path = os.path.join(mask_dir, mask_file)
            _, _, percentage = image_analyze(mask_path, show=False)
            percentages.append(percentage)
        except:
            continue

    percentages = np.array(percentages)

    if method == 'original':
        # Progi oryginalne
        thresholds = [5.0, 10.0, 20.0]
        print(f"\n✓ Progi oryginalne:")
        print(f"  0-5%: włosowe")
        print(f"  5-10%: male")
        print(f"  10-20%: srednie")
        print(f"  >20%: duze")
    else:  # balanced
        # Zbalansowane: max różnica 15%
        # Szukamy progów takich że każda grupa będzie maksymalnie różna o 15%
        p20 = np.percentile(percentages, 20)
        p45 = np.percentile(percentages, 45)
        p70 = np.percentile(percentages, 70)

        thresholds = [p20, p45, p70]
        print(f"\n✓ Progi zbalansowane (+15% max różnica):")
        print(f"  {p20:.2f}%: granica włosowe-male")
        print(f"  {p45:.2f}%: granica male-srednie")
        print(f"  {p70:.2f}%: granica srednie-duze")

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


def create_categorized_dataset(image_dir, mask_dir, output_base_dir, thresholds=None, method='original'):
    """
    Kopiuje zdjęcia i maski do folderów według kategorii

    Struktura wyjściowa:
    output_base_dir/
        0_brak/
            images/
            masks/
        1_wlosowe/
            images/
            masks/
        2_male/
            images/
            masks/
        3_srednie/
            images/
            masks/
        4_duze/
            images/
            masks/
    """
    # Jeśli nie podano progów, oblicz je automatycznie
    if thresholds is None:
        thresholds = compute_optimal_thresholds(mask_dir, method=method)

    category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]

    os.makedirs(output_base_dir, exist_ok=True)

    for cat_name in category_names:
        os.makedirs(os.path.join(output_base_dir,
                    cat_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir,
                    cat_name, "masks"), exist_ok=True)

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Kategoryzowanie {len(mask_files)} plików...")

    category_counts = {i: 0 for i in range(4)}
    skipped = 0

    for mask_file in tqdm(mask_files, desc="Kopiowanie plików"):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, mask_file)

        if not os.path.exists(image_path):
            skipped += 1
            continue

        try:
            _, _, percentage = image_analyze(mask_path, show=False)
            cat_id, _ = categorize_crack(percentage, thresholds=thresholds)

            # Sprawdź czy cat_id jest w zasięgu
            if cat_id < 0 or cat_id >= len(category_names):
                skipped += 1
                continue

            dest_img = os.path.join(
                output_base_dir, category_names[cat_id], "images", mask_file)
            dest_mask = os.path.join(
                output_base_dir, category_names[cat_id], "masks", mask_file)

            shutil.copy2(image_path, dest_img)
            shutil.copy2(mask_path, dest_mask)

            category_counts[cat_id] += 1

        except Exception as e:
            skipped += 1
            continue

    print("\n" + "="*70)
    print("PODSUMOWANIE KATEGORYZACJI")
    print("="*70)
    print(f"Output directory: {output_base_dir}\n")
    print(f"Używane progi: {[f'{t:.2f}%' for t in thresholds]}\n")

    for cat_id, cat_name in enumerate(category_names):
        count = category_counts[cat_id]
        print(f"{cat_name:15} : {count:5} plików ({count/len(mask_files)*100:.1f}%)")

    print(f"\nPominięto: {skipped} plików")
    print(f"Sukces: {sum(category_counts.values())} plików")
    print("="*70)

    return category_counts


def plot_category_histogram(category_counts, category_names, filename=None, title=None):
    """Tworzy histogram rozkładu plików po kategoriach"""
    import matplotlib.pyplot as plt

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

    # Dodaj wartości na słupkach
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
    print(f"\n✓ Histogram zapisany: {save_filename}")
    plt.close()


def augment_single_image(args):
    """
    Augmentuje jeden obraz - dla multiprocessing.
    Zwraca True jeśli sukces, False jeśli błąd.
    """
    from PIL import ImageOps, ImageEnhance

    img_folder, mask_folder, original_name, aug_variant_idx = args

    try:
        original_img_path = os.path.join(img_folder, original_name)
        original_mask_path = os.path.join(mask_folder, original_name)

        # Wczytaj oryginalny obraz i maskę
        image = Image.open(original_img_path).convert('RGB')
        mask = Image.open(original_mask_path).convert('L')

        # Kombinuj różne augmentacje losowo
        if random.random() > 0.3:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() > 0.3:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle, expand=False)
            mask = mask.rotate(angle, expand=False)

        if random.random() > 0.6:
            factor = random.uniform(0.7, 1.4)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        if random.random() > 0.6:
            factor = random.uniform(0.7, 1.4)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        if random.random() > 0.6:
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)

        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(factor)

        # Zapisz augmentowaną wersję
        base_name = original_name.split('.')[0]
        aug_name = f"{base_name}_aug_{aug_variant_idx:05d}.png"
        image.save(os.path.join(img_folder, aug_name))
        mask.save(os.path.join(mask_folder, aug_name))

        return True
    except Exception as e:
        return False


def augment_unbalanced_dataset(dataset_dir, target_count=None):
    """
    Augmentuje obrazy - STRATEGIA: wyrównaj WSZYSTKIE KATEGORIE do tej samej liczby (równo po 25%)

    ВАЖНО: Każdy augmentowany obraz = DOKŁADNIE JEDEN nowy plik (nie wiele wariantów na jeden plik!)
    """
    from PIL import ImageOps, ImageEnhance
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    category_dirs = sorted([d for d in os.listdir(dataset_dir)
                            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith(('1_', '2_', '3_', '4_'))])

    if not category_dirs:
        print("❌ Nie znaleziono katalogów kategorii!")
        return

    # Policz pliki w każdej kategorii
    category_counts = {}
    for cat_dir in category_dirs:
        img_dir = os.path.join(dataset_dir, cat_dir, 'images')
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
            category_counts[cat_dir] = count

    # Docelowa liczba = max kategoria (wyrównaj wszystkie do największej)
    max_count = max(category_counts.values()) if category_counts else 1000
    target_count = max_count

    print("\n" + "="*70)
    print("AUGMENTACJA DANYCH - WYRÓWNANIE DO RÓWNYCH PORCJI")
    print("="*70)
    print(f"CPU cores: {cpu_count()}")
    print(f"Docelowa liczba (równo dla każdej): {target_count}\n")
    print("Docelowe liczby plików:")
    for cat_dir in category_dirs:
        current = category_counts.get(cat_dir, 0)
        print(f"  {cat_dir:15} : {current:5} -> {target_count:5}")
    print()

    num_processes = cpu_count()
    all_tasks = []
    global_aug_idx = 0

    # Przygotuj listę WSZYSTKICH tasków augmentacji (każdy task = 1 nowy plik)
    for cat_dir in category_dirs:
        img_folder = os.path.join(dataset_dir, cat_dir, 'images')
        mask_folder = os.path.join(dataset_dir, cat_dir, 'masks')

        if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
            print(f"⚠️  Katalog {cat_dir} nie istnieje, pomijam...")
            continue

        current_count = category_counts.get(cat_dir, 0)
        needed = target_count - current_count

        if needed <= 0:
            print(f"✓ {cat_dir:15} {current_count:5} plików (OK)")
            continue

        print(
            f"⧗ {cat_dir:15} {current_count:5} -> {target_count:5} plików (augmentacja: {needed})")

        # Zbierz istniejące pliki
        image_files = sorted([f for f in os.listdir(
            img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Każdy task = jeden augmentowany plik
        # Rozprowadź augmentacje równomiernie między obrazkami
        for needed_idx in range(needed):
            img_idx = needed_idx % len(image_files)
            original_name = image_files[img_idx]

            task = (img_folder, mask_folder, original_name, global_aug_idx)
            all_tasks.append(task)
            global_aug_idx += 1

    print()

    # Augmentuj równolegle - KAŻDY PROCESS TWORZY DOKŁADNIE JEDEN PLIK
    if all_tasks:
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(augment_single_image, all_tasks),
                total=len(all_tasks),
                desc="Augmentacja",
                leave=True
            ))

    # Podsumowanie
    print("\n" + "="*70)
    print("PODSUMOWANIE AUGMENTACJI")
    print("="*70)

    for cat_dir in category_dirs:
        img_folder = os.path.join(dataset_dir, cat_dir, 'images')
        if os.path.exists(img_folder):
            new_count = len([f for f in os.listdir(img_folder)
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            original = category_counts.get(cat_dir, 0)
            aug_added = new_count - original
            print(
                f"{cat_dir:15} : {original:5} -> {new_count:5} (dodano {aug_added:5})")

    print("="*70)


def main():
    OUTPUT_DIR_ORIGINAL = r"../../../../datasets/multi_class/train_original/"
    OUTPUT_DIR_BALANCED = r"../../../../datasets/multi_class/train_balanced/"
    OUTPUT_DIR_AUGMENTED = r"../../../../datasets/multi_class/train_augmented/"

    category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]

    print("\n" + "="*70)
    print("KROK 1: KATEGORYZACJA Z PROGAMI ORYGINALNYMI")
    print("="*70)
    # Kategoryzuj z progami oryginalnymi
    category_counts_original = create_categorized_dataset(
        img_path, msk_path, OUTPUT_DIR_ORIGINAL, method='original')

    # Histogram 1
    plot_category_histogram(category_counts_original, category_names,
                            filename='01_histogram_original_progi_train.png',
                            title='Histogram 1: Progi Oryginalne (0-5%, 5-10%, 10-20%, >20%)')

    print("\n" + "="*70)
    print("KROK 2: KATEGORYZACJA Z PROGAMI ZBALANSOWANYMI")
    print("="*70)
    # Kategoryzuj z progami zbalansowanymi
    category_counts_balanced = create_categorized_dataset(
        img_path, msk_path, OUTPUT_DIR_BALANCED, method='balanced')

    # Histogram 2
    plot_category_histogram(category_counts_balanced, category_names,
                            filename='02_histogram_balanced_progi_train.png',
                            title='Histogram 2: Progi Zbalansowane (max różnica 15%)')

    print("\n" + "="*70)
    print("KROK 3: AUGMENTACJA DANYCH")
    print("="*70)
    # Augmentuj dane zbalansowane
    augment_unbalanced_dataset(OUTPUT_DIR_BALANCED)

    # Policz pliki po augmentacji
    category_counts_augmented = {}
    for cat_name in category_names:
        img_dir = os.path.join(OUTPUT_DIR_BALANCED, cat_name, 'images')
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
            cat_id = category_names.index(cat_name)
            category_counts_augmented[cat_id] = count

    # Histogram 3
    plot_category_histogram(category_counts_augmented, category_names,
                            filename='03_histogram_po_augmentacji_train.png',
                            title='Histogram 3: Po Augmentacji (wyrównane)')


if __name__ == "__main__":
    main()
