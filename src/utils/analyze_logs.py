
import os
import sys
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def extract_scalar_events(log_dir):
    """
    Extracts all scalar events from a TensorBoard log directory.
    """
    # Find event file
    event_file = None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_file = os.path.join(root, file)
                break
        if event_file:
            break

    if not event_file:
        return None

    print(f"Loading events from: {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()['scalars']

    data = {}

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # We assume steps are consistent across tags for the same run usually,
        # but to be safe we store as dict or verify alignment.
        # For simplicity, we just take the values and steps of the first tag to define index if possible,
        # or better: Dataframe per tag.

        data[tag] = pd.Series(values, index=steps)

    df = pd.DataFrame(data)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <path_to_log_dir>")
        # Default check for demo
        log_dir = "/home/krzeslaav/Projects/bachlor/logi_unet/"
        print(f"No path provided, scanning default: {log_dir}")
    else:
        log_dir = sys.argv[1]

    # Find latest run in the dir if it's a parent dir
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(
        log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    subdirs.sort(key=os.path.getmtime, reverse=True)

    if not subdirs:
        print("No subdirectories found.")
        return

    latest_run = subdirs[0]
    print(f"Analyzing latest run: {latest_run}")

    df = extract_scalar_events(latest_run)

    if df is not None and not df.empty:
        print("\nLast 5 Epochs Metrics:")
        print(df.tail(5))

        # Check for specific metrics
        if 'IoU/val' in df.columns:
            print(f"\nBest Val IoU: {df['IoU/val'].max():.4f}")

        if 'Loss/train' in df.columns and 'Loss/val' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['Loss/train'], label='Train Loss')
            plt.plot(df.index, df['Loss/val'], label='Val Loss')
            plt.title("Loss over Epochs")
            plt.legend()
            plt.show()  # This might not show in headless, but code is there.
    else:
        print("No scalar data found or empty log.")


if __name__ == "__main__":
    main()
