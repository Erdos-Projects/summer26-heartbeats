import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")

DATA_DIR = Path(__file__).resolve().parent.parent / "PAMAP2_Dataset" / "Protocol"
SAMPLE_RATE = 100

# PAMAP2 activity labels
ACTIVITIES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

NAME_TO_ID = {name: activity_id for activity_id, name in ACTIVITIES.items()}


def imu_cols(prefix):
    # PAMAP2 has the same IMU block repeated for hand/chest/ankle.
    names = [f"{prefix}_temp"]
    names += [f"{prefix}_acc16_{a}" for a in "xyz"]
    names += [f"{prefix}_acc6_{a}" for a in "xyz"]
    names += [f"{prefix}_gyro_{a}" for a in "xyz"]
    names += [f"{prefix}_mag_{a}" for a in "xyz"]
    names += [f"{prefix}_orient_{i}" for i in range(4)]
    return names


COLUMNS = ["timestamp", "activity_id", "heart_rate"]
for location in ["hand", "chest", "ankle"]:
    COLUMNS += imu_cols(location)


def load_subject(subject):
    path = f"{DATA_DIR}/subject{subject}.dat"
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS, na_values="NaN")
    return df


def get_segment(df, activity, start, seconds):
    if activity not in NAME_TO_ID:
        print(f"Unknown activity '{activity}'.")
        return None

    activity_rows = df[df["activity_id"] == NAME_TO_ID[activity]]

    if len(activity_rows) == 0:
        print(f"This subject did not perform {activity}.")
        print("Activities present:", sorted(df["activity_id"].unique()))
        return None

    # start is measured in seconds from the beginning of this activity.
    start_row = start * SAMPLE_RATE
    end_row = start_row + seconds * SAMPLE_RATE
    segment = activity_rows.iloc[start_row:end_row]

    if len(segment) == 0:
        print(f"The window (start={start}s) is past the end of the {activity} data.")
        return None

    return segment


def plot_activity(subject="101", activity="walking", sensor="hand_acc16",
                  start=0, seconds=10, magnitude=False):
    df = load_subject(subject)
    segment = get_segment(df, activity, start, seconds)

    if segment is None:
        return

    time = np.arange(len(segment)) / SAMPLE_RATE + start

    plt.figure(figsize=(10, 5))

    if magnitude:
        # Combine x/y/z into one size-of-motion signal.
        size = np.sqrt(segment[f"{sensor}_x"] ** 2
                       + segment[f"{sensor}_y"] ** 2
                       + segment[f"{sensor}_z"] ** 2)
        plt.plot(time, size, label="magnitude")
    else:
        for axis in "xyz":
            plt.plot(time, segment[f"{sensor}_{axis}"], label=axis)

    plt.xlabel("time (seconds)")
    plt.ylabel(sensor)
    plt.title(f"{sensor} during {activity} (subject {subject})")
    plt.legend()
    plt.show()


def plot_activity_multi(subject="101", activity="walking",
                        sensors=("hand_acc16", "chest_acc16"),
                        start=0, seconds=10, magnitude=False):
    df = load_subject(subject)
    segment = get_segment(df, activity, start, seconds)

    if segment is None:
        return

    time = np.arange(len(segment)) / SAMPLE_RATE + start

    # One panel per sensor so the signals can be compared side by side.
    fig, axes = plt.subplots(1, len(sensors), figsize=(6 * len(sensors), 5),
                             sharex=True)

    for ax, sensor in zip(np.atleast_1d(axes), sensors):
        if magnitude:
            size = np.sqrt(segment[f"{sensor}_x"] ** 2
                           + segment[f"{sensor}_y"] ** 2
                           + segment[f"{sensor}_z"] ** 2)
            ax.plot(time, size, label="magnitude")
        else:
            for axis in "xyz":
                ax.plot(time, segment[f"{sensor}_{axis}"], label=axis)

        ax.set_xlabel("time (seconds)")
        ax.set_ylabel(sensor)
        ax.set_title(sensor)
        ax.legend()

    fig.suptitle(f"{activity} (subject {subject})")
    plt.show()


def plot_subjects_multi(subjects=("101", "102"), activity="walking",
                        sensor="hand_acc16", starts=0, seconds=10,
                        magnitude=False):
    # starts may be one value shared by everyone or one value per subject.
    if np.isscalar(starts):
        starts = [starts] * len(subjects)

    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 5))

    for ax, subject, start in zip(np.atleast_1d(axes), subjects, starts):
        segment = get_segment(load_subject(subject), activity, start, seconds)
        if segment is None:
            continue

        time = np.arange(len(segment)) / SAMPLE_RATE + start

        if magnitude:
            size = np.sqrt(segment[f"{sensor}_x"] ** 2
                           + segment[f"{sensor}_y"] ** 2
                           + segment[f"{sensor}_z"] ** 2)
            ax.plot(time, size, label="magnitude")
        else:
            for axis in "xyz":
                ax.plot(time, segment[f"{sensor}_{axis}"], label=axis)

        ax.set_xlabel("time (seconds)")
        ax.set_ylabel(sensor)
        ax.set_title(f"subject {subject}")
        ax.legend()

    fig.suptitle(f"{sensor} during {activity}")
    plt.show()


def compute_spectrum(values, sample_rate=SAMPLE_RATE):
    # Interpolate missing values so the FFT still sees evenly spaced samples.
    clean = pd.Series(values).interpolate().bfill().ffill().to_numpy()
    clean = clean - clean.mean()

    spectrum = np.abs(np.fft.rfft(clean))
    freqs = np.fft.rfftfreq(len(clean), d=1 / sample_rate)

    return freqs, spectrum


def plot_spectrum(subject="101", activity="walking", sensor="hand_acc16",
                  start=0, seconds=10, max_freq=15):
    df = load_subject(subject)
    segment = get_segment(df, activity, start, seconds)

    if segment is None:
        return

    plt.figure(figsize=(10, 5))

    for axis in "xyz":
        freqs, spectrum = compute_spectrum(segment[f"{sensor}_{axis}"].to_numpy())
        plt.plot(freqs, spectrum, label=axis)

    # Most body-motion signal is at the low-frequency end.
    plt.xlim(0, max_freq)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strength")
    plt.title(f"Frequency content of {sensor} during {activity} (subject {subject})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_activity("101", "lying")
    plot_activity("101", "running")

    plot_activity("101", "walking", start=30, seconds=10)
    plot_activity("101", "running", magnitude=True)

    plot_activity_multi("101", "running", sensors=("hand_acc16", "ankle_acc16"))
    plot_subjects_multi(("101", "102"), "walking", starts=(0, 30))

    plot_spectrum("101", "running")
    plot_spectrum("101", "lying")
