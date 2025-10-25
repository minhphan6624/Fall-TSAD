import numpy as np

WINDOW_SIZE = 200     # 1 s at 200 Hz
STRIDE      = 100     # 50 % overlap

def label_window(start, end, fall_range, threshold=0.3):
    f_start = fall_range.start
    f_end = fall_range.stop

    overlap = max(0, min(end, f_end) - max(start, f_start))

    window_length = end - start
    return 1 if overlap / window_length >= threshold else 0

def segment_and_label(data, smv, meta, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Segment accelerometer data into windows and label them using SMV peaks.
    Only windows around the impact in fall trials are labelled 1.
    """
    impact_idx = np.argmax(smv)

    # -0.75s to +0.75s around impact
    start = max(0, impact_idx - 150)
    end = min(len(smv), impact_idx + 150)
    fall_range = range(start, end)

    X, y = [], []
    # Slide a 1-second window through the signal with 0.5-s overlap.
    for start in range(0, len(data) - window_size, stride):
        end = start + window_size
        window = data[start:end]

        label = 0
        if meta["is_fall"]:
            label = label_window(start, end, fall_range, 0.3)

        X.append(window)
        y.append(label)

    return np.stack(X), np.array(y)