import numpy as np

WINDOW_SIZE = 200     # 1 s at 200 Hz
STRIDE      = 100     # 50 % overlap

def segment_and_label(data, smv, meta, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Segment accelerometer data into windows and label them using SMV peaks.
    Only windows around the impact in fall trials are labelled 1.
    """
    impact_idx = np.argmax(smv)

    # -1s to +2s around impact
    fall_range = range(max(0, impact_idx - 200), min(len(smv), impact_idx + 200))

    X, y = [], []
    # Slide a 1-second window through the signal with 0.5-s overlap.
    for start in range(0, len(data) - window_size, stride):
        end = start + window_size
        window = data[start:end]

        label = 0
        # If it's a fall trial and window overlaps with fall range, label as 1
        if meta["is_fall"] and any(i in fall_range for i in range(start, end)):
            label = 1

        X.append(window)
        y.append(label)

    return np.stack(X), np.array(y)