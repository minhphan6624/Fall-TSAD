import re

_NAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(filename: str):
    m = _NAME_RE.match(filename)
    if not m:
        raise ValueError(f"Invalid filename {filename}")
    
    # Extract metadata from filename
    activity = m.group("code")
    subject = m.group("subject")
    trial = m.group("trial")
    is_fall = activity.startswith("F")
    group = "adult" if subject.startswith("SA") else "elderly"

    return {
        "activity": activity,
        "subject": subject,
        "group": group,
        "trial": int(trial[1:]),
        "activity_type": "fall" if is_fall else "adl",
        "is_fall": int(is_fall),
    }
