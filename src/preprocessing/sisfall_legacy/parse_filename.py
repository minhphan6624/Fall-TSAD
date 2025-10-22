import re
from pathlib import Path
import pandas as pd

# Filename pattern: D01_SA01_R01.txt
_NAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(name: str) -> dict:
    """ Parse a file and extract the important metadata """
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid filename: {name}")
    
    # Extract metadata from regex groups
    activity = m.group("code")
    subject = m.group("subject")    
    trial = m.group("trial")
    is_fall = activity.startswith("F")
    group = "young" if subject.startswith("SA") else "elderly"

    return {
        "activity": activity,
        "subject": subject,
        "group": group,
        "trial": trial,
        "is_fall": int(is_fall)
    }