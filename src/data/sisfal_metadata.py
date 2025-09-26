import re
from pathlib import Path
import pandas as pd
from .sisfall_paths import RAW_DATA_DIR

# Filename pattern: D01_SA01_R01.txt
_NAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(name: str) -> dict:
    """
    Parse a file and extract the important metadata
    """
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid filename: {name}")
    
    
    code = m.group("code")
    subject = m.group("subject")    
    trial = m.group("trial")
    is_fall = code.startswith("F")
    group = "Adult" if subject.startswith("SA") else "Elderly"

    return {
        "filename": name,
        "code": code,
        "subject": subject,
        "group": group,
        "trial": trial,
        "is_fall": int(is_fall)
    }

def build_metadata(raw_dir: str) -> pd.DataFrame:
    rows = []

    # Iterate over subject directories and files to parse metadata
    for subj_dir in sorted(raw_dir.glob("S[A|E][0-9][0-9]")): #raw/SA01/*.txt
        for p in sorted(subj_dir.glob("*.txt")):
            info = parse_filename(p.name)
            info["path"] = str(p.resolve())
            rows.append(info) 

    if not rows:
        raise FileNotFoundError(f"No data found in {raw_dir}")
    
    # Create a DataFrame
    df = pd.DataFrame(rows)

    # Return sorted DataFrame by subject and filename
    return df.sort_values(by=["subject", "filename"]).reset_index(drop=True)
