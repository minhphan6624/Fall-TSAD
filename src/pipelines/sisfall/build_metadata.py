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
    code = m.group("code")
    subject = m.group("subject")    
    trial = m.group("trial")
    is_fall = code.startswith("F")
    group = "young" if subject.startswith("SA") else "elderly"

    return {
        "filename": name,
        "code": code,
        "subject": subject,
        "group": group,
        "trial": trial,
        "is_fall": int(is_fall)
    }

def build_metadata(raw_dir: str) -> pd.DataFrame:
    """
    Build a metadata DataFrame from the raw data directory
    Each row corresponds to a data file with extracted metadata
    """
    rows = []
    # Iterate over subject directories and files to parse metadata
    for subj_dir in sorted(raw_dir.glob("S[A|E][0-9][0-9]")): #raw/SA01/*.txt
        for file in sorted(subj_dir.glob("*.txt")):
            info = parse_filename(file.name)
            info["path"] = str(file.resolve())
            rows.append(info)

    if not rows:
        raise FileNotFoundError(f"No data found in {raw_dir}")

    df = pd.DataFrame(rows) 
    
    return df.sort_values(by=["subject", "filename"]).reset_index(drop=True)
