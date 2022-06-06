import pandas as pd


class DatasetReadingOperation:
    def read(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, header=0, index_col=0, on_bad_lines='skip', engine="python")
