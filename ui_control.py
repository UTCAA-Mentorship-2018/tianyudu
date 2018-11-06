"""
UI control tools
"""
from pprint import pprint
from typing import Dict


def choose_dataset(file_dirs: Dict[str, str]) -> str:
    print("Data set avaiable: ")
    pprint(file_dirs)
    print("Note that file directory selection is case senstive.")
    dataset_selection = input("Where is the dataset? [C: customize]>>> ")
    if dataset_selection.lower() == "c":
        chosen = input("Dataset dir >>> ")
    else:
        try:
            chosen = file_dirs[dataset_selection]
        except KeyError:
            raise Warning("Dataset not found.")
    return chosen
