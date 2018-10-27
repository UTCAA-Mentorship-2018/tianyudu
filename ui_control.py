"""
UI control tools
"""

def choose_dataset(FILE_DIRS: dict) -> str:
    print("Data set avaiable: ")
    pprint(FILE_DIRS)
    dataset_selection = input("Where is the dataset? [C: customize]>>> ")
    if dataset_selection.lower() == "c":
        chosen = input("Dataset dir >>> ")
    try:
        chosen = FILE_DIRS[dataset_selection.upper()]
    except KeyError:
        raise Warning("Dataset not found.")
    return  chosen
