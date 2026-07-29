Every file in this directory is either:
1) A set of antarctic ladder measures
or
2) A helper file.

Each Antarctic Measure is a class of the following structure:

class AntarcticMeasure:
    def __init__(self) -> None:
        # Init Code

    def country_dict(self) -> dict:
        return {Dictionary Mapping Country strings to values}

    def figure_title(self) -> str:
        return {Title of antarctic measure metric}
    
    def save_full_figures(self, path:str): {OPTIONAL}
        # Exports a csv breaking down figures by time. E.g. yearly or decadely figures.
