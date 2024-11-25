from pathlib import Path


class Configurator:
    def __init__(self):
        self.generate_paths()

    def generate_paths(self):
        self.base_path = Path("..")

        self.clean_data_path = self.base_path / "cleaned_data"

        self.results_path = self.base_path / "results"


