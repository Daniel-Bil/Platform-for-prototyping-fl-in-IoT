import numpy as np

from logic.dataProcesing import find_interrupts_withTime


class BetterDataClass:
    def __init__(self, data, name: str=""):
        self.name = name
        self.iot = data.copy(deep=True)
        self.iot_dict = self.iot.to_dict(orient='list')
        self.length = len(self.iot_dict["value_temp"])
        self.good = find_interrupts_withTime(self.iot_dict)

        self.samples = None
        self.labels = None

        self.errors = {"time": [],
                       "value_temp": [],
                       "value_hum": [],
                       "value_acid": [],
                       "value_PV": []}
        self.fill_errors()


    def fill_errors(self, key: str = "time"):
        if key == "time":
            self.errors[key] = find_interrupts_withTime(self.iot_dict, True)
        elif key == "value_temp":
            pass
        elif key == "value_hum":
            pass
        elif key == "value_acid":
            pass
        elif key == "value_PV":
            pass
        else:
            raise NotImplementedError

    def return_labels(self):
        raise NotImplementedError

    def create_samples(self, window: int = 20):
        samples = []
        for i in range(self.length-window):
            component1 = self.iot_dict["value_temp"][i:i+window]
            component2 = self.iot_dict["value_hum"][i:i+window]
            component3 = self.iot_dict["value_acid"][i:i+window]
            sample = np.array([component1, component2, component3]).flatten()
            samples.append(sample)

        self.samples = np.array(samples)

        labels = np.zeros(self.length-window)
        keys = ["time", "value_temp", "value_hum", "value_acid", "value_PV"]
        for key in keys:
            for error in self.errors[key]:
                start = error-(window/2-1)
                end = error+(window/2)
                if start < 0:
                    start = 0
                if end >= self.length-window:
                    end = self.length-window
                labels[int(start):int(end)] = 1

        self.labels = np.array(labels)
        print("brek")
        print()