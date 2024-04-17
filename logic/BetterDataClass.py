from logic.dataProcesing import find_interrupts_withTime


class BetterDataClass:
    def __init__(self, data):
        self.iot = data.copy()
        self.iot_dict = self.iot.to_dict(orient='list')

        self.good = find_interrupts_withTime(self.iot_dict)

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