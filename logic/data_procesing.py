import pandas as pd


def find_interrupts(data: dict | pd.DataFrame):



        number_of_measurements = len(data["time"])
        idxs = []
        for i in range(number_of_measurements):
            pvd = data['value_PV'][i] - data['value_PV'][i + 1] if i < number_of_measurements - 1 else 0
            print(pvd)
            if abs(pvd) > 3:
                idxs.append(i)
        print(idxs)
        start = 0

        good = []
        for z, idx in enumerate(idxs):
            time = data['time'][start:idx]
            value_temp = data['value_temp'][start:idx]
            value_hum = data['value_hum'][start:idx]
            value_acid = data['value_acid'][start:idx]
            value_PV = data['value_PV'][start:idx]

            good_course = {"time": time,
                           "value_temp": value_temp,
                           "value_hum": value_hum,
                           "value_acid": value_acid,
                           "value_PV": value_PV}
            good.append(good_course)
            start = idx+1
        return good



if __name__ == "__main__":
    x = {"value_PV": [1,2,3,4,5,56,6,8,7,78],
         "time": [1,2,3,4,5,56,6,8,7,78]}
    find_interrupts(x)