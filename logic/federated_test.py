import random

import keras.callbacks
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from colorama import Fore
class FederatedClass1:
    def __init__(self, betterdata):
        self.betterdata = betterdata

        self.federated_magic(num_rounds=25)
        self.federated_magic_partial(1,num_rounds=25)
        self.federated_magic_quantized_float(2,num_rounds=25)
        self.federated_magic_quantized_int(3,num_rounds=25)
        self.test_predictions()


    def test_predictions(self):
        import matplotlib.pyplot as plt
        for kk in range(4):
            loaded_model = tf.keras.models.load_model(f'./models/model_{kk}')
            data = [(bd.samples[0:2000], bd.labels[0:2000]) for bd in self.betterdata]
            data = data[0]
            loss, accuracy = loaded_model.evaluate(data[0], data[1])
            print(f"Test Loss: {loss}")
            print(f"Test Accuracy: {accuracy}")

            predictions = loaded_model.predict(data[0])

            x = []
            y = []
            y2 = []
            x2 = []
            for i in range(len(self.betterdata[0].iot_dict["value_temp"][0:2000])):
                if data[1][i] == 1:
                    x.append(i)
                    y.append(self.betterdata[0].iot_dict["value_temp"][0:2000][i])

            for idx, pred in enumerate(predictions):
                if pred == 1:
                    x2.append(idx)
                    y2.append(self.betterdata[0].iot_dict["value_temp"][0:2000][idx])

            plt.plot(self.betterdata[0].iot_dict["value_temp"][0:2000])



            plt.scatter(x,y,c="g",alpha=0.5)
            plt.scatter(x2,y2,c="r",alpha=0.5)
            plt.show()

    def federated_magic(self, magic_id = 0, num_rounds =20):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?


        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):
                weights = self.teach_model(data=data[client_id],weights=model.get_weights(), client_id=client_id, round_num = round_num, magic_id = magic_id)


                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")


    def federated_magic_partial(self, magic_id = 0, num_rounds =20):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_partial {Fore.RESET}")
        # Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata)  # 7?


        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            errors = []
            for client_id in range(NUM_CLIENTS):
                error = random.randint(5, 10)
                weights = self.teach_model_batch(data=data[client_id], weights=model.get_weights(), client_id=client_id,
                                           round_num=round_num, magic_id=magic_id, error_batches= error)
                weights = [w*error for w in weights]
                learned_weights.append(weights)
                errors.append(error)

            avg_weights = np.mean(learned_weights, axis=0)/sum(errors)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg + partial")

    def federated_magic_quantized_float(self, magic_id = 0, num_rounds =20):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_quantized_float {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?

        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):
                weights = self.teach_model_quantization_floats(data=data[client_id],weights=self.simple_quantize_floats(model.get_weights()), client_id=client_id, round_num = round_num, magic_id = magic_id)


                weights = self.simple_dequantize_floats(weights)

                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")

    def federated_magic_quantized_int(self, magic_id = 0, num_rounds =20):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_quantized_int {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?

        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):

                quantized, parms = self.quantize_weights_int(model.get_weights())
                weights, params = self.teach_model_quantization_int(data=data[client_id],weights=quantized,params=parms, client_id=client_id, round_num = round_num, magic_id = magic_id)


                weights = self.dequantize_weights_int(weights, params)

                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")


    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(60,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def teach_model(self, data, weights, client_id, round_num, magic_id):
        log_dir = f"logs/{magic_id}/{client_id}"


        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self._create_model()


        c_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]

        with client_writer.as_default():
            tf.summary.scalar("Loss", loss, step=round_num)
            tf.summary.scalar("Accuracy", accuracy, step=round_num)


        return c_model.get_weights()

    def teach_model_batch(self, data, weights, client_id, round_num, magic_id, error_batches=10):
        num_batches = 10
        batch_size = 50

        log_dir = f"logs/{magic_id}/{client_id}"
        batches = []
        for i in range(num_batches):
            bat = [data[0][i:batch_size+i],data[1][i:batch_size+i]]
            batches.append(bat)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self._create_model()




        c_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        c_model.set_weights(weights)


        for idx, batch in enumerate(batches):

            if idx == error_batches-1:
                break
            features, labels = batch
            loss, accuracy = c_model.train_on_batch(features,labels)


        with client_writer.as_default():
            tf.summary.scalar("Loss", loss, step=round_num)
            tf.summary.scalar("Accuracy", accuracy, step=round_num)


        return c_model.get_weights()

    def teach_model_quantization_floats(self, data, weights, client_id, round_num, magic_id):
        log_dir = f"logs/{magic_id}/{client_id}"
        weights = self.simple_dequantize_floats(weights)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self._create_model()


        c_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]

        with client_writer.as_default():
            tf.summary.scalar("Loss", loss, step=round_num)
            tf.summary.scalar("Accuracy", accuracy, step=round_num)


        return self.simple_quantize_floats(c_model.get_weights())


    def teach_model_quantization_int(self, data, weights, params, client_id, round_num, magic_id):
        log_dir = f"logs/{magic_id}/{client_id}"
        weights = self.dequantize_weights_int(weights, params)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self._create_model()


        c_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]

        with client_writer.as_default():
            tf.summary.scalar("Loss", loss, step=round_num)
            tf.summary.scalar("Accuracy", accuracy, step=round_num)


        return self.quantize_weights_int(c_model.get_weights())

    def avg_weights(self, weights):
        stacked_weights = np.stack(weights, axis=-1)

        # Compute the average weights across all models
        average_weights = np.mean(stacked_weights, axis=-1)
        return average_weights

    def simple_quantize_floats(self, weights_list: list):
        quantized_weights_list = []
        for weight in weights_list:
            quantized_weights = weight.astype(np.float16)
            quantized_weights_list.append(quantized_weights)
        return quantized_weights_list

    def simple_dequantize_floats(self, quantized_weights_list: list):
        dequantized_weights_list = []
        for quantized_weights in quantized_weights_list:
            dequantized_weights = quantized_weights.astype(np.float32)
            dequantized_weights_list.append(dequantized_weights)
        return dequantized_weights_list

    def quantize_weights_int(self, weights: list) -> tuple[list[np.ndarray], list[dict]]:
        quantized_weights = []
        params = []
        for weight in weights:
            mean = np.mean(weight)
            std_dev = np.std(weight)

            # Define clipping thresholds
            clip_min = mean - 2 * std_dev
            clip_max = mean + 2 * std_dev

            # Clip data
            clipped_data = np.clip(weight, clip_min, clip_max)
            max1 = np.max(clipped_data)
            min1 = np.min(clipped_data)

            norm_data = 2 * ((clipped_data - min1) / (max1 - min1)) - 1

            quant_data = np.round(127 * norm_data).astype(np.int8)
            param = {'min': min1,
                     'max': max1}

            quantized_weights.append(quant_data)
            params.append(param)
        return quantized_weights, params

    def dequantize_weights_int(self, quatized_weights: list, params: list[dict]) -> list:
        dequantized_weights = []
        for weight, param in zip(quatized_weights, params):
            dequantized_data = weight.astype(np.float32) / 127

            denorm_data = (dequantized_data + 1) / 2 * (param["max"] - param["min"]) + param["min"]
            dequantized_weights.append(denorm_data)

        return dequantized_weights


