import random

import keras.callbacks
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class FederatedClass1:
    def __init__(self, betterdata):
        self.betterdata = betterdata

        self.federated_magic()
        self.federated_magic_quantized_no_waste(1)




    def federated_magic(self, magic_id = 0):
        #Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        data = [(bd.samples[:500], bd.labels[:500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?
        num_rounds = 20

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

        print("finished agregating fed avg")


    def federated_magic_quantized_no_waste(self, magic_id = 0):
        # Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        data = [(bd.samples[:500], bd.labels[:500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata)  # 7?
        num_rounds = 20

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

        print("finished agregating fed avg + partial")


    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(60,)),
            tf.keras.layers.Dense(16, activation='relu'),
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



    def avg_weights(self, weights):
        stacked_weights = np.stack(weights, axis=-1)

        # Compute the average weights across all models
        average_weights = np.mean(stacked_weights, axis=-1)
        return average_weights

    def simple_quantize(self, weights_list):
        quantized_weights_list = []
        for weight in weights_list:
            # Cast float32 weights to int8 directly
            quantized_weights = weight.astype(np.int8)
            quantized_weights_list.append(quantized_weights)
        return quantized_weights_list

    def simple_dequantize(self, quantized_weights_list):
        dequantized_weights_list = []
        for quantized_weights in quantized_weights_list:
            # Cast int8 weights back to float32
            dequantized_weights = quantized_weights.astype(np.float32)
            dequantized_weights_list.append(dequantized_weights)
        return dequantized_weights_list




