import os
import random

import keras.callbacks
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from colorama import Fore
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import json


def create_confusion_matrix_better(samples, labels, model, epoch, magic_id, model_id):
    # Ensure samples are in the correct format (batch_size, features)
    if len(samples.shape) == 1:  # if it's a 1D array
        samples = np.expand_dims(samples, axis=0)

    # Get predictions for the entire batch
    outputs = model.predict(samples)
    y_pred = np.argmax(outputs, axis=1)  # Convert softmax outputs to class predictions

    y_true = labels  # Ground truth labels (already a list)

    # Constant for classes
    classes = ['Class 0', 'Class 1']

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap=sn.color_palette("flare", as_cmap=True))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot as an image and log to TensorBoard
    plt.tight_layout()
    fig = plt.gcf()

    # Convert the plot to a tensor
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    log_dir = f"logs/{model_id}/{magic_id}/matrix"

    client_writer = tf.summary.create_file_writer(log_dir)
    # Log the image to TensorBoard
    with client_writer.as_default():
        tf.summary.image("Confusion Matrix", np.expand_dims(img, axis=0), step=epoch)

    plt.close()


# def create_confusion_matrix(samples, labels, model, epoch, magic_id):
#     y_pred = []  # save predictions
#     y_true = labels  # save ground truth (already a list)
#
#     # Iterate over data
#     for inputs in samples:
#         inputs = np.expand_dims(inputs, axis=0)  # Add batch dimension
#         output = model.predict(inputs)  # Feed Network
#
#         # Get predictions
#         output = np.argmax(output, axis=1)
#         y_pred.extend(output)  # save predictions
#
#     # Constant for classes
#     classes = ['Class 0', 'Class 1']
#
#     # Build confusion matrix
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
#
#     # Plot the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sn.heatmap(df_cm, annot=True, fmt='d', cmap=sn.color_palette("flare", as_cmap=True))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#
#     # Save the plot as an image and log to TensorBoard
#     plt.tight_layout()
#     fig = plt.gcf()
#
#     # Convert the plot to a tensor
#     fig.canvas.draw()
#     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     log_dir = f"logs/{magic_id}/matrix"
#
#
#     client_writer = tf.summary.create_file_writer(log_dir)
#     # Log the image to TensorBoard
#     with client_writer.as_default():
#         tf.summary.image("Confusion Matrix", np.expand_dims(img, axis=0), step=epoch)
#
#     plt.close()


class FederatedClass1:
    def __init__(self, betterdata):
        self.betterdata = betterdata
        model_ids = os.listdir("./models2")
        for i, model_id in enumerate(model_ids):
            self.federated_magic(num_rounds=5+i*5, model_id = model_id)
            self.federated_magic_partial(1, num_rounds=5+i*5, model_id = model_id)
            self.federated_magic_quantized_float(2, num_rounds=5+i*5, model_id = model_id)
            self.federated_magic_quantized_int(3, num_rounds=5+i*5, model_id = model_id)
        # self.single_benchmark(5, num_rounds=10)
        # self.test_predictions()

    def test_predictions(self):
        import matplotlib.pyplot as plt
        # for kk in range(4):
        loaded_model = tf.keras.models.load_model(f'./models/model_{0}')
        data = [(bd.samples[0:2000], bd.labels[0:2000]) for bd in self.betterdata]
        data = data[0]
        history = loaded_model.evaluate(data[0], data[1])
        print(history)
        print(f"Test Loss: {history[0]}")
        print(f"Test Accuracy: {history[1]}")

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

    def single_benchmark(self, magic_id = 0, num_rounds =20):
        print(f"{Fore.LIGHTBLUE_EX} single_benchmark {Fore.RESET}")
        # Create main model *ON THE SERVER*
        whole_data = (self.betterdata[1].samples, self.betterdata[1].labels)
        data = (self.betterdata[1].samples[:1500], self.betterdata[1].labels[:1500])
        data2 = (self.betterdata[1].samples[1500:], self.betterdata[1].labels[1500:])

        print(len(data[1][data[1]==1]))
        print(len(data[1][data[1]==0]))

        print("data2")
        print(len(data2[1][data2[1] == 1]))
        print(len(data2[1][data2[1] == 0]))

        good_data = []
        bad_data = []
        for i in range(1500):
            if data[1][i]==1:
                bad_data.append([data[0][i], data[1][i]])
            else:
                good_data.append([data[0][i], data[1][i]])

        print(len(good_data))
        print(len(bad_data))

        random.shuffle(good_data)
        random.shuffle(bad_data)
        good_data = good_data[:len(bad_data)]
        print(good_data)

        data_all = [*good_data, *bad_data]
        print(len(data_all))
        random.shuffle(data_all)


        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])

        samples = []
        labels = []
        for d in data_all:
            samples.append(d[0])
            labels.append(d[1])

        print(samples)
        print(labels)
        samples = np.array(samples)
        labels = np.array(labels)
        history = model.fit(samples, labels, epochs=100, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]
        precision = history.history['precision'][0]
        recall = history.history['recall'][0]
        auc = history.history['auc'][0]

        print(loss,accuracy,precision,recall,auc)
        print("END")

        predictions = model.predict(self.betterdata[1].samples)

        x = []
        y = []
        y2 = []
        x2 = []
        for i in range(len(self.betterdata[1].iot_dict["value_temp"])-20):
            if whole_data[1][i] == 1:
                x.append(i)
                y.append(self.betterdata[1].iot_dict["value_temp"][i])

        for idx, pred in enumerate(predictions):
            if pred == 1:
                x2.append(idx)
                y2.append(self.betterdata[1].iot_dict["value_temp"][idx])

        plt.plot(self.betterdata[1].iot_dict["value_temp"])

        plt.scatter(x, y, c="g", alpha=0.5)
        plt.scatter(x2, y2, c="r", alpha=0.5)
        plt.show()



    def federated_magic(self, magic_id = 0, num_rounds =20, model_id="model_0"):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self.recreate_architecture_with_name(model_id)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                            tf.keras.metrics.Precision(name='precision'),
                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                            tf.keras.metrics.AUC(name='auc')])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?


        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):
                weights = self.teach_model(data=data[client_id],weights=model.get_weights(), client_id=client_id, round_num = round_num, magic_id = magic_id, model_name=model_id)


                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")

        create_confusion_matrix_better(data[2][0],data[2][1],model,1,magic_id=magic_id, model_id = model_id)



    def federated_magic_partial(self, magic_id = 0, num_rounds =20, model_id="model_0"):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_partial {Fore.RESET}")
        # Create main model *ON THE SERVER*

        model = self.recreate_architecture_with_name(model_id)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])
        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata)  # 7?


        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            errors = []
            for client_id in range(NUM_CLIENTS):
                error = random.randint(5, 10)
                weights = self.teach_model_batch(data=data[client_id], weights=model.get_weights(), client_id=client_id,
                                           round_num=round_num, magic_id=magic_id, error_batches= error, model_name=model_id)
                weights = [w*error for w in weights]
                learned_weights.append(weights)
                errors.append(error)

            avg_weights = np.mean(learned_weights, axis=0)/sum(errors)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg + partial")

        create_confusion_matrix_better(data[2][0], data[2][1], model, 1, magic_id=magic_id, model_id = model_id)

    def federated_magic_quantized_float(self, magic_id = 0, num_rounds =20, model_id="model_0"):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_quantized_float {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self.recreate_architecture_with_name(model_id)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?

        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):
                weights = self.teach_model_quantization_floats(data=data[client_id],weights=self.simple_quantize_floats(model.get_weights()), client_id=client_id, round_num = round_num,
                                                               magic_id = magic_id, model_name=model_id)


                weights = self.simple_dequantize_floats(weights)

                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")

        create_confusion_matrix_better(data[2][0], data[2][1], model, 1, magic_id=magic_id, model_id = model_id)

    def federated_magic_quantized_int(self, magic_id = 0, num_rounds =20, model_id="model_0"):
        print(f"{Fore.LIGHTBLUE_EX} federated_magic_quantized_int {Fore.RESET}")
        #Create main model *ON THE SERVER*

        model = self.recreate_architecture_with_name(model_id)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])



        data = [(bd.samples[:1500], bd.labels[:1500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?

        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            for client_id in range(NUM_CLIENTS):

                quantized, parms = self.quantize_weights_int(model.get_weights())
                weights, params = self.teach_model_quantization_int(data=data[client_id],weights=quantized,params=parms, client_id=client_id, round_num = round_num,
                                                                    magic_id = magic_id, model_name=model_id)


                weights = self.dequantize_weights_int(weights, params)

                learned_weights.append(weights)


            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)
        model.save(f'./models/model_{magic_id}')
        print("finished agregating fed avg")
        create_confusion_matrix_better(data[2][0], data[2][1], model, 1, magic_id=magic_id, model_id = model_id)

    # def _create_model(self):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Dense(32, activation='relu', input_shape=(60,)),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(32, activation='relu'),
    #         tf.keras.layers.Dense(1, activation='sigmoid')
    #     ])
    #     return model

    def teach_model(self, data, weights, client_id, round_num, magic_id,model_name):
        log_dir = f"logs/{model_name}/{magic_id}/{client_id}"


        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self.recreate_architecture_with_name(model_name)


        c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                            tf.keras.metrics.Precision(name='precision'),
                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                            tf.keras.metrics.AUC(name='auc')])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]
        precision = history.history['precision'][0]
        recall = history.history['recall'][0]
        auc = history.history['auc'][0]

        with client_writer.as_default():
            tf.summary.scalar(f"{model_name}/{magic_id}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Auc", auc, step=round_num)

            tf.summary.scalar(f"{model_name}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/Auc", auc, step=round_num)


        return c_model.get_weights()

    def teach_model_batch(self, data, weights, client_id, round_num, magic_id, model_name, error_batches=10):
        num_batches = 10
        batch_size = 50

        log_dir = f"logs/{model_name}/{magic_id}/{client_id}"
        batches = []
        for i in range(num_batches):
            bat = [data[0][i:batch_size+i],data[1][i:batch_size+i]]
            batches.append(bat)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self.recreate_architecture_with_name(model_name)




        c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                            tf.keras.metrics.Precision(name='precision'),
                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                            tf.keras.metrics.AUC(name='auc')])
        c_model.set_weights(weights)


        for idx, batch in enumerate(batches):

            if idx == error_batches-1:
                break
            features, labels = batch
            results = c_model.train_on_batch(features,labels)
            loss = results[0]
            accuracy = results[1]
            precision = results[2]
            recall = results[3]
            auc = results[4]

        with client_writer.as_default():
            tf.summary.scalar(f"{model_name}/{magic_id}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Auc", auc, step=round_num)

            tf.summary.scalar(f"{model_name}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/Auc", auc, step=round_num)


        return c_model.get_weights()

    def teach_model_quantization_floats(self, data, weights, client_id, round_num, magic_id, model_name):
        log_dir = f"logs/{model_name}/{magic_id}/{client_id}"
        weights = self.simple_dequantize_floats(weights)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self.recreate_architecture_with_name(model_name)


        c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                            tf.keras.metrics.Precision(name='precision'),
                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                            tf.keras.metrics.AUC(name='auc')])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]
        precision = history.history['precision'][0]
        recall = history.history['recall'][0]
        auc = history.history['auc'][0]

        with client_writer.as_default():
            tf.summary.scalar(f"{model_name}/{magic_id}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Auc", auc, step=round_num)

            tf.summary.scalar(f"{model_name}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/Auc", auc, step=round_num)


        return self.simple_quantize_floats(c_model.get_weights())


    def teach_model_quantization_int(self, data, weights, params, client_id, round_num, magic_id, model_name):
        log_dir = f"logs/{model_name}/{magic_id}/{client_id}"
        weights = self.dequantize_weights_int(weights, params)

        client_writer = tf.summary.create_file_writer(log_dir)

        c_model = self.recreate_architecture_with_name(model_name)


        c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                            tf.keras.metrics.Precision(name='precision'),
                                                                            tf.keras.metrics.Recall(name='recall'),
                                                                            tf.keras.metrics.AUC(name='auc')])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)

        loss = history.history['loss'][0]  # Loss for the first (and only) epoch
        accuracy = history.history['accuracy'][0]
        precision = history.history['precision'][0]
        recall = history.history['recall'][0]
        auc = history.history['auc'][0]

        with client_writer.as_default():
            tf.summary.scalar(f"{model_name}/{magic_id}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/{magic_id}/Auc", auc, step=round_num)


            tf.summary.scalar(f"{model_name}/Loss", loss, step=round_num)
            tf.summary.scalar(f"{model_name}/Accuracy", accuracy, step=round_num)
            tf.summary.scalar(f"{model_name}/Precision", precision, step=round_num)
            tf.summary.scalar(f"{model_name}/Recall", recall, step=round_num)
            tf.summary.scalar(f"{model_name}/Auc", auc, step=round_num)


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

    def recreate_architecture(self, id=0):
        directory = "./models2"
        model_names = os.listdir(directory)
        with open(f"{directory}/{model_names[id]}", 'r') as json_file:
            data = json.load(json_file)
        model = tf.keras.Sequential()

        for idx, layer in enumerate(data):
            if idx == 0:
                if layer["layer_type"] == "Dense":
                    model.add(
                        tf.keras.layers.Dense(units=layer["units"], activation=layer["activation"], input_shape=(60,)))
            else:
                if layer["layer_type"] == "Dense":
                    model.add(tf.keras.layers.Dense(units=layer["units"], activation=layer["activation"]))

        model.summary()
        return model

    def recreate_architecture_with_name(self, name="model_0"):
        directory = "./models2"
        with open(f"{directory}/{name}", 'r') as json_file:
            data = json.load(json_file)
        model = tf.keras.Sequential()

        for idx, layer in enumerate(data):
            if idx == 0:
                if layer["layer_type"] == "Dense":
                    model.add(
                        tf.keras.layers.Dense(units=layer["units"], activation=layer["activation"], input_shape=(60,)))
            else:
                if layer["layer_type"] == "Dense":
                    model.add(tf.keras.layers.Dense(units=layer["units"], activation=layer["activation"]))

        # model.summary()
        return model