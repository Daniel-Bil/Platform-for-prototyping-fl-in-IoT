import numpy as np
import tensorflow as tf

class FederatedClass1:
    def __init__(self, betterdata):
        self.betterdata = betterdata
        self.federated_magic()

    def federated_magic(self):
        #Create main model *ON THE SERVER*

        model = self._create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        data = [(bd.samples[:500], bd.labels[:500]) for bd in self.betterdata]

        NUM_CLIENTS = len(self.betterdata) # 7?
        num_rounds = 10
        losses, accur = [], []

        for round_num in range(num_rounds):
            print("Round:", round_num + 1)
            learned_weights = []
            l= []
            a = []
            for client_id in range(NUM_CLIENTS):
                weights, history = self.teach_model(data=data[client_id],weights=model.get_weights())
                learned_weights.append(weights)
                loss = history.history['loss'][0]
                acc = history.history['accuracy'][0]
                l.append(loss)
                a.append(acc)

                print(f"    Client {client_id}: Loss = {loss}, Accuracy = {acc}")
            losses.append(np.mean(l))
            accur.append(np.mean(a))
            for lw in learned_weights:
                print()
                for w in lw:
                    print(type(w))
                    print(len(w))
                    print(np.array(w).shape)
            avg_weights = np.mean(learned_weights, axis=0)
            # avg_weights = self.avg_weights(learned_weights)
            print("agregated weights!")
            model.set_weights(avg_weights)

        print("finished agregating fed avg")
        print(losses)
        print(accur)

    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(60,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def teach_model(self, data, weights):
        c_model = self._create_model()
        c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        c_model.set_weights(weights)

        history = c_model.fit(data[0], data[1], epochs=1, verbose=0)



        return c_model.get_weights() , history


    def avg_weights(self, weights):
        stacked_weights = np.stack(weights, axis=-1)

        # Compute the average weights across all models
        average_weights = np.mean(stacked_weights, axis=-1)
        return average_weights





