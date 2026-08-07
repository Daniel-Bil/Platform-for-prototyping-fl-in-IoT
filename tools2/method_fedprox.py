import time
import tensorflow as tf


def train_fedprox_client(local_model, global_weights, dataset, epochs, mu=0.01, client_id="Unknown"):
    """
    Trening lokalny z regularyzacją FedProx (człon proksymalny).
    Zoptymalizowany pod kątem szybkości na CPU (@tf.function) z pełnym logowaniem postępu.
    """
    optimizer = tf.keras.optimizers.Adam()

    # Automatyczny dobór funkcji straty
    output_shape = local_model.output_shape[-1]
    if output_shape == 1:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Kompilacja kroku uczenia do szybkiego grafu TF (klucz do przyspieszenia na CPU!)
    @tf.function
    def train_step(x_batch, y_batch, global_w):
        with tf.GradientTape() as tape:
            logits = local_model(x_batch, training=True)
            base_loss = loss_fn(y_batch, logits)

            # Szybkie sumowanie kwadratów różnic wag
            proximal_term = tf.add_n([
                tf.reduce_sum(tf.square(lw - gw))
                for lw, gw in zip(local_model.trainable_weights, global_w)
            ])

            total_loss = base_loss + (mu / 2.0) * proximal_term

        grads = tape.gradient(total_loss, local_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
        return total_loss

    print(f"    📡 [FedProx] Start treningu klienta: {client_id} (Epoki: {epochs}, mu: {mu})")

    for epoch in range(epochs):
        t0 = time.perf_counter()
        step_count = 0
        last_loss = 0.0

        for x_batch, y_batch in dataset:
            last_loss = train_step(x_batch, y_batch, global_weights)
            step_count += 1

        dt = time.perf_counter() - t0
        print(f"      -> Epoka {epoch + 1}/{epochs} zakończona w {dt:.2f}s | Batche: {step_count} | Ostatnia strata (Loss): {last_loss:.4f}")

    return local_model.get_weights()