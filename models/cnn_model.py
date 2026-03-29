import tensorflow as tf
from tensorflow.keras import layers

# Utilisation de tf.keras.utils pour assurer la compatibilité avec toutes les versions de TF 2.x
@tf.keras.utils.register_keras_serializable(package="models")
class CustomCNN(tf.keras.Model):
    """
    Réseau de neurones convolutif personnalisé pour CIFAR-10.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4, **kwargs):
        # On passe les kwargs au parent pour la gestion interne de Keras
        super(CustomCNN, self).__init__(**kwargs)

        # ── Data Augmentation (intégrée au modèle) ──────────────────────────
        self.augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ],
            name="data_augmentation",
        )

        # ── Bloc 1 : 32 filtres ─────────────────────────────────────────────
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.pool1 = layers.MaxPooling2D((2, 2), name="pool1")

        # ── Bloc 2 : 64 filtres ─────────────────────────────────────────────
        self.conv2a = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2a")
        self.bn2a = layers.BatchNormalization(name="bn2a")
        self.conv2b = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2b")
        self.bn2b = layers.BatchNormalization(name="bn2b")
        self.pool2 = layers.MaxPooling2D((2, 2), name="pool2")

        # ── Bloc 3 : 128 filtres ────────────────────────────────────────────
        self.conv3a = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3a")
        self.bn3a = layers.BatchNormalization(name="bn3a")
        self.conv3b = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3b")
        self.bn3b = layers.BatchNormalization(name="bn3b")
        self.pool3 = layers.MaxPooling2D((2, 2), name="pool3")

        # ── Classificateur MLP ───────────────────────────────────────────────
        self.flatten = layers.Flatten(name="flatten")
        self.dense1 = layers.Dense(256, activation="relu", name="dense1")
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout1")
        self.dense2 = layers.Dense(128, activation="relu", name="dense2")
        self.dropout2 = layers.Dropout(dropout_rate / 2, name="dropout2")
        self.output_layer = layers.Dense(num_classes, activation="softmax", name="output")

    def call(self, inputs, training: bool = False):
        # Augmentation uniquement pendant l'entraînement
        x = self.augmentation(inputs, training=training)

        # Bloc 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        # Bloc 2
        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.pool2(x)

        # Bloc 3
        x = self.conv3a(x)
        x = self.bn3a(x, training=training)
        x = self.conv3b(x)
        x = self.bn3b(x, training=training)
        x = self.pool3(x)

        # Classificateur
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        """Nécessaire pour la sérialisation / désérialisation Keras."""
        config = super().get_config()
        config.update({
            "num_classes":  self.output_layer.units,
            "dropout_rate": self.dropout1.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Recrée le modèle depuis sa config."""
        return cls(**config)

    def build_graph(self, input_shape=(32, 32, 3)):
        """Construit le graphe pour afficher model.summary()."""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x, training=False))