import tensorflow as tf
import tensorflow.keras as keras


class SimpleConvNet(keras.Model):
    def __init__(
        self,
        input_shape: tuple,
        conv1_size: int,
        conv2_size: int,
        conv3_size: int,
        conv4_size: int,
        output_size: int,
        **kwargs
    ):
        super(SimpleConvNet, self).__init__(name="SimpleConvNet")
        self.inputs_ = keras.layers.Input(shape=input_shape)
        self._set_input_layer(self.inputs_)

        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.conv3_size = conv3_size
        self.conv4_size = conv4_size
        self.output_size = output_size

        self.conv1 = keras.layers.Conv2D(conv1_size, 3)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.ReLU()

        self.conv2 = keras.layers.Conv2D(conv2_size, 3)
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()

        self.max1 = keras.layers.MaxPooling2D(2)

        self.conv3 = keras.layers.Conv2D(conv3_size, 3)
        self.bn3 = keras.layers.BatchNormalization()
        self.relu3 = keras.layers.ReLU()

        self.conv4 = keras.layers.Conv2D(conv4_size, 3)
        self.bn4 = keras.layers.BatchNormalization()
        self.relu4 = keras.layers.ReLU()

        self.pooling = keras.layers.GlobalMaxPooling2D()
        self.dense1 = keras.layers.Dense(output_size)

        # adding batch dim with None
        self.build((None,) + self.input_shape_)

    # AFAIK: The most convenient method to print model.summary() and keras.utils.plot_model()
    # similar to the sequential or functional API like.
    def build_graph(self):
        x = keras.Input(shape=self.input_shape_)
        return keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.max1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pooling(x)

        return self.dense1(x)

    def get_config(self):
        return {
            "input_shape_": self.input_shape_,
            "conv1_size": self.conv1_size,
            "conv2_size": self.conv2_size,
            "conv3_size": self.conv3_size,
            "conv4_size": self.conv4_size,
            "output_size": self.output_size,
        }
