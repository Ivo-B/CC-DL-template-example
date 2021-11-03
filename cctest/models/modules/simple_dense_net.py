import tensorflow as tf
import tensorflow.keras as keras


class SimpleDenseNet(keras.Model):
    def __init__(self, input_shape: tuple, lin1_size: int, lin2_size: int, lin3_size: int, output_size: int, **kwargs):
        super(SimpleDenseNet, self).__init__(name="SimpleDenseNet")
        self.inputs_ = keras.layers.Input(shape=input_shape)
        self._set_input_layer(self.inputs_)

        self.lin1_size = lin1_size
        self.lin2_size = lin2_size
        self.lin3_size = lin3_size
        self.output_size = output_size

        self.flatt = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(lin1_size)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.ReLU()

        self.dense2 = keras.layers.Dense(lin2_size)
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()

        self.dense3 = keras.layers.Dense(lin3_size)
        self.bn3 = keras.layers.BatchNormalization()
        self.relu3 = keras.layers.ReLU()

        self.dense4 = keras.layers.Dense(output_size)

        self.build()

    def _set_input_layer(self, inputs):
        """add inputLayer to model and display InputLayers in model.summary()

        Args:
            inputs ([dict]): the result from `tf.keras.Input`
        """
        if isinstance(inputs, dict):
            self.inputs_layer = {
                n: keras.layers.InputLayer(input_tensor=i, name=n)
                for n, i in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            self.inputs_layer = [
                keras.layers.InputLayer(input_tensor=i, name=i.name)
                for i in inputs
            ]
        elif tf.is_tensor(inputs):
            self.inputs_layer = keras.layers.InputLayer(input_tensor=inputs, name=inputs.name)

    def build(self):
        super(SimpleDenseNet, self).build(self.inputs_.shape if tf.is_tensor(self.inputs_) else self.inputs_)
        self.out = self.call(self.inputs_)

    def call(self, inputs, training=None, mask=None):
        # (batch, width, height, 1) -> (batch, width*height*1)
        x = self.flatt(inputs)

        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dense3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return self.dense4(x)

    def get_config(self):
        return {
            "lin1_size": self.lin1_size,
            "lin2_size": self.lin2_size,
            "lin3_size": self.lin3_size,
            "output_size": self.output_size,
        }
