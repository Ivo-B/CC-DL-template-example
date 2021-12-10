import tensorflow.keras as keras


class SimpleUNet(keras.Model):
    def __init__(
        self,
        input_shape: list[int, int, int],
        start_filters: int,
        kernel_size: tuple[int, int],
        num_down_blocks: int,
        num_classes: int,
        **kwargs,
    ):
        super(SimpleUNet, self).__init__(name="SimpleUNet")
        self.input_shape_ = tuple(input_shape)
        self.start_filters = start_filters
        self.kernel_size = tuple(kernel_size)
        self.num_down_blocks = num_down_blocks
        self.num_classes = num_classes

        self.down_layers = []
        for idx in range(num_down_blocks):
            self.down_layers += SimpleUNet.conv2d_block(start_filters, kernel_size, name=f"down{idx}_CB")
            self.down_layers += [keras.layers.MaxPooling2D((2, 2), strides=2, name=f"down{idx}_MP")]
            start_filters = start_filters * 2  # double the number of filters with each layer

        self.latent = SimpleUNet.conv2d_block(start_filters, kernel_size, name=f"latent_CB")

        self.up_layers = []
        for idx in range(num_down_blocks):
            start_filters //= 2  # decreasing number of filters with each layer
            self.up_layers += [
                keras.layers.Conv2DTranspose(start_filters, (2, 2), strides=(2, 2), padding="same", name=f"up{idx}_CT"),
            ]
            self.up_layers += [keras.layers.Concatenate(name=f"up{idx}_Concat")]
            self.up_layers += SimpleUNet.conv2d_block(start_filters, kernel_size, name=f"up{idx}_CB")

        self.conv1 = keras.layers.Conv2D(num_classes, (1, 1), name="conv_logits")
        self.out_act = keras.layers.Activation("softmax", dtype="float32", name="act_predictions")

        # adding batch dim with None
        self.build((None,) + self.input_shape_)

    @staticmethod
    def conv2d_block(filters: int, kernel_size: tuple[int, int], name: str = "ConvBlock", **kwargs):
        out_list = []
        for i in range(2):
            out_list.append(
                keras.layers.Conv2D(filters, kernel_size, padding="same", use_bias="none", name=f"{name}_{i}_conv"),
            )
            out_list.append(keras.layers.BatchNormalization(name=f"{name}_{i}_BN"))
            out_list.append(keras.layers.LeakyReLU(alpha=2e-1, name=f"{name}_{i}_leakyRelu"))
        return out_list

    # AFAIK: The most convenient method to print model.summary() and keras.utils.plot_model()
    # similar to the sequential or functional API like.
    def build_graph(self):
        x = keras.Input(shape=self.input_shape_)
        return keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, inputs, training=None, mask=None):
        x = self.down_layers[0](inputs)
        down_layers = []
        for layer in self.down_layers[1:]:
            if type(layer) is keras.layers.MaxPooling2D:
                down_layers.append(x)
            x = layer(x)

        for layer in self.latent:
            x = layer(x)

        concat_counter = -1
        for layer in self.up_layers:
            if type(layer) is keras.layers.Concatenate:
                x = layer([x, down_layers[concat_counter]])
                concat_counter -= 1
            else:
                x = layer(x)

        x = self.conv1(x)
        return self.out_act(x)

    def get_config(self):
        return {
            "input_shape": self.input_shape_,
            "start_filters": self.start_filters,
            "kernel_size": self.kernel_size,
            "num_down_blocks": self.num_down_blocks,
            "num_classes": self.num_classes,
        }
