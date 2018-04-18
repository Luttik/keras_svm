from keras.models import Model
import sklearn
from keras.utils import to_categorical


class ModelSVMWrapper:
    """
    Linear stack of layers with the option to replace the end of the stack with a Support Vector Machine
    # Arguments
        layers: list of layers to add to the model.
        svm: The Support Vector Machine to use.
    """

    def __init__(self, model, svm=None):
        super().__init__()

        self.model = model
        self.intermediate_model = None  # type: Model
        self.svm = svm  # type: sklearn.svm.base.BaseSVC

        if svm is None:
            self.svm = sklearn.svm.SVC(kernel='linear')

    def add(self, layer):
        return self.model.add(layer)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data.
                If the input layer in the model is named, you can also pass a
                dictionary mapping the input name to a Numpy array.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data.
                If the output layer in the model is named, you can also pass a
                dictionary mapping the output name to a Numpy array.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, it will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                This will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        fit = self.model.fit(x, to_categorical(y), batch_size, epochs, verbose, callbacks, validation_split,
                             validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                             validation_steps, **kwargs)

        # Store intermediate model
        self.intermediate_model = Model(inputs=self.model.input,
                                        outputs=self.__get_split_layer().output)

        # Use output of intermediate model to train SVM
        intermediate_output = self.intermediate_model.predict(x)
        self.svm.fit(intermediate_output, y)

        return fit

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, steps=None):
        """
        Computes the accuracy on some input data, batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: labels, as a Numpy array.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            sample_weight: sample weights, as a Numpy array.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.

        # Returns
            Accuracy of the model mappin x to y

        # Raises
            RuntimeError: if the model was never compiled.
        """

        if self.intermediate_model is None:
            raise Exception("A model must be fit before running evaluate")
        output = self.predict(x, batch_size, verbose, steps)
        correct = [output[i] == y[i]
                   for i in range(len(output))]

        accuracy = sum(correct) / len(correct)

        return accuracy

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        """
        Computes the loss on some input data, batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: labels, as a Numpy array.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            sample_weight: sample weights, as a Numpy array.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            RuntimeError: if the model was never compiled.
        """
        intermediate_prediction = self.intermediate_model.predict(x, batch_size, verbose, steps)
        output = self.svm.predict(intermediate_prediction)

        return output

    def __get_split_layer(self):
        """
        Gets the layer to split on either "split_layer" or the second to last layer.

        :return: The layer to split on: from where the svm must replace the existing model.
        :raises ValueError: If not enough layers exist for a good split (at least two required)
        """
        if len(self.model.layers) < 3:
            raise ValueError('self.layers to small for a relevant split')

        for layer in self.model.layers:
            if layer.name == "split_layer":
                return layer

        # if no specific cut of point is specified we can assume we need to remove only the last (softmax) layer
        return self.model.layers[-3]
