# Keras SVM

[![PyPI - Status](https://img.shields.io/pypi/status/keras-svm.svg)](https://pypi.org/project/keras-svm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/keras-svm.svg)](https://pypi.org/project/keras-svm/)
[![PyPI - License](https://img.shields.io/pypi/l/keras-svm.svg)](https://github.com/Luttik/keras_svm/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/keras-svm.svg)](https://pypi.org/project/keras-svm/)

## Purpose
Provides a wrapper class that effectively replaces the softmax of your Keras model with a SVM.

The SVM has no impact on the training of the Neural Network, but replacing softmax with an SVM has been shown to perform better on unseen data.

## Code examples
### Example construction
```
# Build a classical model
def build_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten(name="intermediate_output"))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  
  # The extra metric is important for the evaluate function
  model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

# Wrap it in the ModelSVMWrapper
wrapper = ModelSVMWrapper(build_model())
```

### Training while maintaining an accuracy score
```
accuracy = {
    "with_svm": [],
    "without_svm": []
}

epochs = 10
for i in range(epochs):
  print('Starting run: {}'.format(i))
  wrapper.fit(train_images, train_labels, epochs=1, batch_size=64)
  accuracy["with_svm"].append(wrapper.evaluate(test_images, test_labels))
  accuracy["without_svm"].append(
      wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])
```
