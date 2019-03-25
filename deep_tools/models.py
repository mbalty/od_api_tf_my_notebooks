from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD

from keras.layers import Input, Dense, concatenate, Flatten
from keras.models import Model


def binary_classification_keras_application_model_transfer_learning_imagenet(keras_application_type,
                                optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                                input_edge=224, intermediate_dense_layer_size=64):
    keras_application_model = keras_application_type(input_tensor=Input((input_edge, input_edge, 3)),
                                                     weights="imagenet", include_top=True)
    keras_application_model.layers.pop()
    lastLayer = keras_application_model.layers[-1]
    if len(lastLayer.output.shape) > 2:
        flat = Flatten()(lastLayer.output)
    else:
        flat = lastLayer.output
    dense = Dense(intermediate_dense_layer_size, activation='relu')(flat)
    out = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=keras_application_model.input, outputs=out)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


def resnet50base_object_validation(n_classes, optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)):
    resnetModel = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
    resnetModel.layers.pop()

    classInput = Input(shape=(n_classes,))
    imageInput = resnetModel.input
    lastResnetTensor = resnetModel.layers[-1].output
    mergeLayer = concatenate([classInput, lastResnetTensor])

    dense = Dense(512, activation='relu')(mergeLayer)
    dense = Dense(512, activation='relu')(dense)
    out = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[classInput, imageInput], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model