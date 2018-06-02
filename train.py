import argparse

import keras

from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser("A Python command-line tool for training ocr models")
    parser.add_argument('generator', choices=['iam'])
    parser.add_argument('data_path', type=str)

    args = parser.parse_args()

    # parameters
    img_height = 48
    epochs = 1
    data_path = args.data_path

    # data generators
    train_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                        img_height=img_height)
    test_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                       img_height=img_height,
                                                       sample_size=100,
                                                       alphabet=train_data_generator.alphabet)

    # model
    model = ModelOcropy(train_data_generator.alphabet, img_height)

    # callbacks
    callback_lr_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='ctc_loss',
        factor=0.1,
        patience=5,
        verbose=1)
    callbacks = [callback_lr_plateau]

    # trainer
    trainer = Trainer(
        model,
        train_data_generator,
        test_data_generator,
        epochs=epochs,
        callbacks=callbacks)

    trainer.train()


if __name__ == "__main__":
    main()
