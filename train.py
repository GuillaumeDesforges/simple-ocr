import argparse

import keras

from engine.callbacks.levenshtein import LevenshteinCallback
from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


def main():
    # cmd args
    parser = argparse.ArgumentParser("A Python command-line tool for training ocr models")
    parser.add_argument('generator', choices=['iam'])
    parser.add_argument('data_path', type=str)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--plateau_reduce_lr', type=bool, default=True)
    parser.add_argument('--image_height', type=int, default=48)
    args = parser.parse_args()

    # parameters
    img_height = args.image_height
    data_path = args.data_path
    epochs = args.epochs
    lr = args.lr
    reduce_lr_on_plateau = args.plateau_reduce_lr

    # data generators
    train_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                        img_height=img_height)

    steps_per_epochs = args.steps_epochs

    test_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                       img_height=img_height,
                                                       sample_size=100,
                                                       alphabet=train_data_generator.alphabet)

    # model
    model = ModelOcropy(train_data_generator.alphabet, img_height)

    # callbacks
    callbacks = []
    if reduce_lr_on_plateau:
        callback_lr_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor='val_ctc_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            cooldown=5)
        callbacks.append(callback_lr_plateau)
    callback_levenshtein = LevenshteinCallback(test_data_generator, size=10)
    callbacks.append(callback_levenshtein)

    # trainer
    trainer = Trainer(
        model,
        train_data_generator,
        test_data_generator,
        lr=lr,
        epochs=epochs,
        steps_per_epochs=steps_per_epochs,
        callbacks=callbacks)

    trainer.train()


if __name__ == "__main__":
    main()
