import argparse
import os
from time import gmtime, strftime

import keras

from engine.callbacks.levenshtein import LevenshteinCallback
from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting
from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


def main():
    # cmd args
    parser = argparse.ArgumentParser("A Python command-line tool for training ocr models")
    parser.add_argument('generator', choices=['iam', 'bodmer'])
    parser.add_argument('data_path', type=str)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps-epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--plateau-reduce-lr', type=bool, default=True)
    parser.add_argument('--image-height', type=int, default=48)
    parser.add_argument('--levenshtein', type=bool, default=True)
    parser.add_argument('--tensorboard', type=bool, default=True)
    args = parser.parse_args()

    # parameters
    generator_type = args.generator
    img_height = args.image_height
    data_path = args.data_path
    epochs = args.epochs
    steps_per_epochs = args.steps_epochs
    lr = args.lr
    reduce_lr_on_plateau = args.plateau_reduce_lr
    levenshtein = args.levenshtein
    tensorboard = args.tensorboard

    # data generators
    if generator_type == 'iam':
        train_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                            img_height=img_height)
        test_data_generator = BatchGeneratorIAMHandwriting(data_path,
                                                           img_height=img_height,
                                                           sample_size=100,
                                                           alphabet=train_data_generator.alphabet)
    elif generator_type == 'bodmer':
        train_data_generator = BatchGeneratorManuscript(data_path,
                                                        img_height=img_height)
        test_data_generator = BatchGeneratorManuscript(data_path,
                                                       img_height=img_height,
                                                       sample_size=100,
                                                       alphabet=train_data_generator.alphabet)
    else:
        raise Exception("Data generator is not defined.")

    # model
    model = ModelOcropy(train_data_generator.alphabet, img_height)
    print(model.summary())

    # callbacks
    str_date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    callbacks = []
    if reduce_lr_on_plateau:
        callback_lr_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor='val_ctc_loss',
            factor=0.1,
            patience=4,
            verbose=1)
        callbacks.append(callback_lr_plateau)
    if levenshtein:
        callback_levenshtein = LevenshteinCallback(test_data_generator, size=10)
        callbacks.append(callback_levenshtein)
    if tensorboard:
        log_path = os.path.join("logs", str_date_time)
        callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=1, )
        callbacks.append(callback_tensorboard)
    if True:
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        checkpoints_path = os.path.join("checkpoints", str_date_time + '.hdf5')
        callback_checkpoint = keras.callbacks.ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1,
                                                              save_best_only=True, save_weights_only=True)
        callbacks.append(callback_checkpoint)

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
