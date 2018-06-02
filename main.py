import argparse

from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser("A Python command-line tool for training ocr model and using them")
    parser.add_argument('mode',
                        choices=['train'])

    args = parser.parse_args()

    if args.mode == 'train':
        # parameters
        img_height = 48
        epochs = 1

        # data generators
        train_data_generator = BatchGeneratorIAMHandwriting('data/',
                                                            img_height=img_height)
        test_data_generator = BatchGeneratorIAMHandwriting('fixtures/iam_handwriting/',
                                                           img_height=img_height,
                                                           alphabet=train_data_generator.alphabet)

        # model
        model = ModelOcropy(train_data_generator.alphabet, img_height)

        # callbacks
        # TODO add LR callback

        # trainer
        trainer = Trainer(model, train_data_generator, test_data_generator, epochs=epochs)

        trainer.train()


if __name__ == "__main__":
    main()
