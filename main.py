import argparse

from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript
from engine.managers.lr import ConstantLearningRateManager
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser("A Python command-line tool for training ocr model and using them")
    parser.add_argument('mode',
                        choices=['train'])

    args = parser.parse_args()

    if args.mode == 'train':
        img_height = 48
        epochs = 1
        train_data_generator = BatchGeneratorManuscript('data/', img_height)
        test_data_generator = BatchGeneratorManuscript('fixtures/manuscript/', img_height,
                                                       train_data_generator.alphabet)
        model = ModelOcropy(train_data_generator.alphabet, img_height)
        lr_manager = ConstantLearningRateManager(lr=0.0001)
        trainer = Trainer(model, train_data_generator, test_data_generator, lr_manager, epochs=epochs)

        trainer.train()


if __name__ == "__main__":
    main()
