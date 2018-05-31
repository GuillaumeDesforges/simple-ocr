from unittest import TestCase

from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript


class TestBatchGeneratorManuscript(TestCase):
    def setUp(self):
        self.generator = BatchGeneratorManuscript('fixtures/data/manuscript')

    def test_len(self):
        self.assertEqual(len(self.generator) == 2)
