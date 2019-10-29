import unittest

from src.output import Secret, output
from src.secint import secint as s
from tests.reveal import reveal


class OutputTest(unittest.TestCase):
    def test_output_sec_int(self):
        self.assertEqual(reveal(s(42)), 42)

    def test_output_secret(self):
        class SomeSecret(Secret):
            async def __output__(self):
                return 42

        self.assertEqual(reveal(SomeSecret()), 42)
