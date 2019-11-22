import unittest

from src.output import Secret
from src.secint import secint as s
from tests.reveal import reveal


class OutputTest(unittest.TestCase):
    def test_output_sec_int(self):
        self.assertEqual(reveal(s(42)), 42)

    def test_output_secret(self):
        self.assertEqual(reveal(SomeSecret()), 42)

    def test_output_list_of_secrets(self):
        self.assertEqual(reveal([SomeSecret(), SomeSecret()]), [42, 42])


class SomeSecret(Secret):
    async def __output__(self):
        return 42
