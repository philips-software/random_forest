import unittest
from mpyc.runtime import mpc
from src.output import output, Secret

s = mpc.SecInt()


class OutputTest(unittest.TestCase):
    def test_output_sec_int(self):
        self.assertEqual(mpc.run(output(s(42))), 42)

    def test_output_secret(self):
        class SomeSecret(Secret):
            async def output(self):
                return 42

        self.assertEqual(mpc.run(output(SomeSecret())), 42)
