# Owner(s): ["module: nn"]

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests


class TestUpsamplingErrors(TestCase):
    def test_invalid_scale_factor_type(self):
        constructors = (
            ("Upsample", lambda: nn.Upsample(scale_factor=[1], mode="nearest")),
            ("UpsamplingNearest2d", lambda: nn.UpsamplingNearest2d(scale_factor=[1])),
            ("UpsamplingBilinear2d", lambda: nn.UpsamplingBilinear2d(scale_factor=[1])),
        )

        for name, constructor in constructors:
            with self.subTest(module=name):
                with self.assertRaisesRegex(
                    TypeError,
                    "scale_factor must be a float or a tuple of floats",
                ):
                    constructor()


if __name__ == "__main__":
    run_tests()
