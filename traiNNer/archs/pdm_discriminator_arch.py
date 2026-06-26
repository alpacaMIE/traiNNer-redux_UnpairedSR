from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class PDMPatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        nf: int = 64,
        nb: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(in_c, nf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, nb):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence.extend(
                [
                    nn.Conv2d(
                        nf * nf_mult_prev,
                        nf * nf_mult,
                        kernel_size=kw,
                        stride=stride,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(nf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**nb, 8)
        sequence.extend(
            [
                nn.Conv2d(
                    nf * nf_mult_prev,
                    nf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        )

        sequence.append(
            nn.Conv2d(nf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        )
        self.model = nn.Sequential(*sequence)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
