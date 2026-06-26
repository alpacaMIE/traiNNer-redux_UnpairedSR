import torch
import torch.nn.functional as F

from traiNNer.models.pdm_sr_blind_model import PDMSRBlindModel
from traiNNer.utils.redux_options import ReduxOptions


class PDMResShiftBlindModel(PDMSRBlindModel):
    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

    def _sr_forward(self) -> None:
        assert self.syn_hr is not None

        if (not self.optim_deg) or self.fake_real_lr is None:
            self.fake_real_lr, self.predicted_kernel, self.predicted_noise = (
                self.net_deg(self.syn_hr)
            )

        self.fake_real_lr_quant = self.quant(self.fake_real_lr)
        lr_rgb = self.fake_real_lr_quant.detach()
        lr_up = F.interpolate(
            lr_rgb,
            size=self.syn_hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        net_g = self.net_g.module if hasattr(self.net_g, "module") else self.net_g

        t = torch.randint(
            1, net_g.num_timesteps + 1, (self.syn_hr.shape[0],), device=self.device
        )
        noise = torch.randn_like(self.syn_hr)
        x_t = net_g.q_sample(self.syn_hr, lr_up, t, noise)
        x0_pred = net_g.predict_x0(x_t, t, lr_up)

        self.syn_sr = x0_pred
        self.output = self.syn_sr
