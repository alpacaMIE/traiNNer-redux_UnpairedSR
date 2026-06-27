from copy import deepcopy

from traiNNer.models.base_model import BaseModel
from traiNNer.models.pdm_sr_blind_model import PDMSRBlindModel
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions

__all__ = ["build_model"]


def build_model(opt: ReduxOptions) -> BaseModel:
    """Build the unpaired (blind / PDM) super-resolution model from options.

    Args:
        opt (ReduxOptions): Configuration. The blind method is selected via
            ``opt.blind.method`` ("pdm_sr" by default, or "pdm_resshift").
    """
    opt = deepcopy(opt)
    logger = get_root_logger()

    method = opt.blind.method if opt.blind is not None else "pdm_sr"
    if method == "pdm_resshift":
        from traiNNer.models.pdm_resshift_blind_model import PDMResShiftBlindModel

        model: BaseModel = PDMResShiftBlindModel(opt)
    elif method == "pdm_sr":
        model = PDMSRBlindModel(opt)
    else:
        raise ValueError(f"Unsupported blind.method: {method}")

    logger.info(
        "Model [bold]%s[/bold] is created.",
        model.__class__.__name__,
        extra={"markup": True},
    )
    return model
