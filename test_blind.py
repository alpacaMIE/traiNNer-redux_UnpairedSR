from os import path as osp

from traiNNer.check.check_dependencies import check_dependencies
from traiNNer.models.pdm_sr_blind_model import PDMSRBlindModel

import test as test_entry


def build_blind_model(opt):
    method = opt.blind.method if opt.blind is not None else "pdm_sr"
    if method == "pdm_sr":
        return PDMSRBlindModel(opt)
    if method == "pdm_resshift":
        from traiNNer.models.pdm_resshift_blind_model import PDMResShiftBlindModel

        return PDMResShiftBlindModel(opt)
    raise ValueError(f"Unsupported blind.method: {method}")


if __name__ == "__main__":
    check_dependencies()
    test_entry.build_model = build_blind_model
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_entry.test_pipeline(root_path)
