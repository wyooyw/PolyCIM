import pytest
from config import get_config
import benchmark
from pipeline import run_pipeline
import tempfile

@pytest.mark.parametrize("op_config",
[
    {
        "b": 1, "oc": 32, "ic": 32, "oh": 8, "ow": 8, "kh": 3, "kw": 3, "stride": 1
    },
    {
        "b": 2, "oc": 32, "ic": 256, "oh": 8, "ow": 8, "kh": 3, "kw": 3, "stride": 1
    }
])
def test_conv2d(op_config):
    b, oc, ic, oh, ow, kh, kw, stride = op_config["b"], op_config["oc"], op_config["ic"], op_config["oh"], op_config["ow"], op_config["kh"], op_config["kw"], op_config["stride"]

    skew = False
    virtual_axis = not skew
    operator = benchmark.get_op_conv2d(b=b, oc=oc, ic=ic, oh=oh, ow=ow, kh=kh, kw=kw, stride=stride, virtual_axis=virtual_axis)
    flops = int(str(operator.domain.count_val()))
    cim_cfg = get_config()
    total_cell = cim_cfg.n_row * cim_cfg.n_group_vcol * cim_cfg.n_group

    # with tempfile.TemporaryDirectory(keep=True) as temp_dir:
    temp_dir = tempfile.mkdtemp()
    result = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=temp_dir)
    assert len(result) > 0
    assert result[0].stats is not None
    n_pim_compute = result[0].stats['pim']['pim_compute']
    avg_num_cell_use = flops / n_pim_compute
    assert avg_num_cell_use > 0 and avg_num_cell_use <= total_cell

if __name__=="__main__":
    test_conv2d({
        "b": 1, "oc": 32, "ic": 32, "oh": 8, "ow": 8, "kh": 3, "kw": 3, "stride": 1
    })