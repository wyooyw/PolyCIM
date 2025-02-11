from pipeline import run_pipeline
import polycim.op.benchmark as benchmark
import pytest
from polycim.config import CIMConfig    
@pytest.mark.parametrize("cim_row_col_celluse",
[
    (16, 2, 18),
    (16, 4, 36),
    (16, 8, 72),
])
def test_conv2d_2_8_8_3_3(cim_row_col_celluse):
    cim_row = cim_row_col_celluse[0]
    cim_col = cim_row_col_celluse[1]
    cim_celluse = cim_row_col_celluse[2]

    skew = True
    virtual_axis = not skew
    operator = benchmark.get_op_conv2d(b=1, oc=2, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=virtual_axis)
    cim_cfg = CIMConfig(
        n_row=1,
        n_group_vcol=cim_col,
        n_comp=cim_row,
        n_group=1,
        n_macro_per_group=1,
        n_macro=1
    )
    flops = operator.domain.count_val().get_num_si()
    op,cim_flops = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")
    
    mean_cell_use = flops / cim_flops

    assert mean_cell_use == cim_celluse, f"{mean_cell_use=}, {cim_celluse=}"

@pytest.mark.parametrize("cim_row_col_celluse",
[
    (16, 4, 36),
])
def test_conv3d_8_8_8_3_3_3(cim_row_col_celluse
):
    
    cim_row = cim_row_col_celluse[0]
    cim_col = cim_row_col_celluse[1]
    cim_celluse = cim_row_col_celluse[2]

    skew = True
    virtual_axis = not skew
    operator = benchmark.get_op_dwconv3d(
        ic=1, ox=4, oy=4, oz=4, kx=3, ky=3, kz=3, stride=1
    )
    cim_cfg = CIMConfig(
        n_row=1,
        n_group_vcol=cim_col,
        n_comp=cim_row,
        n_group=1,
        n_macro_per_group=1,
        n_macro=1
    )
    flops = operator.domain.count_val().get_num_si()
    op,cim_flops = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")
    
    mean_cell_use = flops / cim_flops

    assert mean_cell_use == cim_celluse, f"{mean_cell_use=}, {cim_celluse=}"