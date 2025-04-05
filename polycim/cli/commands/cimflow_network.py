import json
import os
import tempfile
from dataclasses import dataclass
from functools import reduce

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from polycim.op import benchmark
from polycim.config import get_config, get_memory_base, get_memory_size, set_raw_config_by_path
from polycim.depth_first.pipeline2 import run_cimflow
import argparse
from types import SimpleNamespace
import subprocess

def get_final_code(final_code):
    # final_code = os.path.join(op.attr["BackendCompilePass"]["code_file"])
    if os.path.exists(final_code):
        with open(final_code, "r") as f:
            final_code = json.load(f)
    else:
        final_code = None

    return final_code


def get_code_set_regs(reg_and_value_list):
    code_list = []
    for reg, value in reg_and_value_list:
        code_li = {"class": 0b10, "type": 0b11, "opcode": 0, "rd": reg, "imm": value}
        code_list.append(code_li)
    return code_list


def get_code_single_send(src_addr, dst_core, data_size, unqiue_id):
    """
    通用寄存器立即数赋值指令：general-li
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为00
    - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
    """
    reg_src_addr = 0
    reg_dst_core = 1
    reg_data_size = 2
    reg_unique_id = 3

    code_set_regs = get_code_set_regs(
        [
            (reg_src_addr, src_addr),
            (reg_dst_core, dst_core),
            (reg_data_size, data_size),
            (reg_unique_id, unqiue_id),
        ]
    )

    """
    指令字段划分：
    - [31, 29]，3bit：class，指令类别码，值为110
    - [28, 27]，2bit：type，指令类型码，值为10
    - [26, 26]，1bit：sync，是否进行同步通信
    - 0：表示本次通信为同步通信，即会阻塞直至对方接收到数据
    - 1：表示本次通信为异步通信，即不会阻塞，异步通信可通过wait指令进行同步
    - [25, 21]，5bit：rs，通用寄存器1，表示传输源地址
    - [20, 16]，5bit：rd1，通用寄存器2，表示传输的目的core的编号
    - [15, 11]，5bit：rd2，通用寄存器3，表示传输的目的core的目的地址
    - [10, 6]，5bit：reg-id，通信id码，唯一标识本次传输，用于通信双方进行确认
    - [5, 0]，6bit：reserve，保留字段
    """
    code_send = {
        "class": 0b110,
        "type": 0b10,
        "sync": 0,
        "rs": reg_src_addr,
        "rd1": reg_dst_core,
        "rd2": 0,
        "reg_id": reg_unique_id,
        "reg_len": reg_data_size,
    }
    code_list = [*code_set_regs, code_send]
    return code_list


def get_code_single_receive(dst_addr, src_core, data_size, unqiue_id):
    """
    通用寄存器立即数赋值指令：general-li
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为00
    - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
    """
    reg_dst_addr = 0
    reg_src_core = 1
    reg_data_size = 2
    reg_unique_id = 3

    code_set_regs = get_code_set_regs(
        [
            (reg_dst_addr, dst_addr),
            (reg_src_core, src_core),
            (reg_data_size, data_size),
            (reg_unique_id, unqiue_id),
        ]
    )

    """
    核间数据接收指令：receive
    指令字段划分：
    - [31, 29]，3bit：class，指令类别码，值为110
    - [28, 27]，2bit：type，指令类型码，值为11
    - [26, 26]，1bit：sync，是否进行同步通信
    - 0：表示本次通信为同步通信，即会阻塞直至接收到对方发送的数据
    - 1：表示本次通信为异步通信，即不会阻塞，异步通信可通过wait指令进行同步
    - [25, 21]，5bit：rs1，通用寄存器1，表示传输的源core的编号
    - [20, 16]，5bit：rs2，通用寄存器2，表示传输的源core的源地址
    - [15, 11]，5bit：rd，通用寄存器3，表示传输的目的地址
    - [10, 6]，5bit：reg-id，通信id码，唯一标识本次传输，用于通信双方进行确认
    - [5, 0]，6bit：reserve，保留字段
    """
    code_recv = {
        "class": 0b110,
        "type": 0b11,
        "sync": 0,
        "rs1": reg_src_core,
        "rs2": 0,
        "rd": reg_dst_addr,
        "reg_id": reg_unique_id,
        "reg_len": reg_data_size,
    }
    code_list = [*code_set_regs, code_recv]
    return code_list


def get_code_trans(src_addr, dst_addr, data_size):
    """
    核内数据传输指令：trans（copy）
    指令字段划分：
    - [31, 29]，3bit：class，指令类别码，值为110
    - [28, 28]，1bit：type，指令类型码，值为0
    - [27, 26]，1bit：offset mask，偏移值掩码，0表示该地址不使用偏移值，1表示使用偏移值
    - [27]，1bit：source offset mask，源地址偏移值掩码
    - [26]，1bit：destination offset mask，目的地址偏移值掩码
    - [25, 21]，5bit：rs1，通用寄存器1，表示传输源地址的基址
    - [20, 16]，5bit：rs2，通用寄存器2，表示传输数据的字节大小
    - [15, 11]，5bit：rd，通用寄存器3，表示传输目的地址的基址
    - [10, 0]，11bit：offset，立即数，表示寻址的偏移值
    - 源地址计算公式：$rs + offset * [27]
    - 目的地址计算公式：$rd + offset * [26]
    """
    reg_src_addr = 0
    reg_dst_addr = 1
    reg_data_size = 2
    code_set_regs = get_code_set_regs(
        [
            (reg_src_addr, src_addr),
            (reg_dst_addr, dst_addr),
            (reg_data_size, data_size),
        ]
    )
    code_trans = {
        "class": 0b110,
        "type": 0,
        "offset_mask": 0b00,
        "rs1": reg_src_addr,
        "rs2": reg_data_size,
        "rd": reg_dst_addr,
        "offset": 0,
    }
    return [*code_set_regs, code_trans]


def get_code_trans_by_memory_type(src_memory_type, dst_memory_type, data_size):
    src_addr = get_memory_base(src_memory_type)
    dst_addr = get_memory_base(dst_memory_type)
    return get_code_trans(src_addr, dst_addr, data_size)


def get_code_read_global(attr):
    assert "tensor_type" in attr
    if attr["tensor_type"] == "weight":
        dst_memory_type = "macro"
    elif attr["tensor_type"] == "feature":
        dst_memory_type = "input_memory"
    else:
        assert False, attr["tensor_type"]

    return get_code_trans_by_memory_type(
        "global", dst_memory_type, reduce(lambda x, y: x * y, attr["shape"])
    )


def get_code_write_global(attr):
    return get_code_trans_by_memory_type(
        "output_memory", "global", reduce(lambda x, y: x * y, attr["shape"])
    )


def get_code_add(attr):
    """
    SIMD计算：SIMD-compute
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为01
    - [29, 28]，2bit：input num，input向量的个数，范围是1到4
    - 00：1个输入向量
    - 01：2个输入向量
    - 10：3个输入向量
    - 11：4个输入向量
    - 备注：input 1和input 2的地址由寄存器rs1和rs2给出，input 3和input 4的地址由专用寄存器给出
    - [27, 20]，8bit：opcode，操作类别码，表示具体计算的类型
    - 0x00：add，向量加法
    - 0x01：add-scalar，向量和标量加法
    - 0x02：multiply，向量逐元素乘法
    - 0x03：quantify，量化
    - 0x04：quantify-resadd，resadd量化
    - 0x05：quantify-multiply，乘法量化
    - [19, 15]，5bit：rs1，通用寄存器1，表示input 1的起始地址
    - [14, 10]，5bit：rs2，通用寄存器2，表示input 2的起始地址
    - [9, 5]，5bit：rs3，通用寄存器3，表示input向量长度
    - [4, 0]，5bit：rd，通用寄存器4，表示output写入的起始地址
    使用的专用寄存器：
    - 16：input 1 bit width：输入向量1每个元素的bit长度
    - 17：input 2 bit width：输入向量2每个元素的bit长度
    - 18：input 3 bit width：输入向量3每个元素的bit长度
    - 19：input 4 bit width：输入向量4每个元素的bit长度
    - 20：output bit width：输出向量每个元素的bit长度
    - 21：input 3 address：input 3的起始地址
    - 22：input 4 address：input 4的起始地址
    """
    data_size = reduce(lambda x, y: x + y, attr["shape"])
    reg_input_1_addr = 1
    reg_input_2_addr = 2
    reg_data_size = 3
    reg_output_addr = 4
    code_set_regs = get_code_set_regs(
        [
            (reg_input_1_addr, get_memory_base("input_memory")),
            (reg_input_2_addr, get_memory_base("input_memory") + data_size),
            (reg_data_size, data_size),
            (reg_output_addr, get_memory_base("output_memory")),
        ]
    )
    code_add = {
        "class": 0b01,
        "input_num": 0b01,
        "opcode": 0b00,
        "rs1": reg_input_1_addr,
        "rs2": reg_input_2_addr,
        "rs3": reg_data_size,
        "rd": reg_output_addr,
    }
    return [*code_set_regs, code_add]

    # exit()


cache_conv2d_result = dict()


def get_code_conv2d(args, attr, cache_dir):
    global cache_conv2d_result
    """
    'X_shape': [1, 32, 224, 224], 'W_shape': [32, 3, 3, 3], 'padding': [1, 1, 1, 1], 'strides': [2, 2]
    """
    # import pdb; pdb.set_trace()
    batch, in_channel, in_height, in_width = attr["X_shape"]
    out_channel, in_channel, kernel_height, kernel_width = attr["W_shape"]
    padding = attr["padding"][0]
    stride = attr["strides"][0]

    out_h = (in_height + 2 * padding - kernel_height) // stride + 1
    out_w = (in_width + 2 * padding - kernel_width) // stride + 1

    cache_key = (
        batch,
        out_channel,
        in_channel,
        out_h,
        out_w,
        kernel_height,
        kernel_width,
        padding,
        stride,
    )
    if cache_key in cache_conv2d_result:
        result = cache_conv2d_result[cache_key]
    else:
        # with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = tempfile.mkdtemp(dir=cache_dir)
        operator = benchmark.get_op_conv2d(
            b=batch,
            oc=out_channel,
            ic=in_channel,
            oh=out_h,
            ow=out_w,
            kh=kernel_height,
            kw=kernel_width,
            stride=stride,
            virtual_axis=True,
        )
        operator_compile_args = {
            "config_path": args.config_path,
            "output_path": temp_dir,
            "data_movement_full_vectorize": True,
            "cimflow": True,
            "verify": args.verify,
        }
        operator_compile_args = SimpleNamespace(**operator_compile_args)
        # import pdb; pdb.set_trace()
        operator.set_attr("name", "conv")
        result = run_cimflow(
            args=operator_compile_args,
            cim_config=get_config(),
            op=operator
        )
        assert len(result) == 1, f"Fail when generating conv2d code. {attr=}"
        result = result[0]
        if args.verify:
            assert result.attr["VerifyPass"]["check_result"], f"Fail when generating conv2d code. {attr=}"
        cache_conv2d_result[cache_key] = result

    # read code from result
    final_code = get_final_code(os.path.join(result.attr["BackendCompilePass"]["code_file"]))
    assert final_code is not None, f"Fail when generating conv2d code. {attr=}"

    return final_code


def fill_template(src_path, dst_path, context):

    src_folder, src_file = os.path.split(src_path)

    # 创建 Jinja2 环境和加载器
    env = Environment(loader=FileSystemLoader(src_folder), undefined=StrictUndefined)

    # 加载模板
    template = env.get_template(src_file)

    # 渲染模板
    output = template.render(context)

    with open(dst_path, "w") as f:
        f.write(output)


@dataclass
class ProfileResult:
    stats: str
    save_path: int


cache_dwconv2d_result = dict()


def get_dwcode_conv2d(args, attr):
    global cache_dwconv2d_result
    POLYCIM_HOME = os.environ.get("POLYCIM_HOME")
    template_path = (
        os.path.join(POLYCIM_HOME, "polycim/template/depthwise_conv.cim")
    )
    temp_dir = tempfile.mkdtemp()
    code_path = os.path.join(temp_dir, "depthwise_conv.cim")

    X_size = reduce(lambda x, y: x * y, attr["X_shape"])
    assert X_size <= get_memory_size("input_memory"), f"{X_size=}"

    batch, _, in_height, in_width = attr["X_shape"]
    in_channel, _, kernel_height, kernel_width = attr["W_shape"]

    padding = 1
    stride = 1

    out_h = (in_height + 2 * padding - kernel_height) // stride + 1
    out_w = (in_width + 2 * padding - kernel_width) // stride + 1

    cache_key = (
        batch,
        in_channel,
        out_h,
        out_w,
        kernel_height,
        kernel_width,
        padding,
        stride,
    )
    if cache_key in cache_dwconv2d_result:
        result = cache_dwconv2d_result[cache_key]
    else:
        context = {
            "INPUT_ROW": in_height,
            "INPUT_COL": in_width,
            "INPUT_CHANNEL": in_channel,
            "OUTPUT_ROW": out_h,
            "OUTPUT_COL": out_w,
            "OUTPUT_CHANNEL": in_channel,
            "KERNEL_SIZE": kernel_height,
            "STRIDE": stride,
            "BATCH": batch,
        }

        fill_template(template_path, code_path, context)

        # backend_compile_cmd = (
        #     f"input_file={code_path} output_path={temp_dir} bash run.sh "
        # )
        # os.system(backend_compile_cmd)
        subprocess.run([
            "cim-compiler", "compile",
            "--input-file", code_path,
            "--output-dir", temp_dir,
            "--config-file", args.config_path
        ], check=True)

        result = ProfileResult(stats=None, save_path=temp_dir)
        cache_dwconv2d_result[cache_key] = result
    # import pdb; pdb.set_trace()
    final_code = get_final_code(os.path.join(result.save_path, "depthwise_conv.cim"))
    assert final_code is not None
    # import pdb; pdb.set_trace()
    return final_code


def get_core_row_and_col_from_core_name(core_name):
    core_row, core_col = core_name.split("_")[1:]
    return int(core_row), int(core_col)


core_max_row = None
core_max_col = None


def get_max_core_row_col(core_name_list):
    global core_max_row
    global core_max_col
    row, col = get_core_row_and_col_from_core_name(core_name_list[0])
    for core_name in core_name_list:
        core_row, core_col = get_core_row_and_col_from_core_name(core_name)
        row = max(row, core_row)
        col = max(col, core_col)
    core_max_row = row
    core_max_col = col
    return row, col


def get_core_id_from_core_name(core_name):
    global core_max_col
    """
    core_name: core_{row}_{col}
    """
    # parse row and col from core_name
    row, col = core_name.split("_")[1:]
    core_id = int(row) * (core_max_col + 1) + int(col)
    return core_id


unique_name_to_unqiue_id = dict()
max_unique_id = 0


def get_unique_id_from_unique_name(unique_name):
    global unique_name_to_unqiue_id
    global max_unique_id
    if unique_name in unique_name_to_unqiue_id:
        return unique_name_to_unqiue_id[unique_name]
    else:
        unique_name_to_unqiue_id[unique_name] = max_unique_id
        max_unique_id += 1
        return unique_name_to_unqiue_id[unique_name]


def parse_instructions(args, core_name, stage_id, instructions, cache_dir):
    code_list = []
    max_data_size = 0
    max_shape = None
    for idx, instruction in enumerate(instructions):
        # if not (core_name=="core_0_1" and stage_id=="0" and instruction["op"] == "conv" and idx==2):
        #     continue
        # import pdb; pdb.set_trace()
        if instruction["op"] == "read":
            code = get_code_read_global(instruction["attr"])
        elif instruction["op"] == "write":
            code = get_code_write_global(instruction["attr"])
        elif instruction["op"] == "conv":
            code = get_code_conv2d(args, instruction["attr"], cache_dir)
        elif instruction["op"] == "depthwise_conv":
            code = get_dwcode_conv2d(args, instruction["attr"])
        elif instruction["op"] in ("send", "send_ring"):
            data_size = reduce(lambda x, y: x * y, instruction["attr"]["shape"])
            assert data_size <= get_memory_size("output_memory"), f"{data_size=}"
            code = get_code_single_send(
                src_addr=get_memory_base("output_memory"),
                dst_core=get_core_id_from_core_name(
                    instruction["attr"]["dist_core_name"]
                ),
                data_size=reduce(lambda x, y: x * y, instruction["attr"]["shape"]),
                unqiue_id=get_unique_id_from_unique_name(instruction["attr"]["name"]),
            )
        elif instruction["op"] in ("receive", "receive_ring"):
            data_size = reduce(lambda x, y: x * y, instruction["attr"]["shape"])
            assert data_size <= get_memory_size("input_memory"), f"{data_size=}"
            code = get_code_single_receive(
                dst_addr=get_memory_base("input_memory"),
                src_core=get_core_id_from_core_name(
                    instruction["attr"]["src_core_name"]
                ),
                data_size=reduce(lambda x, y: x * y, instruction["attr"]["shape"]),
                unqiue_id=get_unique_id_from_unique_name(instruction["attr"]["name"]),
            )
        elif instruction["op"] == "add":
            code = get_code_add(instruction["attr"])
        else:
            assert False, f"{instruction=}"

        assert code is not None, f"{instruction=}"
        code_list.extend(code)
    return code_list


def get_special_reg_set_code():
    """
    专用寄存器立即数赋值指令：special-li
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为01
    - [25, 21]，5bit：rd，专用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
    """
    code_simd_add1 = {"class": 0b10, "type": 0b11, "opcode": 0b01, "rd": 16, "imm": 8}
    code_simd_add2 = {"class": 0b10, "type": 0b11, "opcode": 0b01, "rd": 17, "imm": 8}
    code_simd_out = {"class": 0b10, "type": 0b11, "opcode": 0b01, "rd": 20, "imm": 8}
    return [code_simd_add1, code_simd_add2, code_simd_out]


def parse_noc_tasks(args, json_path, code_save_path, cache_dir):
    """
    {
    "core_0_0": {
        "stages": {
            "0": {
                "cluster_id": "Conv_0_quant",
                "weight_replica_id": 3,
                "instructions": [
                    {
                        "op": "read",
                        "attr": {
                            "shape": [32,3,3,3]
                        }
                    },
                    {
                        "op": "read",
                        "attr": {
                            "shape": [1,4,224,224]
                        }
                    },
                    {
    """
    os.makedirs(code_save_path, exist_ok=True)

    with open(json_path, "r") as f:
        tasks = json.load(f)

    get_max_core_row_col(list(tasks.keys()))

    total_save_files = []
    for core_name, stages in tasks.items():
        code_list = []

        code_list.extend(get_special_reg_set_code())

        for stage_id, stage in stages["stages"].items():
            code = parse_instructions(args, core_name, stage_id, stage["instructions"], cache_dir)
            code_list.extend(code)

        # padding, forbidden branch code be last codes
        code_list.extend(get_special_reg_set_code())

        # import pdb; pdb.set_trace()

        core_id = get_core_id_from_core_name(core_name)
        core_code_save_path = os.path.join(code_save_path, f"{core_id}.json")
        total_save_files.append(core_code_save_path)
        with open(core_code_save_path, "w") as f:
            json.dump(code_list, f, indent=2)

    return total_save_files


def tidy_json_format(save_path, total_save_files):
    total_code_str = "{"
    for core_idx, file_path in enumerate(total_save_files):
        file_name_with_extension = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
        assert str(core_idx) == file_name_without_extension

        with open(file_path, "r") as f:
            code_list = json.load(f)

        code_str = "[\n"
        for idx, code in enumerate(code_list):
            code_str += json.dumps(code, separators=(",", ":"))
            if idx < len(code_list) - 1:
                code_str += ","
            code_str += "\n"
        code_str += "]"

        total_code_str += f'"{core_idx}":' + code_str
        if core_idx < len(total_save_files) - 1:
            total_code_str += ","
        total_code_str += "\n"
    total_code_str += "}"

    with open(save_path, "w") as f:
        f.write(total_code_str)

def parse_cimflow_network_args(subparsers):
    parser = subparsers.add_parser('cimflow_network')
    parser.add_argument("--read-json", "-i", type=str, help="read path")
    parser.add_argument("--save-dir", "-o", type=str, help="save path")
    parser.add_argument("--config-path", "-c", type=str, help="config path")
    parser.add_argument("--verify", action="store_true", help="verify")

def run_cimflow_network(args):
    set_raw_config_by_path(args.config_path)
    each_core_save_dir = os.path.join(args.save_dir, "each_core")
    with tempfile.TemporaryDirectory() as cache_dir:
        total_save_files = parse_noc_tasks(
            args,
            args.read_json, 
            each_core_save_dir,
            cache_dir
        )

    save_file_name = os.path.basename(args.read_json)
    save_file_name = "isa_" + save_file_name
    all_core_code_path = os.path.join(args.save_dir, save_file_name)
    tidy_json_format(
        all_core_code_path,
        total_save_files=total_save_files,
    )

    print("Finish!")
    print(f"{each_core_save_dir = }")
    print(f"{all_core_code_path = }")

# if __name__ == "__main__":
#     main()
