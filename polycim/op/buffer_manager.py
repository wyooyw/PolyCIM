from polycim.config import get_memory_types
from dataclasses import dataclass
import copy
import islpy as isl
from polycim.op.base_operator import (DataMovement, DataMovementOperator,
                           TensorAccessRelation)

@dataclass
class BufferInfo:
    name: str
    shape: list
    memory_type: str

def get_name_and_shape(access):
    sizes = access.sizes.range()
    sizes = [sizes.dim_max_val(i) for i in range(sizes.dim(isl.dim_type.set))]

    offsets = access.offsets.range()
    offsets = [offsets.dim_max_val(i) for i in range(offsets.dim(isl.dim_type.set))]

    shape = [sizes + offsets for sizes, offsets in zip(sizes, offsets)]

    name = access.offsets.get_tuple_name(isl.dim_type.out)
    return name, shape

class BufferManager:
    def __init__(self):
        self.buffer_name_to_info = dict()
        self.valid_memory_types = set(get_memory_types())

    def add_buffer(self, name, shape, memory_type):
        if name in self.buffer_name_to_info:
            raise ValueError(f"{name} already exists")  
        if memory_type not in self.valid_memory_types:
            raise ValueError(f"{memory_type} is not a valid memory name")
        self.buffer_name_to_info[name] = BufferInfo(name, shape, memory_type)

    def add_buffer_info(self, buffer_info):
        if buffer_info.name in self.buffer_name_to_info:
            raise ValueError(f"{buffer_info.name} already exists")  
        if buffer_info.memory_type not in self.valid_memory_types:
            raise ValueError(f"{buffer_info.memory_type} is not a valid memory name")
        self.buffer_name_to_info[buffer_info.name] = buffer_info

    def add_buffers_from_op(self, op):

        buffer_to_size = dict()
        buffer_to_memory_type = dict()

        def _update_shape(name, shape):
            if name not in buffer_to_size:
                buffer_to_size[name] = shape
            else:
                old_shape = buffer_to_size[name]
                assert len(old_shape) == len(shape), f"{old_shape=}, {shape=}"
                max_shape = [max(old_shape[i], shape[i]) for i in range(len(shape))]
                buffer_to_size[name] = max_shape

        def _update_memory_type(name, memory_type):
            if name not in buffer_to_memory_type:
                buffer_to_memory_type[name] = memory_type
            else:
                assert (
                    buffer_to_memory_type[name] == memory_type
                ), f"{buffer_to_memory_type[name]=}, {memory_type=}"

        # import pdb; pdb.set_trace()
        I_name, I_shape = get_name_and_shape(op.access_I)  # this maybe incorrect.
        W_name, W_shape = get_name_and_shape(op.access_W)
        O_name, O_shape = get_name_and_shape(op.access_O)

        _update_shape(I_name, I_shape)
        _update_shape(W_name, W_shape)
        _update_shape(O_name, O_shape)

        _update_memory_type(I_name, op.access_I.memory_type)
        _update_memory_type(W_name, op.access_W.memory_type)
        _update_memory_type(O_name, op.access_O.memory_type)

        for buffer in ["I", "W", "O"]:
            for data_movement in op.data_movement[buffer]:
                assert type(data_movement) == DataMovement

                name, shape = get_name_and_shape(data_movement.access_I)
                _update_shape(name, shape)
                _update_memory_type(name, data_movement.access_I.memory_type)

                name, shape = get_name_and_shape(data_movement.access_O)
                _update_shape(name, shape)
                _update_memory_type(name, data_movement.access_O.memory_type)

        # buffer_name_to_info = dict()
        for name in buffer_to_size.keys():
            shape = buffer_to_size[name]
            memory_type = buffer_to_memory_type[name]
            self.add_buffer(name, shape, memory_type)
        #     buffer_name_to_info[name] = BufferInfo(
        #         name=name, shape=shape, memory_type=memory_type
        #     )
        # return buffer_name_to_info

    def get_buffers_by_memory_type(self, memory_type):
        return [buffer_info for buffer_info in self.buffer_name_to_info.values() if buffer_info.memory_type == memory_type]

    def get_buffer_by_memory_type(self, memory_type):
        buffers = self.get_buffers_by_memory_type(memory_type)
        if len(buffers) == 0:
            raise ValueError(f"{memory_type} has no buffers")
        if len(buffers) > 1:
            raise ValueError(f"{memory_type} has multiple buffers")
        return buffers[0]

    def get_buffer_name_to_info(self):
        return self.buffer_name_to_info