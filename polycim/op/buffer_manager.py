from polycim.config import get_memory_names
from dataclasses import dataclass
import copy
import islpy as isl
from polycim.op.base_operator import (DataMovement, DataMovementOperator,
                           TensorAccessRelation)

@dataclass
class BufferInfo:
    name: str
    shape: list
    memory_name: str

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
        self.valid_memory_names = set(get_memory_names())

    def add_buffer(self, name, shape, memory_name):
        if name in self.buffer_name_to_info:
            raise ValueError(f"{name} already exists")  
        if memory_name not in self.valid_memory_names:
            raise ValueError(f"{memory_name} is not a valid memory name")
        self.buffer_name_to_info[name] = BufferInfo(name, shape, memory_name)

    def add_buffer_info(self, buffer_info):
        if buffer_info.name in self.buffer_name_to_info:
            raise ValueError(f"{buffer_info.name} already exists")  
        if buffer_info.memory_name not in self.valid_memory_names:
            raise ValueError(f"{buffer_info.memory_name} is not a valid memory name")
        self.buffer_name_to_info[buffer_info.name] = buffer_info

    def add_buffers_from_op(self, op):

        buffer_to_size = dict()
        buffer_to_memory_name = dict()

        def _update_shape(name, shape):
            if name not in buffer_to_size:
                buffer_to_size[name] = shape
            else:
                old_shape = buffer_to_size[name]
                assert len(old_shape) == len(shape), f"{old_shape=}, {shape=}"
                max_shape = [max(old_shape[i], shape[i]) for i in range(len(shape))]
                buffer_to_size[name] = max_shape

        def _update_memory_name(name, memory_name):
            if name not in buffer_to_memory_name:
                buffer_to_memory_name[name] = memory_name
            else:
                assert (
                    buffer_to_memory_name[name] == memory_name
                ), f"{buffer_to_memory_name[name]=}, {memory_name=}"

        # import pdb; pdb.set_trace()
        I_name, I_shape = get_name_and_shape(op.access_I)  # this maybe incorrect.
        W_name, W_shape = get_name_and_shape(op.access_W)
        O_name, O_shape = get_name_and_shape(op.access_O)

        _update_shape(I_name, I_shape)
        _update_shape(W_name, W_shape)
        _update_shape(O_name, O_shape)

        _update_memory_name(I_name, op.access_I.memory_name)
        _update_memory_name(W_name, op.access_W.memory_name)
        _update_memory_name(O_name, op.access_O.memory_name)

        for buffer in ["I", "W", "O"]:
            for data_movement in op.data_movement[buffer]:
                assert type(data_movement) == DataMovement

                name, shape = get_name_and_shape(data_movement.access_I)
                _update_shape(name, shape)
                _update_memory_name(name, data_movement.access_I.memory_name)

                name, shape = get_name_and_shape(data_movement.access_O)
                _update_shape(name, shape)
                _update_memory_name(name, data_movement.access_O.memory_name)

        # buffer_name_to_info = dict()
        for name in buffer_to_size.keys():
            shape = buffer_to_size[name]
            memory_name = buffer_to_memory_name[name]
            self.add_buffer(name, shape, memory_name)
        #     buffer_name_to_info[name] = BufferInfo(
        #         name=name, shape=shape, memory_name=memory_name
        #     )
        # return buffer_name_to_info

    def get_buffers_by_memory_name(self, memory_name):
        return [buffer_info for buffer_info in self.buffer_name_to_info.values() if buffer_info.memory_name == memory_name]

    def get_buffer_by_memory_name(self, memory_name):
        buffers = self.get_buffers_by_memory_name(memory_name)
        if len(buffers) == 0:
            raise ValueError(f"{memory_name} has no buffers")
        if len(buffers) > 1:
            raise ValueError(f"{memory_name} has multiple buffers")
        return buffers[0]

    def get_buffer_name_to_info(self):
        return self.buffer_name_to_info