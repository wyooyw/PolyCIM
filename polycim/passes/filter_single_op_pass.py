from polycim.passes.base import BreadthFirstPass


class FilterSingleOpPass(BreadthFirstPass):
    def __init__(self):
        super().__init__()
        self.op_list = list()

    def apply(self, operator):
        self.op_list.append(operator)

    def apply_all(self):
        pass

    def get_result(self):
        return self.op_list[:1]
