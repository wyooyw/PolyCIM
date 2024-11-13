import utils
import islpy as isl

class AccessRelation:
    def __init__(self, offsets):
        assert type(offsets) in (isl.BasicMap, isl.Map), f"{type(offsets)}"
        self.offsets = offsets

    def __repr__(self):
        return f"{self.offsets}"

class TensorAccessRelation(AccessRelation):
    def __init__(self, offsets, sizes):
        super().__init__(offsets)
        self.sizes = sizes

class BasicOperator:
    def __init__(self, domain, access_I, access_O, access_W, history_domains=list(), history_schedules=list()):
        assert type(domain) == isl.BasicSet
        assert type(access_I) == isl.BasicMap
        assert type(access_O) == isl.BasicMap
        assert type(access_W) == isl.BasicMap
        self.domain = domain
        self.access_I = access_I
        self.access_O = access_O
        self.access_W = access_W
        self.history_domains = history_domains
        self.history_schedules = history_schedules

        self.rename_domain_and_access()

    def rename_domain_and_access(self):
        self.domain = utils.rename_all_dims_for_basic_set(self.domain)
        self.access_I = utils.rename_all_dims_for_basic_map(self.access_I)
        self.access_O = utils.rename_all_dims_for_basic_map(self.access_O)
        self.access_W = utils.rename_all_dims_for_basic_map(self.access_W)

    def apply_schedule(self, schedule, skip_simplify=False):
        assert type(schedule)==isl.BasicMap, f"{type(schedule)}"

        # transform by scheudle
        concrete_schedule = schedule.intersect_domain(self.domain)
        assert concrete_schedule.reverse().is_single_valued(), f"{concrete_schedule.reverse()}"
        domain = concrete_schedule.range()

        access_I = concrete_schedule.reverse().apply_range(self.access_I)
        access_O = concrete_schedule.reverse().apply_range(self.access_O)
        access_W = concrete_schedule.reverse().apply_range(self.access_W)

        assert type(access_I)==isl.BasicMap, f"{type(access_I)}"
        assert type(access_O)==isl.BasicMap, f"{type(access_I)}"
        assert type(access_W)==isl.BasicMap, f"{type(access_I)}"

        if not access_O.is_single_valued():
            import pdb; pdb.set_trace()

        if not skip_simplify:
            access_I = utils.simplify_basic_map(access_I)
            access_O = utils.simplify_basic_map(access_O)
            access_W = utils.simplify_basic_map(access_W)

        return BasicOperator(
            domain=domain,
            access_I=access_I,
            access_O=access_O,
            access_W=access_W,
            history_domains=[*self.history_domains, self.domain],
            history_schedules=[*self.history_schedules, schedule]
        )

    def get_access_by_name(self, buffer_name):
        if buffer_name=="I":
            return self.access_I
        elif buffer_name=="O":
            return self.access_O
        elif buffer_name=="W":
            return self.access_W
        else:
            raise Exception(f"Unknown buffer name: {buffer_name}")
        

    def concrete_access(self, access):
        concrete_access = access.intersect_domain(self.domain)
        return concrete_access

    def concrete_access_I(self):
        return self.concrete_access(self.access_I)
    
    def concrete_access_O(self):
        return self.concrete_access(self.access_O)

    def concrete_access_W(self):
        return self.concrete_access(self.access_W)


class DataMovement:
    def __init__(self, domain, access_I, access_O, level):
        assert type(access_I) in (isl.BasicMap, isl.Map, AccessRelation, TensorAccessRelation)
        assert type(access_O) in (isl.BasicMap, isl.Map, AccessRelation, TensorAccessRelation)

        self.domain = domain
        self.access_I = AccessRelation(access_I) if type(access_I) in (isl.BasicMap, isl.Map) else access_I
        self.access_O = AccessRelation(access_O) if type(access_O) in (isl.BasicMap, isl.Map) else access_O
        self.level = level

class DataMovementOperator:
    def __init__(self, domain, access_I, access_O, access_W, 
        history_domains=list(), 
        history_schedules=list(),
        data_movement = None
        ):
        assert type(domain) in (isl.BasicSet, isl.Set)
        assert type(access_I) in (isl.BasicMap, isl.Map, AccessRelation, TensorAccessRelation)
        assert type(access_O) in (isl.BasicMap, isl.Map, AccessRelation, TensorAccessRelation)
        assert type(access_W) in (isl.BasicMap, isl.Map, AccessRelation, TensorAccessRelation)

        self.domain = domain
        self.access_I = AccessRelation(access_I) if type(access_I) in (isl.BasicMap, isl.Map) else access_I
        self.access_O = AccessRelation(access_O) if type(access_O) in (isl.BasicMap, isl.Map) else access_O
        self.access_W = AccessRelation(access_W) if type(access_W) in (isl.BasicMap, isl.Map) else access_W
        self.history_domains = history_domains
        self.history_schedules = history_schedules

        if data_movement is None:
            self.data_movement = {"I": list(), "O": list(), "W": list()}
        else:
            assert isinstance(data_movement, dict)
            self.data_movement = data_movement

    def insert_buffer(self, buffer_name, _data_movement):
        assert buffer_name in ["I", "O", "W"]
        self.data_movement[buffer_name].append(_data_movement)