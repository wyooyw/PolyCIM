import islpy as isl
from base_operator import BasicOperator
from affine_transform import shift_to_positive
from functools import partial

domain = isl.BasicSet("{ [i,j] : 0 <= i < 4 and 0 <= j < 4 }")
skew = isl.BasicMap("{ [i,j] -> [i+j,j] }")
merge = isl.BasicMap("{ [i,j] -> [i*4+j] }")
schedule = skew.apply_range(merge)

new_domain = domain.apply(schedule)

def record_points(point, record):
    multi_val = point.get_multi_val()
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))]
    
    record.append(val[0])

def print_points(set_):
    record = []
    record_points_fn = partial(record_points, record=record)
    set_.foreach_point(record_points_fn)
    record = sorted(record)
    print(record)

print(new_domain)
print_points(new_domain)
convex_hull = new_domain.convex_hull()
print("------------------")
print(convex_hull)
print_points(convex_hull)