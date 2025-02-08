import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from base_operator import BasicOperator
import islpy as isl
def plot_3d_scene(points, arrows):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制小球
    for point in points:
        x, y, z, color = point
        ax.scatter(x, y, z, c=color, s=100)  # s 是大小

    # 绘制箭头
    for arrow in arrows:
        start, end, color = arrow
        ax.quiver(start[0], start[1], start[2], 
                  end[0] - start[0], end[1] - start[1], end[2] - start[2], 
                  color=color, length=1, arrow_length_ratio=0.1)

    # 自适应坐标轴
    all_points = np.array([p[:3] for p in points] + [a[0] for a in arrows] + [a[1] for a in arrows])
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])

    # 计算统一的刻度间隔
    max_range = max(max_vals - min_vals)
    tick_interval = max_range / 10  # 选择合适的刻度间隔

    # 设置刻度间隔
    ax.set_xticks(np.arange(min_vals[0], max_vals[0] + tick_interval, tick_interval))
    ax.set_yticks(np.arange(min_vals[1], max_vals[1] + tick_interval, tick_interval))
    ax.set_zticks(np.arange(min_vals[2], max_vals[2] + tick_interval, tick_interval))

    # 设置坐标轴标签和字号
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # 启用拖动/缩放
    plt.show()

# # 示例数据
# points = [
#     (1, 2, 3, 'r'),  # (x, y, z, color)
#     (4, 5, 6, 'g'),
#     (7, 8, 9, 'b')
# ]

# arrows = [
#     ((1, 2, 3), (4, 5, 6), 'r'),  # (start, end, color)
#     ((4, 5, 6), (7, 8, 9), 'g')
# ]

# plot_3d_scene(points, arrows)

def point_to_list(point, return_int=False):
    ls = []
    multi_val = point.get_multi_val()
    for i in range(len(multi_val)):
        if return_int:
            ls.append(int(str(multi_val.get_val(i))))
        else:
            ls.append(str(multi_val.get_val(i)))
    return ls

def operator_to_domain_input_points(operator):
    domain = operator.domain
    access_I = operator.access_I
    domain_input_points = []
    def get_input_points_from_domain_point(domain_point):
        domain_point_list = point_to_list(domain_point, return_int=True)
        domain_point_set = isl.Set(f"{{ [{','.join(point_to_list(domain_point))}] }}")
        
        input_point = []
        access_I.intersect_domain(domain_point_set).range().foreach_point(
            lambda point: input_point.append(point)
        )
        assert len(input_point) == 1
        input_point_list = point_to_list(input_point[0], return_int=True)
        domain_input_points.append((tuple(domain_point_list), tuple(input_point_list)))

    domain.foreach_point(get_input_points_from_domain_point)
    return domain_input_points

color_list = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF8000",
    "#8000FF",
    "#FF0080",
    "#0080FF",
    "#00FF80",
    "#80FF00",
    "#FF80FF",
    "#80FFFF",
    "#FF8080",
    "#80FF80",
    "#8080FF",
    "#FF8080",
]
def num_to_color(num):
    return color_list[num % len(color_list)]

def main():
    operator = BasicOperator(
        domain = isl.BasicSet(
            f"{{ [i,k]: 0<=i<56 and 0<=k<5 }}"
        ),
        access_I = isl.BasicMap("{ [i,k] -> I[i + k] }"),
        access_O = isl.BasicMap("{ [i,k] -> O[i] }"),
        access_W = isl.BasicMap("{ [i,k] -> W[k] }"),
    )
    schedule = isl.BasicMap("{ [i,k] -> [floor(i/2), (i%2), k] }")
    operator = operator.apply_schedule(schedule)
    
    domain_input_points = operator_to_domain_input_points(operator)
    domain_point_colors = []
    for domain_point, input_point in domain_input_points:
        domain_point_color = (*domain_point, num_to_color(input_point[0]))
        domain_point_colors.append(domain_point_color)
    # print(domain_point_colors)
    plot_3d_scene(domain_point_colors, [])

if __name__ == "__main__":
    main()