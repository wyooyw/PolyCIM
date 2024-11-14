import itertools
from sympy import symbols, simplify

# 定义符号变量
s1, s2, s3, s4, s5 = symbols('s1 s2 s3 s4 s5')

# 定义一个表达式
f1 = (s1*s2+s2*s3+s1*s3-s1-s2-s3+1)*s4*s5
f2 = (s4+s5-1)*s1*s2*s3



# 使用 simplify 函数化简表达式
simplified_f1 = simplify(f1)
print(f"{simplify(f1)=}")
print(f"{simplify(f2)=}")
print(f"{simplify(f1-f2)=}")

exit()
r = 2
domain = list(range(r, 10))
lt_cnt = 0
gt_cnt = 0
eq_cnt = 0
for s1,s2,s3,s4,s5 in itertools.product(domain, domain, domain, domain, domain):
    if min(s1,s2,s3)>r or min(s4,s5) > r:
        continue

    f1 = (s1*s2+s2*s3+s1*s3-s1-s2-s3+1)*s4*s5
    f2 = (s4+s5-1)*s1*s2*s3

    if f1 < f2:
        print(f"{lt_cnt}")
        print(f"{f1=}, {f2=}")
        print(f"{s1=}, {s2=}, {s3=}, {s4=}, {s5=}, {f1=}, {f2=}")
        print("")
        lt_cnt += 1
    elif f1 > f2:
        gt_cnt += 1
    else:
        eq_cnt += 1
        # if cnt > 10:
        #     break
    # if f1 > f2:
    #     print(f"{s1=}, {s2=}, {s3=}, {s4=}, {s5=}, {f1=}, {f2=}")

print(f"{lt_cnt=}, {gt_cnt=}, {eq_cnt=}")