from easyvec import Vec3


v1 = Vec3(1, 2, 3)
v2 = Vec3(4, 5, 6)

# wrong_v = Vec3(1, 'a', [1, 2, 3])
# wrong_v.x = "1"
# print(wrong_v.x, wrong_v._x, wrong_v.z)

print(f"{v1+v2=}")
print(f"{v1-v2=}")
print(f"{v1*2=}")
print(f"{v1/2=}")
print(f"{v1==v2=}")
print(f"{v1!=v2=}")
print(f"{v1.dot(v2)=}")
print(f"{v1.cross(v2)=}")
print(f"{v1.distance_to(v2)=}")
print(f"{v1.norm()=}")
print(f"{v1.normsq()=}")
