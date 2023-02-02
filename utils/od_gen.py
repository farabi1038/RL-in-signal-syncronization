import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
from lxml import etree
import argparse
import os
import time
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int)
args = parser.parse_args()
if args.run == None:
    print("enter run flag")
    exit()
np.random.seed(args.run)
run = str(args.run)


class headnode:
    def __init__(self, headID):
        self.headID = headID
        self.head = None

    def return_turn_set(self):
        temp_list = []
        temp = self.head
        while temp is not None:
            temp_list.append(temp.ID)
            if temp.left is not None:
                temp_list.append(temp.left.ID)
            if temp.right is not None:
                temp_list.append(temp.right.ID)
            temp = temp.through
        return temp_list

    def return_through_set(self):
        temp_list = []
        temp = self.head
        while temp is not None:
            temp_list.append(temp.ID)
            temp = temp.through
        return temp_list

    def return_part1(self, commonID):
        temp_list = []
        temp = self.head
        while temp is not None:
            temp_list.append(temp.ID)
            if temp.left is not None:
                if temp.left.ID == commonID:
                    if temp.ramp is not None:
                        temp_list.append(temp.ramp.ID)
                    return temp_list
            if temp.right is not None:
                if temp.right.ID == commonID:
                    if temp.ramp is not None:
                        temp_list.append(temp.ramp.ID)
                    return temp_list
            temp = temp.through

    def return_part2(self, commonID):
        temp_list = []
        temp = self.head
        while temp is not None:
            temp_list.append(temp.ID)
            temp = temp.through
        temp_list = np.array(temp_list)
        idx = np.where(temp_list == commonID)[0][0]
        return list(temp_list[idx:])

class segment:
    def __init__(self, laneID=None):
        self.ID = laneID
        self.through = None
        self.left = None
        self.right = None
        self.ramp = None

def populate_segment(name, lists):
    A = headnode(name)
    first = lists[0]
    if isinstance(first, dict):
        value = list(first.keys())[0]
        A.head = segment(value)
        trav = A.head
        dic = list(first.values())[0].keys()
        for k in dic:
            if k == "L":
                trav.left = segment(first[value][k])
            elif k == "R":
                trav.right = segment(first[value][k])
            else:
                trav.ramp = segment(first[value][k])
    else:
        A.head = segment(first)
        trav = A.head
    for i in lists:
        if i == lists[0]:
            continue
        if isinstance(i, dict):
            value = list(i.keys())[0]
            trav.through = segment(value)
            trav = trav.through
            dic = list(i.values())[0].keys()
            for k in dic:
                if k == "L":
                    trav.left = segment(i[value][k])
                elif k == "R":
                    trav.right = segment(i[value][k])
                else:
                    trav.ramp = segment(i[value][k])
        else:
            trav.through = segment(i)
            trav = trav.through
    return A

def poisson(mu):
    x = np.arange(0, 10000)
    cumsum = (mu**x * np.exp(-mu)) / factorial(x)
    condition = np.where(np.cumsum(cumsum) >= 0.99999)
    if len(condition[0]) != 0:
        return cumsum[:condition[0][0]+1]
    else:
        return cumsum

def routes(root, ids, edges):
    route = etree.SubElement(root, "route", id=ids, edges=edges)

def vehicles(root, ids, routeIDs, t):
    vehicle = etree.SubElement(root, "vehicle", id=ids, type="car", route=routeIDs, depart=t)


class master:
    def __init__(self):

        r1t2 = ["363648686#0", "363648686#2", "363648686#3", {"363648686#5": {"L": "-598977856#0", "R": "10137507#1"}}, "591388269#1", "591388269#1.160", "591388269#1.258", {"692123188": {"L": "692513881#1", "R": "-591388261#0"}}, "692513882", "692513882.119", {"692513882.255": {"L": "692513874#0", "R": "-10124969#6"}}, "692513873", {"692513873.472": {"L": "-591388258#1", "R": "-139051731#19"}}, "146626844#2", "146626844#2.308", {"146626844#5.190": {"R": "145216629#1"}}, "146626841#1", "146626839#0", "146626839#0.73", {"207412432#0.28": {"L": "172099960#1", "R": "-145218802#37"}}, "207412432#3", "207412432#4", "207412432#4.140", {"697666080": {"L": "-10131659#5", "R": "69221466#0"}}, "207412426#0", "697666084", "697666083#0", "697666083#0.332", {"697666081#0": {"L": "-145216628#2", "R": "-508466788#1"}}, "697666082#1", "697666085#0", "697666085#0.242", {"697666085#9": {"L": "10150792#1", "R": "-10133290#0"}}, "697666085#12", "697666085#13", "697666085#13.295", {"697666085#15.103": {"L": "10137014#1", "R": "-10147126#9"}}, "696962877#0", "632830122"]
        r2t1 = ["-632830121#1", "-696962877#4", {"-696962877#1.83": {"L": "-10147126#9", "R": "10137014#1"}}, "-697666085#15", {"-697666085#12": {"L": "-10133290#0", "R": "10150792#1"}}, "-697666085#9", "-697666085#8", "-697666085#8.761", {"-697666082#1": {"L": "-508466788#1", "R": "-145216628#2"}}, "-697666081#1", "-697666083#6", "-697666083#6.199", "-697666084", {"-207412426#1": {"L": "69221466#0", "R": "-10131659#5"}}, "-697666080", "-697666080.66", "-207412432#8", "-207412432#8.103", {"-207412432#3": {"L": "-145218802#37", "R": "172099960#1"}}, "-207412432#0", "-207412432#0.230", {"600491439#0": {"L": "145216629#1"}}, "146626843#0", "146626843#0.250", "146626843#0.250.97", {"146626843#2": {"L": "-139051731#19", "R": "-591388258#1"}}, "146626842#2", {"207412421-AddedOnRampEdge": {"R": "692513874#1", "ramp": "439745512"}}, {"591388255#1": {"L": "-10124969#6"}}, "692513877#1", {"692513877#1.27": {"R": "621228441#0-AddedOnRampEdge", "ramp": "10128564#0"}}, {"692123191": {"L": "-591388261#0"}}, "692513879#1", "683763253#1", {"683763253#1.209": {"L": "10137507#1", "R": "-598977856#0"}}, "105826050#1"]
        r3t4 = ["10150738", {"598977856#0": {"L": "105826050#1", "R": "591388269#1"}}, "10137507#1"]
        r4t3 = ["749736413#1", {"749736413#1.127": {"L": "105826050#1", "R": "591388269#1"}}, "-598977856#0", "138668049"]
        r5t6 = ["621228442#0", {"621228442#1": {"L": "692513879#1", "R": "692513882"}}, "-591388261#0"]
        r6t5 = ["-32921656#0", {"683763238#1.51": {"L": "692513879#1", "R": "692513882"}}, "692513881#1", "621228441#0-AddedOnRampEdge"]
        r7t8 = [{"684464938#1": {"L": "692513873", "R": "692513877#1"}}, "-10124969#6", "-10124969#5"]
        r8t7 = ["10124969#4", {"10124969#6": {"L": "692513877#1", "R": "692513873"}}, "692513874#0", "692513874#1"]
        r9t10 = [{"683763247#0": {"R": "207412421-AddedOnRampEdge", "ramp": "683763248"}}, {"591388258#0": {"L": "146626844#2"}}, "-139051731#19", "-139051731#18"]
        r10t9 = ["139051731#16", {"139051731#19": {"L": "146626842#2", "R": "146626844#2"}}, "-591388258#1", "-683763247#1"]
        r0t11 = ["145216629#1", "146626846#1"]
        r11t0 = ["-146626846#5", {"-146626846#0": {"L": "146626843#0", "R": "146626841#1"}}]
        r12t13 = ["-172099960#8", {"-172099960#1": {"L": "-207412432#0", "R": "207412432#3"}}, "-145218802#37", "-145218802#36"]
        r13t12 = ["145218802#36", {"145218802#37": {"L": "-207412432#0", "R": "207412432#3"}}, "172099960#1", "172099960#2"]
        r14t15 = ["10131659#1", {"10131659#5": {"R": "-697666080.66", "ramp": "gneE41"}}, {"10131659#5.37": {"L": "207412426#0"}}, "69221466#0", "69221466#1"]
        r15t14 = ["-69221466#3", "-69221466#0", {"-69221466#0.18": {"L": "-697666080", "R": "207412426#0"}}, "-10131659#5", "-10131659#4"]
        r16t17 = ["145216628#1", {"145216628#2": {"L": "697666082#1", "R": "-697666081#1"}}, "-508466788#1", "-69232295#3"]
        r17t16 = ["69232295#0", {"508466788#0": {"L": "-697666081#1", "R": "697666082#1"}}, "-145216628#2", "-145216628#1"]
        r18t19 = ["-10150792#4", {"-10150792#1": {"L": "697666085#12", "R": "-697666085#9"}}, "-10133290#0", "-10150790#3"]
        r19t18 = ["10133292", {"10133290#0": {"L": "-697666085#9", "R": "697666085#12"}}, "10150792#1", "10150792#2"]
        r20t21 = [{"-10137014#3": {"L": "696962877#0", "R": "-697666085#15"}}, "-10147126#9"]
        r21t20 = [{"10147126#4": {"L": "-697666085#15", "R": "696962877#0"}}, "10137014#1"]

        o1d2 = populate_segment("1-2", r1t2)
        o2d1 = populate_segment("2-1", r2t1)

        o3d4 = populate_segment("3-4", r3t4)
        o4d3 = populate_segment("4-3", r4t3)

        o5d6 = populate_segment("5-6", r5t6)
        o6d5 = populate_segment("6-5", r6t5)

        o7d8 = populate_segment("7-8", r7t8)
        o8d7 = populate_segment("8-7", r8t7)

        o9d10 = populate_segment("9-10", r9t10)
        o10d9 = populate_segment("9-10", r10t9)

        o0d11 = populate_segment("0-11", r0t11)
        o11d0 = populate_segment("11-0", r11t0)

        o12d13 = populate_segment("12-13", r12t13)
        o13d12 = populate_segment("13-12", r13t12)

        o14d15 = populate_segment("14-15", r14t15)
        o15d14 = populate_segment("15-14", r15t14)

        o16d17 = populate_segment("16-17", r16t17)
        o17d16 = populate_segment("17-16", r17t16)

        o18d19 = populate_segment("18-19", r18t19)
        o19d18 = populate_segment("19-18", r19t18)

        o20d21 = populate_segment("20-21", r20t21)
        o21d20 = populate_segment("21-20", r21t20)

        self.O_list = {1: o1d2, 2: o2d1, 3: o3d4, 4: o4d3, 5: o5d6, 6: o6d5, 7: o7d8, 8: o8d7, 9: o9d10, 10: o10d9, 11: o11d0, 12: o12d13, 13: o13d12, 14: o14d15, 15: o15d14, 16: o16d17, 17: o17d16, 18: o18d19, 19: o19d18, 20: o20d21, 21: o21d20}
        self.D_list = {1: o2d1, 2: o1d2, 3: o4d3, 4: o3d4, 5: o6d5, 6: o5d6, 7: o8d7, 8: o7d8, 9: o10d9, 10: o9d10, 11: o0d11, 12: o13d12, 13: o12d13, 14: o15d14, 15: o14d15, 16: o17d16, 17: o16d17, 18: o19d18, 19: o18d19, 20: o21d20, 21: o20d21}

    def return_route(self, origin, destination):
        # print(f"Origin: {origin}, Destination: {destination}")
        o = self.O_list[origin]
        d = self.D_list[destination]

        if o == d:
            return o.return_through_set()

        o_set = o.return_turn_set()
        d_set = d.return_through_set()

        try:
            intersect = list(set(o_set) & set(d_set))[0]
        except IndexError:
            temp1 = self.D_list[1]
            temp2 = self.D_list[2]

            intersect1 = list(set(o_set) & set(temp1))[0]
            intersect2 = list(set(o_set) & set(temp2))[0]

        p1 = o.return_part1(intersect)
        p2 = d.return_part2(intersect)
        return p1 + p2

if not os.path.isdir(f"route_files"):
    os.makedirs(f"route_files")

if not os.path.isdir(f"npy_files"):
    os.makedirs(f"npy_files")

for samples in range(1):
    com = master()

    x = np.full(21*21,-1).reshape(21,-1)

    for idx, i in enumerate(x):
        for jdx, j in enumerate(i):
            if idx == jdx:
                x[idx][jdx] = 0

    for idx, i in enumerate(x):
        for jdx, j in enumerate(i):
            if idx > 1 and jdx > 1:
                x[idx][jdx] = 0

    x[2][3] = -1
    x[3][2] = -1
    x[1][10] = 0
    x[8][9] = -1
    x[9][8] = -1
    x[11][12] = -1
    x[12][11] = -1
    x[13][14] = -1
    x[14][13] = -1
    x[15][16] = -1
    x[16][15] = -1
    x[17][18] = -1
    x[18][17] = -1
    x[19][20] = -1
    x[20][19] = -1
    x = x.astype(np.int32)


    # fig, ax = plt.subplots(figsize=(16, 14))
    # sns.set()
    # sns.heatmap(x, annot=True, fmt="d")
    # plt.savefig("test.png")
    # plt.cla()
    # plt.clf()
    # plt.close()

    # main_high = 180
    # main_low = 0

    # choice = np.random.choice([0,1])
    # if choice == 1:
    #     x[0][1] = int(np.random.uniform(main_high, main_low))
    #     x[1][0] = int(x[0][1] * .4)
    # else:
    #     x[1][0] = int(np.random.uniform(main_high, main_low))
    #     x[0][1] = int(x[1][0] * .4)
    # main_min = min(x[0][1], x[1][0])
    # main_max = max(x[0][1], x[1][0])

    # side_high = (main_max * .4) / 19
    # side_low = (main_min * .4) / 19

    N = 3600

    soft_cap_1 = 180
    soft_cap_2 = 180
    soft_cap_3 = 120
    soft_cap_4 = 90
    soft_cap_5 = 90
    soft_cap_6 = 60
    soft_cap_7 = 90
    soft_cap_8 = 60
    soft_cap_9 = 120
    soft_cap_10 = 120
    soft_cap_11 = 90
    soft_cap_12 = 90
    soft_cap_13 = 90
    soft_cap_14  = 60
    soft_cap_15 = 120
    soft_cap_16 = 60
    soft_cap_17 = 60
    soft_cap_18 = 60
    soft_cap_19 = 60
    soft_cap_20 = 30
    soft_cap_21 = 30
    soft_cap = [180, 180, 120, 90, 90, 60, 90, 60, 120, 120, 90, 90, 90, 60, 120, 60, 60, 60, 60, 30, 30]
    x[0][1] = int(np.random.normal(((soft_cap_1/2) * int(N/300)),((soft_cap_1/6) * int(N/300)),1).clip(0, (soft_cap_1 * int(N/300))))
    x[1][0] = int(np.random.normal(((soft_cap_2/2) * int(N/300)),((soft_cap_2/6) * int(N/300)),1).clip(0, (soft_cap_2 * int(N/300))))

    res1 = (soft_cap_1 * int(N/300)) - x[0][1]
    res2 = (soft_cap_2 * int(N/300)) - x[1][0]

    # print(f"x[0][1]: {x[0][1]}, res1: {res1}, x[1][0]: {x[1][0]}, res2: {res2}")
    idx = 0
    for i, soft in zip(x,soft_cap):
        # breakpoint()
        jdx = 0
        neg = np.where(i==-1)[0]
        soft_s = (soft * int(N/300))//len(neg)
        for j in i:
            if x[idx][jdx] == -1:
                if idx == 0:
                    soft_s = res1//19
                    mean = soft_s//2
                    if res1 == 0:
                        std = 0
                    else:
                        std = max((soft_s//2)//3, 1)
                    # print(f"{idx} {jdx} {soft_s}, mean {mean}, std {std}")
                    x[idx][jdx] = int(np.random.normal(mean, std,1).clip(0))
                    jdx += 1
                    continue
                elif idx == 1:
                    soft_s = res2//19
                    mean = soft_s//2
                    if res2 == 0:
                        std = 0
                    else:
                        std = max((soft_s//2)//3, 1)
                    # print(f"{idx} {jdx} {soft_s}, mean {mean}, std {std}")
                    x[idx][jdx] = int(np.random.normal(mean, std,1).clip(0))
                    jdx += 1
                    continue
                mean = soft_s//2
                std = max((soft_s//2)//3, 1)
                # print(f"{idx} {jdx} {soft_s}, mean {mean}, std {std}")
                # x[idx][jdx] = int(np.random.uniform(soft, 0))
                x[idx][jdx] = int(np.random.normal(mean, std,1).clip(0))
            jdx += 1
        if sum(x[idx]) > (soft * int(N/300)):
            extra = sum(x[idx]) - (soft * int(N/300))
            nonzero = np.where(x[idx][neg] != 0)[0]
            random = np.random.choice(nonzero)
            x[idx][random] = x[idx][random] - extra
        idx += 1
    x = x.astype(np.int64)
    # print(f"Vehicles allocated, total: {sum(sum(x))}")
    # fig, ax = plt.subplots(figsize=(16, 14))
    # sns.heatmap(x, annot=True, fmt="d")
    # plt.savefig(f"testing_{samples}.png")
    # plt.cla()
    # plt.clf()
    # plt.close()

    root = etree.Element("routes")
    vType = etree.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", length="5", minGap="2.5", maxSpeed="55.55", guiShape="passenger")

    for idx, i in enumerate(x):
        for jdx, j in enumerate(i):
            if x[idx][jdx] == 0:
                continue
            route = com.return_route(idx+1, jdx+1)
            routes(root, str(f"{idx+1}_{jdx+1}"), " ".join(route))

    vehNr = 0
    # t0 = time.time()
    cumsum_dic = {}
    # breakpoint()
    # for t in range(N):
    for idx, i in enumerate(x):
        for jdx, j in enumerate(i):
            if x[idx][jdx] == 0:
                continue
            mu = x[idx][jdx] / N
            probabilities = poisson(mu)
            cumsum = np.cumsum(probabilities)
            if idx not in cumsum_dic:
                cumsum_dic[idx] = {}
            if jdx not in cumsum_dic[idx]:
                cumsum_dic[idx][jdx] = cumsum
    # breakpoint()
    # bplot = []
    for t in range(N):
        t_cum = 0
        rng = np.random.uniform(0, 1)
        for i in cumsum_dic.keys():
            for j in cumsum_dic[i].keys():
                try:
                    veh_num = np.where(cumsum_dic[i][j] <= rng)[0][-1] + 1
                except IndexError:
                    veh_num = 0
                for _ in range(veh_num):
                    vehicles(root, "veh_" + str(vehNr), str(f"{i+1}_{j+1}"), str(t))
                    vehNr += 1
                    # t_cum += 1
        # bplot.append(t_cum)
    # plt.plot(bplot)
    # plt.savefig("lineplot.png")
                
        # for idx, i in enumerate(x):
        #     for jdx, j in enumerate(i):
        #         if x[idx][jdx] == 0:
        #             continue
        #         mu = x[idx][jdx] / N
        #         probabilities = poisson(mu)
        #         cumsum = np.cumsum(probabilities)
        #         try:
        #             veh_num = np.where(cumsum <= rng)[0][-1] + 1
        #         except IndexError:
        #             veh_num = 0

        #         for _ in range(veh_num):
        #             vehicles(root, "veh_" + str(vehNr), str(f"{idx+1}_{jdx+1}"), str(t))
        #             vehNr += 1
    # t1 = time.time()
    # print(f"time taken : {t1-t0}")
    np.save(f"npy_files/numpy.sample.{run}.{samples}", x)
    s = etree.tostring(root, pretty_print=True, encoding="unicode")
    print(f"created route_files/routes.sample.{run}.{samples}.xml")
    with open(f'route_files/routes.sample.{run}.{samples}.xml', "w") as f:
        f.write(s)
    