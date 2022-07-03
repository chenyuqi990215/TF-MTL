import sys
sys.path.append('../../')

from models.HetETA_utils.map import RoadNetworkMap

import re
import dgl

def getTrajs(file_path):
    ret_records = []
    buffer_records = []
    data = open(file_path, 'r')
    for item in data:
        line = item.strip()
        line = re.split(' |,', line)
        if line[0][0] == '-':
            ret_records.append(buffer_records)
            buffer_records = []
        else:
            buffer_records.append(line)
    return ret_records

from models.HetETA_utils.spatial_func import *
def cross(x, y):
    return x.lat * y.lng - x.lng * y.lat

def dot(x, y):
    return x.lat * y.lat + x.lng * y.lng

def minus(x, y):
    return SPoint(x.lat - y.lat, x.lng - y.lng)


def angle(rn, r1, r2):
    p1 = SPoint(rn.edgeCord[r1][-4], rn.edgeCord[r1][-3])
    p2 = SPoint(rn.edgeCord[r1][-2], rn.edgeCord[r1][-1])
    q1 = minus(p2, p1)

    p1 = SPoint(rn.edgeCord[r2][0], rn.edgeCord[r2][1])
    p2 = SPoint(rn.edgeCord[r2][2], rn.edgeCord[r2][3])
    q2 = minus(p2, p1)
    return cross(q1, q2)

def dott(rn, r1, r2):
    p1 = SPoint(rn.edgeCord[r1][-4], rn.edgeCord[r1][-3])
    p2 = SPoint(rn.edgeCord[r1][-2], rn.edgeCord[r1][-1])
    q1 = minus(p2, p1)

    p1 = SPoint(rn.edgeCord[r2][0], rn.edgeCord[r2][1])
    p2 = SPoint(rn.edgeCord[r2][2], rn.edgeCord[r2][3])
    q2 = minus(p2, p1)
    return dot(q1, q2)

def edge_type(rn, r1, r2):
    a = angle(rn, r1,r2)
    if abs(a) < 1e-10:
        d = dott(rn, r1, r2)
        if d < 0:
            return 'turn_around'
        else:
            return 'straight'
    elif a > 0:
        return 'turn_right'
    else:
        return 'turn_left'


class HetroRoadGraph:
    def __init__(self, map_root, zone_range, traj_path, unit_length=50):
        self.rn = RoadNetworkMap(map_root, zone_range=zone_range, unit_length=unit_length)
        self.trajs = getTrajs(traj_path)
        self.road_hg = self.road_hetro()
        self.traj_hg = self.traj_hetro()

    def traj_hetro(self):
        trans = {}
        for rid in self.rn.valid_edge:
            trans[rid] = {}

        for traj in self.trajs:
            flag = True
            for line in traj:
                if int(line[-1]) not in self.rn.valid_edge:
                    flag = False
            if not flag:
                continue
            pre = int(traj[0][-1])
            for line in traj[1:]:
                cur = int(line[-1])
                if cur not in trans[pre]:
                    trans[pre][cur] = 0
                trans[pre][cur] += 1

        src, dst = [], []
        for rid in self.rn.valid_edge:
            item = list(trans[rid].items())
            item = sorted(item, key=lambda x: x[1], reverse=True)
            src.append(self.rn.valid_edge[rid])
            dst.append(self.rn.valid_edge[rid])

            for (nrid, _) in item[:10]:
                src.append(self.rn.valid_edge[rid])
                dst.append(self.rn.valid_edge[nrid])

        hg = dgl.heterograph({
            ('segment', 'likely_to_go', 'segment'): (src, dst),
            ('segment', 'on_going_likely_to_go', 'segment'): (dst, src),
        })

        return hg

    def road_hetro(self):
        right_src, right_dst, left_src, left_dst, \
        straight_src, straight_dst, around_src, around_dst = [], [], [], [], [], [], [], []

        for rid in self.rn.valid_edge:
            for nrid in self.rn.edgeDict[rid]:
                if nrid not in self.rn.valid_edge:
                    continue
                vrid = self.rn.valid_edge[rid]
                vnrid = self.rn.valid_edge[nrid]
                type = edge_type(self.rn, rid, nrid)
                if 'left' in type:
                    left_src.append(vrid)
                    left_dst.append(vnrid)
                elif 'right' in type:
                    right_src.append(vrid)
                    right_dst.append(vnrid)
                elif 'around' in type:
                    around_src.append(vrid)
                    around_dst.append(vnrid)
                else:
                    straight_src.append(vrid)
                    straight_dst.append(vnrid)

        src, dst = [], []
        for rid in self.rn.valid_edge:
            src.append(self.rn.valid_edge[rid])
            dst.append(self.rn.valid_edge[rid])
            for nrid in self.rn.edgeDict[rid]:
                if nrid not in self.rn.valid_edge:
                    continue
                src.append(self.rn.valid_edge[rid])
                dst.append(self.rn.valid_edge[nrid])
        g = dgl.graph((src, dst), num_nodes=self.rn.valid_edge_cnt)
        hop1_src, hop1_dst = src, dst
        hop2_src, hop2_dst = dgl.transform.khop_graph(g, 2).edges()

        hg = dgl.heterograph({
            ('segment', 'turn_left', 'segment'): (left_src, left_dst),
            ('segment', 'on_going_turn_left', 'segment'): (left_dst, left_src),
            ('segment', 'turn_right', 'segment'): (right_src, right_dst),
            ('segment', 'on_going_turn_right', 'segment'): (right_dst, right_src),
            ('segment', 'turn_around', 'segment'): (around_src, around_dst),
            ('segment', 'on_going_turn_right', 'segment'): (around_dst, around_src),
            ('segment', 'straight', 'segment'): (straight_src, straight_dst),
            ('segment', 'on_going_straight', 'segment'): (straight_dst, straight_src),
            ('segment', 'hop_1', 'segment'): (hop1_src, hop1_dst),
            ('segment', 'hop_2', 'segment'): (hop2_src, hop2_dst),
        })

        return hg

if __name__ == "__main__":
    map_root = '/nas/user/cyq/TrajectoryRecovery/roadnet/Chengdu/'
    zone_range = [30.655347, 104.039711, 30.730157, 104.127151]
    traj_path = '/nas/user/cyq/TrajectoryRecovery/train_data_final/Chengdu/valid/valid_output.txt'
    HetroRoadGraph(map_root, zone_range, traj_path)

