#!/usr/bin/env python

#               ,,))))))));,
#            __)))))))))))))),
# \|/       -\(((((''''((((((((.
# -*-==//////((''  .     `)))))),
# /|\      ))| o    ;-.    '(((((                                  ,(,
#          ( `|    /  )    ;))))'                               ,_))^;(~
#             |   |   |   ,))((((_     _____------~~~-.        %,;(;(>';'~
#             o_);   ;    )))(((` ~---~  `::           \      %%~~)(v;(`('~
#                   ;    ''''````         `:       `:::|\,__,%%    );`'; ~
#                  |   _                )     /      `:|`----'     `-'
#            ______/\/~    |                 /        /
#          /~;;.____/;;'  /          ___--,-(   `;;;/
#         / //  _;______;'------~~~~~    /;;/\    /
#        //  | |                        / ;   \;;,\
#       (<_  | ;                      /',/-----'  _>
#        \_| ||_                     //~;~~~~~~~~~
#            `\_|                   (,~~
#                                    \~\
#                                     ~~

# Standard library imports
from os import path
from math import cos, sin, atan2
# Installed library imports
import yaml


# Package specific file imports

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class Point2D:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()


class Map2D:
    def __init__(self, yellowCones=None, blueCones=None, unknownCones=None, smallOrangeCones=None, bigOrangeCones=None):
        if bigOrangeCones is None:
            bigOrangeCones = []
        if smallOrangeCones is None:
            smallOrangeCones = []
        if unknownCones is None:
            unknownCones = []
        if blueCones is None:
            blueCones = []
        if yellowCones is None:
            yellowCones = []
        self.yellowCones = yellowCones
        self.blueCones = blueCones
        self.unknownCones = unknownCones
        self.smallOrangeCones = smallOrangeCones
        self.bigOrangeCones = bigOrangeCones

    def __str__(self):
        return "{yellowCones: " + str(self.yellowCones) + ",\nblueCones: " + str(self.blueCones) + "}"


class ConeFilterNode:
    def __init__(self):
        yaml_data = yaml.load(open('cone_filter_sorter.yaml', 'r'), Loader=yaml.FullLoader)

        self.yaw = None
        self.position = None
        self.map = None

        self.crop_box_y_max = yaml_data["crop_box_y_max"]
        self.crop_box_x_min = yaml_data["crop_box_x_min"]
        self.crop_box_x_max = yaml_data["crop_box_x_max"]

        self.min_heading_difference = yaml_data["min_heading_difference"]
        self.max_heading_difference = yaml_data["max_heading_difference"]

        self.used_locations = set()
        self.current_depth = 0
        self.MAX_DEPTH = 222

    def map_update(self, data):
        cones_yellow = []
        for cone in data[0]:
            cones_yellow.append(Point2D(x=cone[0], y=cone[1], id=0))
        cones_blue = []
        for cone in data[1]:
            cones_blue.append(Point2D(x=cone[0], y=cone[1], id=0))
        self.map = Map2D(yellowCones=cones_yellow, blueCones=cones_blue)
        return self.filter_data()

    def pose_update(self, data):
        self.yaw = data.phi
        loc = data.location
        self.position = Point(x=loc[0], y=loc[1])

    @staticmethod
    def transform_cones_to_local(pose, yaw, cones):
        output = []
        for cone in cones:
            new_x = cone.x - pose.x
            new_y = cone.y - pose.y
            output.append(Point2D(x=cos(yaw) * new_x + sin(yaw) * new_y,
                                  y=cos(yaw) * new_y - sin(yaw) * new_x, id=cone.id))
        return output

    def transform_map_to_local(self, pose, yaw, map):
        return Map2D(yellowCones=self.transform_cones_to_local(pose, yaw, map.yellowCones),
                     blueCones=self.transform_cones_to_local(
                         pose, yaw, map.blueCones),
                     unknownCones=self.transform_cones_to_local(
                         pose, yaw, map.unknownCones),
                     smallOrangeCones=self.transform_cones_to_local(
                         pose, yaw, map.smallOrangeCones),
                     bigOrangeCones=self.transform_cones_to_local(pose, yaw, map.bigOrangeCones))

    def apply_crop_box_to_cones(self, cones):
        return list(filter(lambda
                               cone: self.crop_box_x_max >= cone.x >= self.crop_box_x_min and self.crop_box_y_max >= cone.y >= -self.crop_box_y_max,
                           cones))

    def apply_crop_box_to_map(self, map):
        return Map2D(yellowCones=self.apply_crop_box_to_cones(map.yellowCones),
                     blueCones=self.apply_crop_box_to_cones(map.blueCones),
                     unknownCones=self.apply_crop_box_to_cones(
                         map.unknownCones),
                     smallOrangeCones=self.apply_crop_box_to_cones(
                         map.smallOrangeCones),
                     bigOrangeCones=self.apply_crop_box_to_cones(map.bigOrangeCones))

    @staticmethod
    def sort_cones_by_distance(cones):
        return [a[1] for a in sorted([((cone.x ** 2 + cone.y ** 2), cone) for cone in cones], key=lambda x: x[0])]

    def find_best_cones(self, source_cones, target_cones, blue_to_yellow=False):
        if len(source_cones) == 0 or len(target_cones) == 0:
            return

        source_best = source_cones[0]
        target_best = target_cones[0]
        if target_best.x ** 2 + target_best.y ** 2 < source_best.x ** 2 + source_best.y ** 2:
            source_cones, target_cones = target_cones, source_cones
            blue_to_yellow ^= 1

        source_heading = atan2(source_cones[0].y, source_cones[0].x)
        for target_cone in target_cones:
            target_heading = atan2(target_cone.y, target_cone.x)
            heading_difference = (
                                         target_heading - source_heading) * (-1) ** blue_to_yellow
            if self.max_heading_difference >= heading_difference >= self.min_heading_difference:
                return source_cones[0], target_cone

        return self.find_best_cones(source_cones[1:], target_cones, blue_to_yellow)

    @staticmethod
    def remap_cones(cones, relations):
        return [relations[cone] for cone in cones]

    def remap_map(self, map, relations):
        return Map2D(yellowCones=self.remap_cones(map.yellowCones, relations),
                     blueCones=self.remap_cones(map.blueCones, relations),
                     unknownCones=self.remap_cones(
                         map.unknownCones, relations),
                     smallOrangeCones=self.remap_cones(
                         map.smallOrangeCones, relations),
                     bigOrangeCones=self.remap_cones(map.bigOrangeCones, relations))

    def get_map(self, pose, yaw, input_map):
        local_map = self.transform_map_to_local(pose, yaw, input_map)
        relations = dict(zip(
            local_map.yellowCones + local_map.blueCones + local_map.unknownCones + local_map.smallOrangeCones + local_map.bigOrangeCones,
            input_map.yellowCones + input_map.blueCones + input_map.unknownCones + input_map.smallOrangeCones + input_map.bigOrangeCones))

        input_map = self.apply_crop_box_to_map(local_map)

        yellow_cones = self.sort_cones_by_distance(input_map.yellowCones)
        blue_cones = self.sort_cones_by_distance(input_map.blueCones)
        unknown_cones = self.sort_cones_by_distance(input_map.unknownCones)

        cone_pair = self.find_best_cones(yellow_cones, blue_cones)
        if cone_pair is None:
            cone_pair = self.find_best_cones(yellow_cones, unknown_cones)
        if cone_pair is None:
            cone_pair = self.find_best_cones(blue_cones, unknown_cones, True)
        if cone_pair is None:
            return Map2D()

        index_switch = 0
        if cone_pair[0] in blue_cones or cone_pair[1] in yellow_cones:
            index_switch = 1

        new_pose = Point(x=(cone_pair[0].x + cone_pair[1].x) / 2,
                         y=(cone_pair[0].y + cone_pair[1].y) / 2)

        new_yaw = atan2(new_pose.y, new_pose.x)

        output_map = Map2D()
        self.current_depth += 1
        if new_pose not in self.used_locations and self.current_depth < self.MAX_DEPTH:
            self.used_locations.add(new_pose)
            output_map = self.get_map(new_pose, new_yaw, input_map)
        output_map.yellowCones = [cone_pair[index_switch]] + output_map.yellowCones
        output_map.blueCones = [cone_pair[1 ^ index_switch]] + output_map.blueCones

        return self.remap_map(output_map, relations)

    def filter_data(self):
        if self.map is None or self.yaw is None or self.position is None:
            return
        # print(self.position, self.yaw, self.map)
        self.current_depth = 0
        return self.get_map(self.position, self.yaw, self.map)


if __name__ == "__main__":
    ConeFilterNode()
