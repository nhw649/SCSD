"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
import os
import copy

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {
        "color": (250, 170, 30),
        "isthing": 0,
        "id": 19,
        "trainId": 6,
        "name": "traffic light",
    },
    {
        "color": (220, 220, 0),
        "isthing": 0,
        "id": 20,
        "trainId": 7,
        "name": "traffic sign",
    },
    {
        "color": (107, 142, 35),
        "isthing": 0,
        "id": 21,
        "trainId": 8,
        "name": "vegetation",
    },
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]


def get_categories_with_prompt_eng():
    CITYSCAPES_CATEGORIES_ = copy.deepcopy(CITYSCAPES_CATEGORIES)
    cityscapes_id_names = (
        open("./scsd/data/datasets/classes_with_prompt_eng.txt")
        .read()
        .splitlines()
    )
    cityscapes_idx = 0
    for line in cityscapes_id_names:
        idx, name = line.split(":")
        idx = int(idx)
        if name == "invalid_class_id":
            continue

        assert CITYSCAPES_CATEGORIES_[cityscapes_idx]["trainId"] == idx
        CITYSCAPES_CATEGORIES_[cityscapes_idx]["name"] = name
        cityscapes_idx += 1
    return CITYSCAPES_CATEGORIES_

if __name__ == "__main__":
    get_categories_with_prompt_eng()
