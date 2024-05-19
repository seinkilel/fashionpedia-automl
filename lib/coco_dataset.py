import json
import time
from typing import Optional
from urllib.request import urlopen


class AutoMLObjectDetectionConverter:
    def __init__(self, coco_data, normalize=True):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.normalize = normalize
        self.image_id_to_data_index = {}
        for i in range(0, len(coco_data["images"])):
            self.json_lines_data.append({})
            self.json_lines_data[i]["image_url"] = ""
            self.json_lines_data[i]["image_details"] = {}
            self.json_lines_data[i]["label"] = []
        for i in range(0, len(coco_data["categories"])):
            self.categories[coco_data["categories"][i]["id"]] = coco_data["categories"][i]["name"]

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]["image_url"] = coco_image["file_name"]
        self.image_id_to_data_index[coco_image["id"]] = index

    def _populate_image_details(self, index, coco_image):
        file_name = coco_image["file_name"]
        self.json_lines_data[index]["image_details"]["format"] = file_name[
                                                                 file_name.rfind(".") + 1:
                                                                 ]
        self.json_lines_data[index]["image_details"]["width"] = coco_image["width"]
        self.json_lines_data[index]["image_details"]["height"] = coco_image["height"]

    def _populate_bbox_in_label(self, label, annotation, image_details):
        if max(annotation["bbox"]) < 1.5:
            width = 1
            height = 1
        else:
            width = image_details["width"]
            height = image_details["height"]
        label["topX"] = annotation["bbox"][0] / width
        label["topY"] = annotation["bbox"][1] / height
        label["bottomX"] = (annotation["bbox"][0] + annotation["bbox"][2]) / width
        label["bottomY"] = (annotation["bbox"][1] + annotation["bbox"][3]) / height

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation["image_id"]]
        image_details = self.json_lines_data[index]["image_details"]
        label = self.categories[annotation["category_id"]]
        spl = label.split(", ")
        label = {"label": spl[0]}
        self._populate_bbox_in_label(label, annotation, image_details)
        self._populate_is_crowd(label, annotation)
        self.json_lines_data[index]["label"].append(label)

    def _populate_is_crowd(self, label, annotation):
        if "iscrowd" in annotation.keys():
            label["isCrowd"] = annotation["iscrowd"]

    def convert(self):
        for i in range(0, len(self.coco_data["images"])):
            self._populate_image_url(i, self.coco_data["images"][i])
            self._populate_image_details(i, self.coco_data["images"][i])
        for i in range(0, len(self.coco_data["annotations"])):
            self._populate_label(self.coco_data["annotations"][i])
        return self.json_lines_data


def write_json_lines(converter: AutoMLObjectDetectionConverter, filename, base_url=None):
    json_lines_data = converter.convert()
    with open(filename, "w") as outfile:
        for json_line in json_lines_data:
            if base_url is not None:
                image_url = json_line["image_url"]
                json_line["image_url"] = (
                        base_url + image_url[image_url.rfind("/") + 1:]
                )
            json.dump(json_line, outfile, separators=(",", ":"))
            outfile.write("\n")
        print(f"Conversion completed. Converted {len(json_lines_data)} lines.")


def create_json_object_detection_dataset(url: str, filename: str, img_base_url: Optional[str] = None):
    print('loading annotations into memory...')
    tic = time.time()
    response = urlopen(url)
    dataset = json.loads(response.read())
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))
    tic = time.time()
    print('write detection dataset...')
    converter = AutoMLObjectDetectionConverter(dataset)
    write_json_lines(converter, filename, base_url=img_base_url)
    print('Done (t={:0.2f}s)'.format(time.time() - tic))
