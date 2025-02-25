import os
import xml.etree.ElementTree as ET

# Ændr klassenavnet til nøjagtig det, der står i dine XML-filer.
classes = ["licence"]

def convert_annotation(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    print(f"Processing {xml_file} with size: {width}x{height}")

    yolo_lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.strip().lower()  # Trim og lav til små bogstaver
        print("Found object with class:", cls)
        # Sammenlign med din liste (sørg for, at både XML og liste er i lowercase)
        if cls not in [c.lower() for c in classes]:
            print("Skipping objekt, fordi class ikke matcher:", cls)
            continue
        cls_id = [c.lower() for c in classes].index(cls)
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        print("Converted YOLO line:", yolo_line)
        yolo_lines.append(yolo_line)

    return yolo_lines

def main():
    # Sørg for at stien her er korrekt – din mappe indeholder XML-filerne.
    annotations_dir = "dataset/annotations"
    labels_output_dir = "dataset/labels"  # Eller den ønskede sti

    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    for xml_filename in os.listdir(annotations_dir):
        if not xml_filename.endswith('.xml'):
            continue
        xml_path = os.path.join(annotations_dir, xml_filename)
        yolo_data = convert_annotation(xml_path, classes)
        txt_filename = os.path.splitext(xml_filename)[0] + '.txt'
        txt_path = os.path.join(labels_output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for line in yolo_data:
                f.write(line + '\n')

    print("Konvertering færdig!")

if __name__ == "__main__":
    main()