from imageai.Detection.Custom import CustomObjectDetection
import os

execution_path = os.getcwd()
model_path= "apple_dataset\models\detection_model-ex-028--loss-8.723.h5"
json_path="apple_dataset\json\detection_config.json"
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.setJsonPath(json_path)
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="input\damaged_apple.jpg", minimum_percentage_probability=60, output_image_path="output\image-new.jpg")

count=0
for detection in detections:
    count+=1
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("--------------------------------")
print("Number of apples:", count)