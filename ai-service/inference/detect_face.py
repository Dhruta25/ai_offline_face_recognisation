import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

def detect_face(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not find or open the image at '{image_path}'.")
        return None
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_detection.process(rgb)

    if not results.detections:
        print("No face found")
        return None

    detection = results.detections[0]

    bbox = detection.location_data.relative_bounding_box

    h,w,_ = image.shape

    x = int(bbox.xmin*w)
    y = int(bbox.ymin*h)
    bw = int(bbox.width*w)
    bh = int(bbox.height*h)

    face = image[y:y+bh,x:x+bw]

    cv2.imwrite(
        "cropped_face.jpg",
        face
    )

    print("Face saved")

    return face


if __name__ == "__main__":
    detect_face("/Users/dhrutamacm2/ai_offline_face_recognisation/ai-service/inference/test.jpg")