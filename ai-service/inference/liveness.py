import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        print("Face Alive")

    cv2.imshow(
        "Liveness",
        frame
    )

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()