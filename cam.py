import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start capturing video input
cap = cv2.VideoCapture(1)


def are_arms_crossed(landmarks):
    # Extract landmarks for left and right wrists and elbows
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    # Check if left wrist is to the right of right wrist
    if left_wrist.x < right_wrist.x:
        return True


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose
    results = pose.process(image)

    # Draw the pose annotation on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Check if arms are crossed
        if are_arms_crossed(results.pose_landmarks.landmark):
            cv2.putText(
                image,
                "Arms Crossed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                "Arms Not Crossed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Display the image
    cv2.imshow("Pose Detection", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
