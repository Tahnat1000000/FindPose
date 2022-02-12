import mediapipe, os, time, cv2, numpy

base_dir = os.path.dirname(os.path.abspath(__file__))
imagepath = os.path.join(base_dir, "images", "image1.jpg") # IMAGE PATH

fxRes = 0.5
fyRes = 0.5

mpDraw = mediapipe.solutions.drawing_utils
mpPose = mediapipe.solutions.pose
pose = mpPose.Pose()

image = cv2.resize(cv2.imread(imagepath), (0, 0), fx=fxRes, fy=fyRes)
rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgbImage)


### DRAW LANDMARKS ON IMAGE
# if results.pose_landmarks:
#     mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


### GETTING NECESSARY DATA FROM LANDMARKS RESULT
left_hip, right_hip = None, None
left_knee, right_knee = None, None
left_ankle, right_ankle = None, None
for id, ln in enumerate(results.pose_landmarks.landmark): # save landmarks we need and save them
    if id == 23:
        left_hip = [ln.x,ln.y]
    elif id == 24:
        right_hip = [ln.x,ln.y]
    elif id == 25:
        left_knee = [ln.x,ln.y]
    elif id == 26:
        right_knee = [ln.x,ln.y]
    elif id == 27:
        left_ankle = [ln.x,ln.y]
    elif id == 28:
        right_ankle = [ln.x,ln.y]

lmaxX = left_hip[0] + 0.08
lminX = left_hip[0] - 0.08
rmaxX = right_hip[0] + 0.08
rminX = right_hip[0] - 0.08


### COMPARE LANDMARKS TO DECIDE IF PEOPLE SITTING OR STANDING
if (left_knee[0] < lmaxX and left_knee[0] > lminX and left_ankle[0] > lminX and left_ankle[0] < lmaxX) and (right_knee[0] < rmaxX and right_knee[0] > rminX and right_ankle[0] > rminX and right_ankle[0] < rmaxX) and (left_knee[1]-left_hip[1]) > 0.1 and (left_ankle[1]-left_knee[1]) > 0.1:
    cv2.putText(image, 'STANDING', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(image, 'SITTING', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

### SHOW IMAGE
cv2.imshow("picture", image)


cv2.waitKey(0)
cv2.destroyAllWindows()