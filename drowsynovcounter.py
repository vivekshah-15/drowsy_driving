from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import pyttsx3
from pygame import mixer
import random
from twilio.rest import Client

# Initialize the text-to-speech engine
engine = pyttsx3.init()


def say_text(text, volume=1.0):
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize the mixer module
mixer.init()
mixer.music.load("music.wav")

cap = cv2.VideoCapture(0)
flag = 0
last_alert_time = time.time()  # Initialize the time of the last alert
music_playing = False  # Track if the music is currently playing
volume_step = 0.2  # Volume increase step for each alert repetition
positive_reinforcement_interval = (
    5  # Interval for providing positive reinforcement (in seconds)
)

positive_reinforcement_last_time = (
    time.time()
)  # Initialize the last time positive reinforcement was given
last_history_update_time = (
    time.time()
)  # Initialize the time of the last alertness history update
last_reminder_time = time.time()  # Initialize the time of the last reminder display
last_phone_call_time = time.time()  # Initialize the time of the last phone call
phone_call_made = False  # Initialize the flag for phone call made
alert_counter=0
alertness_history=[]

# Twilio credentials
account_sid = "AC65a69995353e54c2d7ca9eee8e888019"
auth_token = "0b34f2ab9c09858d093dfb757ba43cea"
twilio_phone_number = "+18507265636"
recipient_phone_number = "+918169161530"


def make_phone_call(message):
    global phone_call_made
    print("Making phone call...")
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        twiml="<Response><Say>" + message + "</Say></Response>",
        to=recipient_phone_number,
        from_=twilio_phone_number,
    )

    print(call.sid)
    phone_call_made = True


while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                say_text("Alert! You seem drowsy. Please stay awake.", volume=1.0)
                print("Alert! You seem drowsy.")
                current_time = time.time()
                # Check if more than 5 seconds have elapsed since the last alert
                if current_time - last_alert_time > 5:
                    say_text("Alert! You may be sleeping.", volume=1.0)
                    print("Alert! You may be sleeping.")
                    last_alert_time = current_time  # Update the time of the last alert
                    # Start playing music if not already playing
                    if not music_playing:
                        mixer.music.play(-1)  # Loop the music indefinitely
                        music_playing = True
                else:
                    # Increase volume for subsequent alerts
                    say_text(
                        "Alert! You may be sleeping.", volume=min(1.0, volume_step)
                    )
                    volume_step += 0.2
                # Increment the alert counter
                alert_counter += 1

        else:
            flag = 0
            # Stop playing music if currently playing
            if music_playing:
                mixer.music.stop()
                music_playing = False
            # Reset volume step when eyes are open
            volume_step = 0.2

            # Check if it's time to give positive reinforcement
            current_time = time.time()
            if (
                current_time - positive_reinforcement_last_time
                > positive_reinforcement_interval
            ):
                positive_reinforcement_last_time = current_time
                # Generate and speak a positive reinforcement message
                positive_messages = [
                    "Great job! You seem alert and focused.",
                    "You're doing well! Keep up the good work.",
                    "Looking good! Stay attentive and engaged.",
                ]
                say_text(
                    random.choice(positive_messages), volume=0.7
                )  # Lower volume for positive reinforcement
                print("Positive reinforcement triggered.")

    # Update alertness history with the current timestamp and ear value
    if time.time() - last_history_update_time >= 15:
        last_history_update_time = (
            time.time()
        )  # Update the time of the last history update
        alertness_history.append((time.time(), ear))

        # Analyze alertness history and provide personalized feedback or recommendations
        if len(alertness_history) > 0:
            average_ear = sum(ear for _, ear in alertness_history) / len(
                alertness_history
            )
            if average_ear < 0.3:
                say_text(
                    "You seem drowsy. Consider taking a break or getting some rest."
                )
            elif average_ear >= 0.3:
                say_text("You seem quite alert! Keep it up.")

            # Clear alertness history
            alertness_history = []

    # Display a reminder message periodically
    if time.time() - last_reminder_time >= 30:  # Display reminder every 5 minutes
        cv2.putText(
            frame,
            "Stay Alert!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        last_reminder_time = time.time()  # Update the time of the last reminder display

    # Check if the alert counter has reached the threshold
    if alert_counter >= 5:
        say_text(
            "You've been alerted multiple times. It's advisable to take a break or rest."
        )
        alert_counter = 0  # Reset the alert counter

    # Check if the person doesn't wake up after 60 seconds
    if time.time() - last_alert_time >= 60:
        if not phone_call_made:  # Check if a call hasn't been made yet
            make_phone_call("The driver is not waking up. Please check on them.")
        else:
            print("Phone call already made.")
        last_phone_call_time = time.time()  # Update the time of the last phone call
        phone_call_made = True  # Set the flag for phone call made

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
