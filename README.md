# GENDER-DETECTION-
import cv2
import os
import urllib.request
import sys


# Model files configuration
MODEL_FILES = {
    'face_proto': {
        'url': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'file': 'deploy.prototxt'
    },
    'face_model': {
        'url': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        'file': 'res10_300x300_ssd_iter_140000.caffemodel'
    },
    'age_proto': {
        'url': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt',
        'file': 'age_deploy.prototxt'
    },
    'age_model': {
        'url': 'https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel',
        'file': 'age_net.caffemodel'
    },
    'gender_proto': {
        'url': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt',
        'file': 'gender_deploy.prototxt'
    },
    'gender_model': {
        'url': 'https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel',
        'file': 'gender_net.caffemodel'
    }
}

MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def download_file(url, filename):
    """Download file with progress"""
    try:
        print(f"Downloading {filename}...")

        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{filename}: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, reporthook)
        print(f"\n✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        return False


def download_all_models():
    """Download all required model files if they don't exist"""
    print("=" * 60)
    print("Checking and downloading required model files...")
    print("=" * 60)

    all_exist = True
    for key, info in MODEL_FILES.items():
        file_path = info['file']
        if not os.path.exists(file_path):
            all_exist = False
            if not download_file(info['url'], file_path):
                print(f"\n✗ Failed to download {file_path}")
                print("Please download manually from:")
                print(f"   {info['url']}")
                return False
        else:
            print(f"✓ {file_path} already exists")

    if all_exist:
        print("\n✓ All model files are ready!")
    else:
        print("\n✓ All downloads completed successfully!")

    print("=" * 60)
    return True


def load_models():
    """Load the pre-trained models"""
    print("\nLoading AI models into memory...")

    face_net = cv2.dnn.readNet(
        MODEL_FILES['face_model']['file'],
        MODEL_FILES['face_proto']['file']
    )

    age_net = cv2.dnn.readNet(
        MODEL_FILES['age_model']['file'],
        MODEL_FILES['age_proto']['file']
    )

    gender_net = cv2.dnn.readNet(
        MODEL_FILES['gender_model']['file'],
        MODEL_FILES['gender_proto']['file']
    )

    print("✓ Models loaded successfully!")
    return face_net, age_net, gender_net


def detect_faces(frame, face_net, conf_threshold=0.7):
    """Detect faces in frame"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append([x1, y1, x2, y2])

    return faces


def predict_age_gender(face_img, age_net, gender_net):
    """Predict age and gender from face image"""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age


def process_frame(frame, face_net, age_net, gender_net):
    """Process single frame for face detection and age/gender prediction"""
    display_frame = frame.copy()
    faces = detect_faces(frame, face_net)

    for (x1, y1, x2, y2) in faces:
        # Add padding
        padding = 20
        y1_pad = max(0, y1 - padding)
        y2_pad = min(frame.shape[0], y2 + padding)
        x1_pad = max(0, x1 - padding)
        x2_pad = min(frame.shape[1], x2 + padding)

        face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            gender, age = predict_age_gender(face_img, age_net, gender_net)

            # Draw rectangle around face
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label background
            label = f'{gender}, {age}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Draw background rectangle for text
            cv2.rectangle(display_frame,
                          (x1, y1 - 35),
                          (x1 + label_size[0] + 10, y1),
                          (0, 255, 0),
                          -1)

            # Draw text
            cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return display_frame


def webcam_mode(face_net, age_net, gender_net):
    """Real-time webcam detection"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n✗ Error: Cannot access webcam")
        print("Please check if:")
        print("  1. Your webcam is connected")
        print("  2. No other application is using the webcam")
        print("  3. You have given permission to access the webcam")
        return

    print("\n" + "=" * 60)
    print("✓ Webcam started successfully!")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'S' to save screenshot")
    print("=" * 60 + "\n")

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Error: Cannot read frame from webcam")
            break

        # Process frame
        processed_frame = process_frame(frame, face_net, age_net, gender_net)

        # Add instructions on frame
        cv2.putText(processed_frame, "Press 'Q' to Quit | 'S' to Save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Gender & Age Detection - Webcam Mode', processed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n✓ Exiting webcam mode...")
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, processed_frame)
            print(f"✓ Screenshot saved as: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def image_mode(image_path, face_net, age_net, gender_net):
    """Process single image"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"\n✗ Error: Cannot read image from {image_path}")
        print("Please check if the file exists and is a valid image.")
        return

    print(f"\n✓ Processing image: {image_path}")

    processed_frame = process_frame(frame, face_net, age_net, gender_net)

    # Save result
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, processed_frame)
    print(f"✓ Result saved as: {output_path}")

    # Display result
    cv2.imshow('Gender & Age Detection - Press any key to close', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_mode(video_path, face_net, age_net, gender_net):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"\n✗ Error: Cannot open video {video_path}")
        print("Please check if the file exists and is a valid video.")
        return

    print(f"\n✓ Processing video: {video_path}")
    print("Press 'Q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✓ Video processing completed!")
            break

        processed_frame = process_frame(frame, face_net, age_net, gender_net)

        cv2.putText(processed_frame, "Press 'Q' to Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Gender & Age Detection - Video Mode', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ Exiting video mode...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("\n" + "=" * 60)
    print(" " * 15 + "GENDER AND AGE DETECTION SYSTEM")
    print("=" * 60)

    # Download models if needed (this will auto-download .caffemodel files)
    if not download_all_models():
        print("\n✗ Setup failed. Please try again or download files manually.")
        return

    # Load models
    try:
        face_net, age_net, gender_net = load_models()
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        print("Please ensure all model files are present and valid.")
        return

    # Menu
    print("\n" + "=" * 60)
    print("Choose detection mode:")
    print("=" * 60)
    print("1. Webcam (Real-time)")
    print("2. Image file")
    print("3. Video file")
    print("=" * 60)

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == '1':
        webcam_mode(face_net, age_net, gender_net)

    elif choice == '2':
        img_path = input("Enter image path: ").strip()
        img_path = img_path.strip('"').strip("'")  # Remove quotes if any
        if os.path.exists(img_path):
            image_mode(img_path, face_net, age_net, gender_net)
        else:
            print(f"\n✗ Error: File not found - {img_path}")

    elif choice == '3':
        vid_path = input("Enter video path: ").strip()
        vid_path = vid_path.strip('"').strip("'")  # Remove quotes if any
        if os.path.exists(vid_path):
            video_mode(vid_path, face_net, age_net, gender_net)
        else:
            print(f"\n✗ Error: File not found - {vid_path}")

    else:
        print("\n✗ Invalid choice! Please run the program again.")

    print("\n" + "=" * 60)
    print("Thank you for using Gender and Age Detection System!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
