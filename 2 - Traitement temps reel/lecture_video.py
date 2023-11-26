import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

def predict_video(frame, model, classes):
    # Ensure the frame has the correct shape
    new_frame = cv2.resize(frame, (model.input_shape[1], model.input_shape[2]))

    x = tf.keras.utils.img_to_array(new_frame, data_format='channels_last')

    x = np.expand_dims(x, axis=0)

    # predict
    pred = model.predict(x, batch_size=1, verbose = 0)[0]

    # Check if fire
    detection_label = None
    for (pos, prob) in enumerate(pred):
        class_name = classes[pos]
        if pos == np.argmax(pred):
            return f"{class_name}: {(prob*100):.2f}%", class_name
        
    return detection_label, None

def reading_video():
    classes = ['fire', 'no_fire', 'start_fire']

    #index video
    ind_video = 0

    # Load your pre-trained models
    model = load_model("../4 - Modele/Groupe9_DB3_VGG16_30_16.h5")  # Replace with your model file

    # Open a video file
    cap = cv2.VideoCapture(f"videos/{classes[ind_video]}.mp4")  # Replace with your video file path

    # Get video frame properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec as needed
    out_annotated = cv2.VideoWriter('annotated_video.mp4', fourcc, 20.0, (frame_width, frame_height))

    # Initialize variables to keep track of the previous class name
    prev_class_name = None
    prev_frame = None
    num_img = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            ind_video+=1
            if ind_video == len(classes):
                break
            cap = cv2.VideoCapture(f"videos/{classes[ind_video]}.mp4")
            ret, frame = cap.read()
            prev_class_name = None
            prev_frame = None
            num_img = 0
                

        detection_label, class_name = predict_video(frame, model, classes)
        
        # Copy the frame without annotation
        frame_not_annoted = frame.copy()
        
        # Display the resulting frame with the detection label
        cv2.putText(frame, detection_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
        # Save the frame to the video file
        out_annotated.write(frame)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Save pictures when class_name changes or when new class_name
        if prev_class_name is not None and class_name != prev_class_name:
            cv2.imwrite(f"images/{classes[ind_video]}/{num_img}_previous_frame.jpg", prev_frame)
            cv2.imwrite(f"images/{classes[ind_video]}/{num_img}_current_frame.jpg", frame_not_annoted)
            num_img += 1
        elif prev_class_name is None:
            cv2.imwrite(f"images/{classes[ind_video]}/starter.jpg", frame_not_annoted)
        
        prev_frame = frame_not_annoted
        prev_class_name = class_name
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and video writers when done
    cap.release()
    out_annotated.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    reading_video()