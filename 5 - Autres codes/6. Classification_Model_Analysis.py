def print_image_prediction(image_path, label, value):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    font = cv2.FONT_HERSHEY_COMPLEX
    textsize = cv2.getTextSize(label, font, 1, 2)[0]
    textX = (img.shape[1] - textsize[0]) / 2
    textY = (img.shape[0] + textsize[1]) / 2
    cv2.putText(img, "%s : %.2f%%" % (label, float(value) * 100), (int(textX)-100, int(textY)-150), font, 1, (255,0,0), 3, cv2.LINE_AA)
    plt.imshow(img)
    plt.show()

def compute_display_errors(test_dataset_path,y_true,y_pred):
  i=0
  for class_name in test_ds.class_names:
    class_path = os.path.join(test_dataset_path, class_name)
    for img_name in os.listdir(class_path):
        image_path = os.path.join(class_path, img_name)
        img = Image.open(image_path).convert('RGB')
        x = tf.keras.utils.img_to_array(img,data_format='channels_last')
        x = tf.keras.preprocessing.image.smart_resize(x, size=(input_dim,input_dim))
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)[0]
        prediction_index = np.argmax(predictions)
        y_true.append(classes.index(class_name))
        y_pred.append(prediction_index)
        if classes.index(class_name) != prediction_index:
            print(image_path)
            print_image_prediction(image_path, classes[prediction_index], max(predictions))
            i=i+1
  print("Number of errors", i)
  
y_true = []
y_pred = []
compute_display_errors(test_dataset,y_true,y_pred)