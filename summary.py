from nets.yolo import yolo_body

if __name__ == "__main__":
    input_shape = [640, 640, 3]
    num_classes = 80

    model = yolo_body(input_shape, num_classes, 's')

    model.summary()
    for i in range(len(model.layers)): 
        print(i, model.layers[i].name)