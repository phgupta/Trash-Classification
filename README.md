# Trash-Classification 

## To Do

1. Change collect_split_data() to implement the following dataset structure,
    Dataset -> Compost -> one.jpg, two.jpg, ...
               Landfill -> one.jpg, two.jpg, ...
2. Check if cnn_model() -> layer2 = create_new_conv_layer(..., NUM_FILTERS*2, ...) is correct.
3. Change constants in cnn_model() -> fully connected layer
4. Change IMAGE_SIZE_RESHAPE.
5. Change structure of code - https://danijar.com/structuring-your-tensorflow-models/
6. Add GPU support.
