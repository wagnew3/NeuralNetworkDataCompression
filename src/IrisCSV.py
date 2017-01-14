import tensorflow as tf

directory = "/home/willie/workspace/TensorFlow/data/iris/*.csv"
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(directory),
    shuffle=True)

# Each file will have a header, we skip it and give defaults and type information
# for each column below.
line_reader = tf.TextLineReader(skip_header_lines=1)

_, csv_row = line_reader.read(filename_queue)

# Type information and column names based on the decoded CSV.
record_defaults = [[0.0], [0.0], [0.0], [0.0], [""]]
sepal_length, sepal_width, petal_length, petal_width, iris_species = \
    tf.decode_csv(csv_row, record_defaults=record_defaults)

# Turn the features back into a tensor.
features = tf.pack([
    sepal_length,
    sepal_width,
    petal_length,
    petal_width])

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # We do 10 iterations (steps) where we grab an example from the CSV file. 
    for iteration in range(1, 11):
        # Our graph isn't evaluated until we use run unless we're in an interactive session.
        example, label = sess.run([features, iris_species])

        print(example, label)
    coord.request_stop()
    coord.join(threads)
