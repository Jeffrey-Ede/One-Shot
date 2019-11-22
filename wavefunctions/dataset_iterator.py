"""Utility function to load and iterate over wavefunction dataset."""

import tensorflow as tf

import os
from scipy.misc import imread

def load_data(
    parent_dir,
    output_types,
    output_shapes,
    batch_size, 
    prefetch_size
    ):

    """sample a batch of examples."""
        
    def generator():
        #Get a list of example filepaths
        os.listdir(parent_dir)

        num_examples = None

        while True:
            #Select random example
            example_num = np.random.randint(0, num_examples)

            #Output
            amplitude_filepath = None
            phase_filepath = None

            amplitude = imread(amplitude_filepath, mode="F")
            phase = imread(phase_filepath, mode="F")

            output = np.stack([amplitude, phase], axis=-1)

            #Input 
            input = None

            yield input, output

    ds = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=output_types,
        output_shapes=output_shapes
        )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_size)

    #Make iterator
    iters = ds.make_one_shot_iterator().get_next()

    #Add batch dimension size to graph
    for iter in iters:
        iter.set_shape([batch_size]+iter.get_shape().as_list()[1:])

    return iters

if __name__ == "__main__":
    pass