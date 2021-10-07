import os
import compress_pickle


def save_lzma_pickle(item, location):
    # Force it to use .lz4
    if location[-4:] != '.lz4':
        location += '.lz4'

    try:
        f = open(location, 'wb')
    except FileNotFoundError:
        # The specified path probably doesn't exist
        # Use mkdir and try again.
        print("WARNING: Pickle path does not exist. Attempting to create... ")
        os.makedirs(os.path.dirname(location))
        f = open(location, 'wb')
    f.close()
    compress_pickle.dump(item, location)


def load_pickle(location):
    """Location must end with .type - i.e. lzma for this project"""
    # This doesn't really need to be its own method, but
    # it's here just for consistency
    item = compress_pickle.load(location)
    return item
