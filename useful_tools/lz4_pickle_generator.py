"""
The data_preprocessing creates
"""
import compress_pickle
import os
from pathlib import Path


def swap_lzma_to_lz4(location):
    """
    Makes lz4 files for all the lzma files in a directory. Can take a long time.
    Also has a check for whether the files already have an lz4. Ensure to delete
    them if you want it to change, because it won't compare contents
    :param location:
    :return:
    """

    files = os.listdir(location)
    files = filter(lambda x: x.endswith('lzma'), files)

    for f in files:
        if _check_if_lz4_exists(location, f):
            continue

        item = compress_pickle.load(location + f)
        print("Swapping compression of", str(location + f))
        save_to_pickle(item, location + _change_lzma_to_lz4_name(f))


def _change_lzma_to_lz4_name(filename):
    filename = filename[:-5]
    filename += '.lz4'
    return filename


def _check_if_lz4_exists(location, filename):
    """
    Checks the file path whether there are already lz4 files
    :param location:
    :param filename:
    :return:
    """
    # This just isn't a very readable syntax so I abstracted it
    return Path(location + _change_lzma_to_lz4_name(filename)).is_file()


def save_to_pickle(item_to_save, location):
    """Location must end with .type - i.e. lzma for this project"""
    try:
        f = open(location, 'wb')
    except FileNotFoundError:
        # The specified path probably doesn't exist
        # Use mkdir and try again.
        print("WARNING: Pickle path does not exist. Creating... If this message persists, something is wrong")
        os.makedirs(os.path.dirname(location))

        save_to_pickle(item_to_save, location)
        return
    f.close()
    compress_pickle.dump(item_to_save, location)


if __name__ == '__main__':
    swap_lzma_to_lz4('../pickles/')
