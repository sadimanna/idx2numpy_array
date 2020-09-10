# idx2numpy_array

# **To convert idx formatted files to numpy array ::**

# 1. For the file **idx2nparray_py3.py** ::

**Just run the code** with the **_proper file path_ in the dictionary _filename_**

This code is a mixture of the above two.

I have stacked blocks of **10000 images** after reading the data of **10000X28X28 bytes = 7840000 bytes** at once.

And the time taken for reading _7840000_ bytes at once and stacking them all in _6 iterations_ in a single numpy array to form a _3D array_ was about **5 seconds** on my laptop.

The numpy arrays can further be converted to images using other libraries like **opencv**, **PIL** or **scipy**.

But for using in a **CNN**, keeping in the **np.ndarray** format will be useful. I guess!!

# **To convert numpy array to image ::**

**Run the file ndarr2img.py**

Takes about less than 20 seconds for the trainiing images and much less than that for the test images.

_Just enter the right path_

The iamges will be saved in '_.jpg_' format

And the labels will be saved in '_.npy_' format
