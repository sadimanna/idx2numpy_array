# idx2numpy_array

To convert idx formatted files to numpy array ::

# 1. For the file **idx2nparr.py** ::

**Just run the code** with the **_proper file path_ in the dictionary _filename_**

The section **Extra Stuffs** in the code is used to speed up the stacking process in the numpy array
Actually stacking all at once was taking **polynomial time** according to my naive calculations (2.5 hours I calculated -_-). So, I decided to divide the whole datasets into sub-blocks of 1000 and then a bigger sub-block of 10000.

And the time taken for converting all the _60000 images_ and stacking them up in a single numpy array to form a _3D array_ was almost **52 secs** on my Laptop.

# 2. For the file **idx2numpyarray.py** ::

**Just run the code** with the **_proper file path_ in the dictionary _filename_**

No section for **Extra Stuffs** in the code.
Reads the whole file at once and converts to a numpy array and then reshapes it.

And the time taken for reading all the _47040000_ bytes in the file and reshaping them up in a single numpy array to form a _3D array_ was almost **7 secs** on my Laptop.

# 3. For the file **idx2ndarray.py** ::

**Just run the code** with the **_proper file path_ in the dictionary _filename_**

This code is a mixture of the above two.

I have stacked blocks of **10000 images** after reading the data of **10000X28X28 bytes = 7840000 bytes** at once.

And the time taken for reading _7840000_ bytes at once and stacking them all in _6 iterations_ in a single numpy array to form a _3D array_ was about **5.30509209633 seconds** on my laptop.

The numpy arrays can further be converted to images using other libraries like **opencv**, **PIL** or **scipy**.

But for using in a **CNN**, keeping in the **np.ndarray** format will be useful. I guess!!
