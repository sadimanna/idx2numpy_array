# idx2numpy_array

To convert idx formatted files to numpy array ::

**Just run the code** with the **_proper file path_ in the dictionary _filename_**

The section **Extra Stuffs** in the code is used to speed up the stacking process in the numpy array
Actually stacking all at once was taking **polynomial time** according to my naive calculations. So, I decided to divided the whole datasets into sub-blocks of 1000 and then a bigger sub-block of 10000.

And HOLA!! The time taken for converting all the 60000 images and stacking them up in a single numpy array to form a 3D array was almost 52 secs on my Laptop.

The numpy arrays can further be converted to images usiing other librarie like opencv, PIL or scipy.

But for using in a CNN, keeping in the np.ndarray format will be useful. I guess!!
