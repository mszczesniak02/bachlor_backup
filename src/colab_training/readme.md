# Colab settings for easier config

In order to use Google Colab, both training and test datasets need to be uploaded to Google Drive, and then mounted to Colab.
```python
    print("todo")
```

Then, clone the repository in Colab notebook
```
!git clone https://github.com/mszczesniak02/bachlor_backup.git
```

Lastly, run the setup.py scipy, which sets the "on_colab" parameters, resulting in changing paths and using the Colab GPU (free tier).
Either run directly from Colab Notebook, or run in Colab terminal (remove '!' from the line below):
```
    !python setup.py
```
