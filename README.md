# Tranformer Model
## Sources
based on Andrej Karpathy's model using his [Video on Youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY&pp=ygUOY3JlYXRlIGNoYXRncHQ%3D)
## Comments
The code includes comments and notes on how the model operates
## Running the code
Simply run the [transformer](transformer.ipynb) file, which will take the [input](input.txt) file and train the model based on its contents
Note that the parameters are set very high and may therefore take very long to load. This is to show that a long character base (block size) will develop a more human-friendly language model. Modify the hyper parameters at the top of the code to change up the model.
The code is also automaticated to be GPU-friendly
## Other files
All files excluding the [transformer.ipynb](transformer.ipynb) file and [input.txt](input.txt) were part of the development process and can be ignored
