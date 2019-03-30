# Train the model
In this document, we show the full process of training the model.

### Compile the kernels
You may need to change the include and library path inside the compilation scripts.
```
cd tf_ops/grouping
sh tf_grouping_compile.sh
cd ../interpolation
sh tf_interpolate_compile.sh
cd ../sampling
sh tf_sampling_compile.sh
cd ../..
```
### Test the model
Download our trained model from [**this link**](http://download.cs.stanford.edu/orion/texturenet/checkpoint.zip) and unzip to this directory. Edit evaluate.sh to specify the directory that holds the scannet-chunks data.
```	
sh evaluate.sh
```
This will provide the predicted labels of the given positions in ../output.

Note that this will produce the results for the evaluation set. For generating predictions of the test set, change --evaluate flag from 1 to 2 in evaluate.sh.

Then, following the instruction in [**evaluate**](https://github.com/hjwdzh/TextureNet/raw/master/evaluate/) to compute the final numbers.

### Train the model
Edit train.sh to specify the directory that holds the scannet-chunks data.
```
sh train.sh
```
It will takes 3~5 days and around 1000 epochs to achieve the reported performance in the [**leaderboard**](http://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=17).