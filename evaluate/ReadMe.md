# Evaluation
After train and evaluate the scannet dataset, you are expected to generate the labeled pointclouds for ScanNet evaluation dataset located in the ../output folder. We show how we evaluate the results.

### Compile the code
	sh compile.sh

### Evaluate in one script
	sh evaluate.sh
By doing this, we will provide the predicted labels consistent with the original vertices' order insider ../output/pred_final. The algorithm is simple: We have predicted labels from the resampled grid positions. We determine the labels of the original mesh vertices by pasting the label from the nearest grid positions.