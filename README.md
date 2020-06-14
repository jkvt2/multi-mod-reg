# multi-mod-reg
Loss function for regressing against multi-modal ground truths.

Using deep learning models to regress continuous outputs using the typical L2 loss fails when the true distribution is multi-modal.\
![Graphs of input, output, results of standard l2 loss, results of multi-modal reg loss](https://github.com/jkvt2/multi-mod-reg/blob/master/mmr.png)\
In the above figure, a toy dataset is generated in which there are two classes (blue and orange).\
First row: a histogram of 1000 example inputs. The two classes are expected to be easily separated by the model.\
Second row: a histogram of target ground truth values for the same 1000 examples. The expected behaviour is straightforward: if the input is from the orange class, the target output is input + 1; if the input is from the blue class, the target output is input + either 0 or 1.\
Third row: if we train a model to regress this target output with l2 loss, the model will predict instead around 1.5 for blue class as this minimizes the l2.\
Fourth row: I tried to make a new output representation (gaussian mixture) and corresponding loss function. As the predicted output is now a distribution rather than a number, I sampled from it in order to obtain this histogram.\
