#include "stdio.h"
#include "time.h"
#include "stdlib.h"
const int inputDim = 65;
const int h1Dim = 100;
const int h2Dim = 32;
const int outputDim = 1;

int topology[] = {2, 3, 2, 1};
int depth = 4; //length of topology
float* weights;
float* biases;
int totalW = 0;
int totalB = 0;
void countParameters(int* tw, int* tb){
  for (int i = 0; i < depth-1; i++) *tw += topology[i]*topology[i+1];
  for (int i = 0; i < depth; i++) *tb += topology[i];
}
// add up total amount of weights and biases
// may regret refactoring this code like this in a minute...
float error = 0;
float* output;
float rando() { return (float)rand()/RAND_MAX; } //[0,1]

float ReLU(float x) {return x>0.0 ? x:0.0;}
float dReLU(float x) {return x>0.0 ? 1.0:0.0;}
void printArr(int n, float arr[])
{
	for (int i = 0; i < n; i++)
	{
    printf("%f ", arr[i]);
	}
  printf("\n");
}

void setZero() 
{
  for (int i = 0; i < totalW; i++){
    weights[i] = 0.0;
  }
  for (int i = 0; i < totalB; i++){
    biases[i] = 0.0;
  }
}

int initialize(int* topology, int depth)
{
  countParameters(&totalW, &totalB);
  weights = malloc(sizeof(float) * totalW);
  biases = malloc(sizeof(float) * totalB);
  setZero();
  srand(time(0));
  for (int i = 0 ; i < totalW ; i++){
    weights[i] = random();
  }
  for (int i = 0 ; i < totalB ; i++){
    biases[i] = random();
  }
}
// 	void forwardProp(float input[inputDim])
// 	{
// 		StockBear::setZero();
// 		for (int i = 0; i < inputDim;i++)
// 		{
// 			inputLayer[i] = input[i];
// 		}
// 		for (int i = 0; i < h1Dim; i++)
// 		{
// 			for (int j = 0; j < inputDim; j++)
// 			{
// 				h1neurons[i] += input[j] * h1weights[j][i];
// 			}
// 			h1neurons[i] += h1biases[i];
// 			h1neurons[i] = ReLU(h1neurons[i]);
// 		}
// 		for (int i = 0; i < h2Dim; i++)
// 		{
// 			for (int j = 0; j < h1Dim; j++)
// 			{
// 				h2neurons[i] += h1neurons[j] * h2weights[j][i];
// 			}
// 			h2neurons[i] += h2biases[i];
// 			h2neurons[i] = ReLU(h2neurons[i]);
// 		}
// 		for (int i = 0; i < outputDim; i++)
// 		{
// 			for (int j = 0; j < h2Dim; j++)
// 			{
// 				output[i] += h2neurons[j] * outweights[j][i];
// 			}
// 			output[i] += outbiases[i]; //linear output when doing regression
// 		}
//
// 	}
// 	void backProp(float target[], float inputs[][inputDim], int batchSize) {
//     // Accumulate gradients over the mini-batch
//     float outputGradients[outputDim] = {0.0};
//
//     for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
//         forwardProp(inputs[batchIndex]);
//
//         // Calculate output layer gradients for each sample in the mini-batch
//         for (int i = 0; i < outputDim; ++i) {
//             outputGradients[i] += output[i] - target[batchIndex];
//         }
//     }
// 	error = outputGradients[0]/batchSize;
//
//     // Update output layer weights and biases
//     for (int i = 0; i < h2Dim; ++i) {
//         for (int j = 0; j < outputDim; ++j) {
//             outweights[i][j] -= eta * (outputGradients[j] / batchSize) * h2neurons[i];
//         }
//         outbiases[i] -= eta * outputGradients[i] / batchSize;
//     }
//
//     // ... Similar updates for hidden layer 2 and hidden layer 1
//
//     // Calculate hidden layer 2 gradients
//     float h2Gradients[h2Dim] = {0.0};
//     for (int i = 0; i < h2Dim; ++i) {
//         for (int j = 0; j < outputDim; ++j) {
//             h2Gradients[i] += outputGradients[j] * outweights[i][j];
//         }
//         h2Gradients[i] *= dReLU(h2neurons[i]);
//     }
//
//     // ... Similar updates for hidden layer 1
//
//     // Calculate hidden layer 1 gradients
//     float h1Gradients[h1Dim] = {0.0};
//     for (int i = 0; i < h1Dim; ++i) {
//         for (int j = 0; j < h2Dim; ++j) {
//             h1Gradients[i] += h2Gradients[j] * h2weights[i][j];
//         }
//         h1Gradients[i] *= dReLU(h1neurons[i]);
//     }
//
//     // ... Similar updates for input layer
//
//     // Update input layer weights and biases
//     for (int i = 0; i < inputDim; ++i) {
//         for (int j = 0; j < h1Dim; ++j) {
//             h1weights[i][j] -= eta * (h1Gradients[j] / batchSize) * inputLayer[i];
//         }
//         h1biases[i] -= eta * h1Gradients[i] / batchSize;
//     }
// }
// 	float predict(float inp[inputDim])
// 	{
// 		StockBear::forwardProp(inp);
// 		return output[0];
// 	}
//
//
// 	int inputLayer[inputDim];
// 	float h1weights[inputDim][h1Dim]; //matrix representing weights between input and HL1
//     float h1biases[h1Dim]; //array representing biases of HL1
//     float h2weights[h1Dim][h2Dim]; //matrix representing weights between HL2 and HL1
//     float h2biases[h2Dim];//array representing biases ofHL1
//     float outweights[h2Dim][outputDim];//matrix representing weights between HL2 and output
//     float outbiases[outputDim]; //array representing biases between input and HL1
//     float h1neurons[h1Dim]; //array representing HL1 activations
//     float h2neurons[h2Dim]; //array representing HL 2 activations
// 	float trainingExs = 0;
// 	float eta = .005;
// 	//float alpha =  .2;

	

int main()
{
	int EPOCHS = 100;
	const int batchSize = 15;
	float input[batchSize][inputDim];
	float target[batchSize];
	float XOR[12] = {0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0};
	//cout << stockbear.predict(input) << endl;
	float errorE = 10;
		for (int i = 0; i<EPOCHS; i++)
		{	
			loadIO(batchSize, input, target);
			//cout<< input[0] <<input[1]<<target[0]<<endl;
			//stockbear.forwardProp(input);
			stockbear.backProp(target, input, batchSize);
			cout << "Epoch " << i+1 << "'s error: " << stockbear.error << endl;
			//cout<<"epoch " << i<< " ";
		}
	//cout << endl << stockbear.error << endl;
	stockbear.saveModelText("stockbear.h");//parse with python
	return 0;
}
