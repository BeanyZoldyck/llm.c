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
float* neurons;
int totalW = 0;
int totalB = 0;
int totalN = 0;

void countParameters(int* tw, int* tb){
  for (int i = 1; i < depth; i++){
    *tw += topology[i]*topology[i-1];
    *tb += topology[i];
  } 
}
// add up total amount of weights and biases
// may regret refactoring this code like this in a minute...
float error = 0;
float rando() { return 2*((float)rand()/RAND_MAX)-1; } //[0,1]

float ReLU(float x) {return x>0.0 ? x:0.0;}
float dReLU(float x) {return x>0.0 ? 1.0:0.0;}
void printArr(int n, float* arr)
{
	for (int i = 0; i < n; i++)
	{
    printf("%f ", arr[i]);
	}
  printf("\n");
}

void setZero() 
{
  for (int i = 0; i < totalB; i++){
    neurons[i] = 0.0;
  }
}

int initialize(int* topology, int depth, float** weights, float** biases, float** neurons){
  countParameters(&totalW, &totalB);
  *weights = malloc(sizeof(float) * totalW);
  *biases = malloc(sizeof(float) * totalB);
  *neurons = malloc(sizeof(float)*(totalB + topology[0]));
  setZero();
}
int randomize(int* topology, int depth){
  srand(time(0));
  for (int i = 0 ; i < totalW ; i++){
    weights[i] = rando();
  }
  for (int i = 0 ; i < totalB ; i++){
    biases[i] = rando();
  }
}
void forwardProp(int* topology, int depth, float* input)
{
  setZero();
  // set input layer
  int inputDim = topology[0];
	for (int i = 0; i < inputDim; i++)
	{
		neurons[i] = input[i];
	}
  int neuronCount = inputDim;
  int weightCount = 0;
	for (int i = 1; i < depth; i++)
	{
		for (int j = 0; j < topology[i]; j++)
		{
      for (int k = neuronCount-topology[i-1]; k < neuronCount;k++){
        neurons[neuronCount+j] += neurons[k] * weights[k+topology[i-1]*j];
      }
      neurons[neuronCount+j] += biases[neuronCount+j-inputDim];
      neurons[neuronCount+j] = ReLU(neurons[neuronCount+j]);
		}
    neuronCount += topology[i];
	}

}
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

void printModel(){
  printf("weights: \n");
  printArr(totalW, weights);
  printf("biases: \n");
  printArr(totalB, biases);
  printf("neurons: \n");
  printArr(totalN, neurons);
}	

int main()
{
  float output[topology[depth-1]];
	int EPOCHS = 100;
	const int batchSize = 15;
	float input[batchSize][topology[0]];
	float target[batchSize];
	float XOR[12] = {0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0};
  totalN = totalB + topology[0];
  initialize(topology, depth, &weights, &biases, &neurons);
  totalN = totalB + topology[0];
  printf("W: %d, B: %d\n", totalW, totalB);
  printArr(totalW, weights);
  printArr(totalB, biases);
  randomize(topology, depth);
  printModel();
  float bits[] = {0.0, 1.0};
  forwardProp(topology, depth, bits);
  printf("Result: \n");
  printArr(totalN, neurons);
	//cout << stockbear.predict(input) << endl;
	// float errorE = 10;
	// 	for (int i = 0; i<EPOCHS; i++)
	// 	{	
	// 		loadIO(batchSize, input, target);
	// 		//cout<< input[0] <<input[1]<<target[0]<<endl;
	// 		//stockbear.forwardProp(input);
	// 		stockbear.backProp(target, input, batchSize);
	// 		cout << "Epoch " << i+1 << "'s error: " << stockbear.error << endl;
	// 		//cout<<"epoch " << i<< " ";
	// 	}
	//cout << endl << stockbear.error << endl;

	return 0;
}
