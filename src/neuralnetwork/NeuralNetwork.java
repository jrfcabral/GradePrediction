package neuralnetwork;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.DoubleStream;

public class NeuralNetwork {
	
	private double[][] values;
	private double[][][] weights;
	
	private double sigmoid(double in){
		return 1/(1+java.lang.Math.exp(in));
	}
	
	public NeuralNetwork(double[] input, int[] layers, int outputs){
		this.values = new double[2+layers.length][];
		this.values[0] = input;
		for(int i = 1; i <= layers.length; i++){
			values[i] = new double[layers[i-1]];
		}
		this.values[layers.length+1] = new double[outputs];
		weights = new double[layers.length+1][][];
		
		for (int i = 1; i < this.values.length;i++){
			weights[i-1] = new double[this.values[i].length][];
			for(int j = 0; j < this.values[i].length; j++){
				weights[i-1][j] = new double[this.values[i-1].length]; 
				weights[i-1][j] = Arrays.stream(weights[i-1][j]).map(s->ThreadLocalRandom.current().nextDouble(-1, 1)).toArray();
			}
		}
		
	}
	
	public void forwardPropagate(){
		for(int i = 1; i < values.length;i++){
			for(int j = 0; j < values[i].length; j++){
				double[] currweights = this.weights[i-1][j];
				double[] inputs = this.values[i-1];
				double a = 0;
				for(int k = 0; k < currweights.length;k++){
					a += currweights[k] * inputs[k];
				}
				this.values[i][j] = a;
			}
		}
	}

	public static void main(String[] args) {
		double[] input = {0.5,0.1,0.2};
		int[] layers = {5};
		NeuralNetwork network = new NeuralNetwork(input, layers, 1);
		network.forwardPropagate();
		System.exit(0);
	}

}
