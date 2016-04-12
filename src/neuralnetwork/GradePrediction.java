package neuralnetwork;


import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxMinNormalizer;
import org.neuroph.util.data.sample.SubSampling;

import java.util.Arrays;
import java.util.List;


/**
 * Created by diogo on 12/04/16.
 */
public class GradePrediction {

    MultiLayerPerceptron network;
    DataSet trainingSet;
    DataSet testSet;

    GradePrediction(){
        DataSet data = DataSet.createFromFile("student-mat.csv",32,1,";",true);
        MaxMinNormalizer normalizer = new MaxMinNormalizer();
        normalizer.normalize(data);
        SubSampling sampling = new SubSampling(66,34);
        List<DataSet> sets = sampling.sample(data);
        trainingSet = sets.get(0);
        testSet = sets.get(1);

        System.out.println("Data Set size:" + data.size());
        System.out.println("Training Set size:" + trainingSet.size());
        System.out.println("Test Set size:" + testSet.size());

        double connections = (double)trainingSet.size()/4.0;
        int nodes = (int)Math.round(connections/32.0);
        System.out.println("Hidden Nodes:" + nodes);
        network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,32,nodes,1);
        BackPropagation bp  = new BackPropagation();
        bp.setNeuralNetwork(network);
        bp.setMaxError(0.005);
        bp.doLearningEpoch(trainingSet);
        network.learn(trainingSet, bp);

        System.out.println();
        System.out.println("######### TRAINING #########");
        System.out.println();

        System.out.println("Total Iterations: " + bp.getCurrentIteration());
        System.out.println("Total error: " + bp.getTotalNetworkError());

    }

    public void testNeuralNetwork() {
        System.out.println();
        System.out.println("######### TESTING #########");
        System.out.println();
        
        double totalError = 0;
        double squareSum = 0;
        for(DataSetRow dataRow : testSet.getRows()) {
            network.setInput(dataRow.getInput());
            network.calculate();
            double[] networkOutput = network.getOutput();
            double error = Math.abs(networkOutput[0]*20 - dataRow.getDesiredOutput()[0]);
            totalError += error;
            squareSum += error * error;
        }

        double mean = totalError/testSet.size();
        double stddev = Math.sqrt(squareSum/testSet.size() - mean*mean);

        System.out.println("Mean: " + mean);
        System.out.println("Std dev: "  + stddev);
    }

}
