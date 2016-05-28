package neuralnetwork;


import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxMinNormalizer;
import org.neuroph.util.data.sample.SubSampling;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Locale;


/**
 * Created by João Guarda on 12/04/16.
 *
 * Grade Prediction class using neuroph API
 */
public class GradePrediction {

    MultiLayerPerceptron network;
    DataSet trainingSet;
    DataSet testSet;
    MomentumBackpropagation bp;
    double max;
    double mean;
    double stdDev;
    DataSet data;
    int hits;

    public GradePrediction(String dataSetPath, int nodes, double learningRate, double maxError, int maxIterations, double momentum, int trainingPercentage){
        hits=0;
        data = DataSet.createFromFile(dataSetPath,31,1,";",true);
        getMaxOutput();
        MaxMinNormalizer normalizer = new MaxMinNormalizer();
        normalizer.normalize(data);
        SubSampling sampling = new SubSampling(trainingPercentage,100-trainingPercentage);
        List<DataSet> sets = sampling.sample(data);
        trainingSet = sets.get(0);
        testSet = sets.get(1);


        network = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,31,nodes,1);
        bp  = new MomentumBackpropagation();
        bp.setNeuralNetwork(network);
        bp.setLearningRate(learningRate);
        bp.setMaxError(maxError);
        bp.setMaxIterations(maxIterations);
        bp.setMomentum(momentum);
        network.learn(trainingSet, bp);


    }

    public int getIterations(){
        return bp.getCurrentIteration();
    }

    public double getTotalNetworkError(){
        return bp.getTotalNetworkError();
    }

    public int testSetSize(){
        return testSet.size();
    }

    public int trainingSetSize(){
        return trainingSet.size();
    }


    public void testNeuralNetwork(){
        hits = 0;
        File f = new File("output.csv");
        try {
            FileOutputStream stream = new FileOutputStream(f);
            double[] desired = new double[this.testSet.size()];
            double[] output = new double[this.testSet.size()];
            double totalError = 0;
            double squareSum = 0;
            int i=0;

            stream.write("output;desired\n".getBytes());

            for(DataSetRow dataRow : testSet.getRows()) {
                network.setInput(dataRow.getInput());
                network.calculate();
                double[] networkOutput = network.getOutput();
                output[i] = Math.round(networkOutput[0] * max);
                desired[i] = dataRow.getDesiredOutput()[0] * max;
                double error = Math.abs(output[i] - desired[i]);
                if(error == 0.0)
                    hits++;
                totalError += error;
                squareSum += error * error;

                String s =  String.format(Locale.FRANCE,"%f", output[i])+ ";" + String.format(Locale.FRANCE,"%f", desired[i]) + "\n";
                stream.write(s.getBytes());
                i++;
            }

            stream.close();
            /*MeanSquaredError error = new MeanSquaredError();
            error.calculatePatternError(output,desired);
            error = error.getTotalError();
            */
            mean = totalError/testSet.size();
            stdDev = Math.sqrt(squareSum/testSet.size() - mean*mean);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void getMaxOutput(){
        max = -1;
        List<DataSetRow> rows = data.getRows();
        for (DataSetRow r: rows){
            double output = r.getDesiredOutput()[0];
            if(output > max){
                max = output;
            }
        }
    }


    public double getStdDev() {
        return stdDev;
    }

    public double getMean() {
        return mean;
    }

    public int hits() {
        return hits;
    }

    public void testNeuralNetworkNoFile() {
        hits = 0;
        double[] desired = new double[this.testSet.size()];
        double[] output = new double[this.testSet.size()];
        double totalError = 0;
        double squareSum = 0;
        int i=0;

        for(DataSetRow dataRow : testSet.getRows()) {
            network.setInput(dataRow.getInput());
            network.calculate();
            double[] networkOutput = network.getOutput();
            output[i] = Math.round(networkOutput[0] * max);
            desired[i] = dataRow.getDesiredOutput()[0] * max;
            double error = Math.abs(output[i] - desired[i]);
            if(error == 0.0)
                hits++;
            totalError += error;
            squareSum += error * error;
            i++;
        }

        mean = totalError/testSet.size();
        stdDev = Math.sqrt(squareSum/testSet.size() - mean*mean);
    }
}