package gui;

import neuralnetwork.GradePrediction;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Created by diogo on 27/05/16.
 */
public class Gui extends JFrame{

    private GradePrediction gradePrediction;
    private JSpinner layers;
    private JSpinner nodes;
    private JPanel panel;
    private JTextField path;
    private JButton chooseFileButton;
    private JSpinner iterations;
    private JSpinner error;
    private JSpinner momentum;
    private JSpinner rate;
    private JButton runButton;
    private JButton testButton;

    public Gui(){
        super("Grade Prediction");
        SpinnerNumberModel nodesModel = new SpinnerNumberModel(15,1,1000,1);
        SpinnerNumberModel iterationsModel = new SpinnerNumberModel(1000,1,100000,100);
        SpinnerNumberModel layersModel = new SpinnerNumberModel(1,1,2,1);
        SpinnerNumberModel momentumModel = new SpinnerNumberModel(0.5,0,1,0.05);
        SpinnerNumberModel errorModel = new SpinnerNumberModel(0.001,0,1,0.001);
        SpinnerNumberModel rateModel = new SpinnerNumberModel(0.2,0,1,0.05);
        layers.setModel(layersModel);
        nodes.setModel(nodesModel);
        momentum.setModel(momentumModel);
        error.setModel(errorModel);
        iterations.setModel(iterationsModel);
        rate.setModel(rateModel);
        setContentPane(panel);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();

        chooseFileButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser fc = new JFileChooser();
                fc.showOpenDialog(panel);
                if(fc.getSelectedFile().getName() != null)
                    path.setText(fc.getSelectedFile().getName());
            }
        });
        Gui that = this;

        runButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (path.getText().equals("")){
                    JOptionPane.showMessageDialog(that,"Input file was not specified", "ERROR", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                gradePrediction = new GradePrediction(path.getText(),(int) nodes.getValue(),(double) rate.getValue(), (double) error.getValue(), (int) iterations.getValue(), (double)momentum.getValue());
                testButton.setEnabled(true);
                JOptionPane.showMessageDialog(that, "Training Set size: " + gradePrediction.trainingSetSize() + "\n" +
                        "Test Set size: " + gradePrediction.testSetSize() + "\n" + "\n" +
                        "Total Iterations: " + gradePrediction.getIterations() + "\n" +
                        "Total Network Error: " + + gradePrediction.getTotalNetworkError() + "\n",
                        "TRAINING RESULTS",
                        JOptionPane.INFORMATION_MESSAGE

                );

            }
        });
        setVisible(true);
        testButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                gradePrediction.testNeuralNetwork();
                JOptionPane.showMessageDialog(that, "Test Set size: " + gradePrediction.testSetSize() + "\n" + "\n" +
                        "Error Mean: " + gradePrediction.getMean() + "\n" +
                        "Error standard deviation: " + gradePrediction.getStdDev() + "\n"+
                        "Hits: " + gradePrediction.hits() + "\n"+
                        "Hits percentage: " + (double)gradePrediction.hits()/(double)gradePrediction.testSetSize()*100.0 + "%"+
                        "\n\n\n"+ "!!! CHECK OUT output.csv !!!",
                        "TEST RESULTS",
                        JOptionPane.INFORMATION_MESSAGE

                );
            }
        });
    }

    public static void main(String[] args){
        new Gui();
    }

}
