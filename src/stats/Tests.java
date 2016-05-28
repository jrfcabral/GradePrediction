package stats;

import neuralnetwork.GradePrediction;
import org.junit.Test;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

/**
 * Created by diogo on 28/05/16.
 */
public class Tests {

    @Test
    public void testNodes(){
        int nodes = 10;
        System.out.println("##### Nodes test #####");
        double maxHits= -1;
        double minError = 0;
        double nrNodes = 10;
        for(int j = 0; j < 5; j++){
            System.out.println("Testing for " + nodes + " nodes.");
            double error = 0;
            double hits = 0;
            long meanTime = 0;
            for (int i = 0; i < 10; i++){
                long time = System.currentTimeMillis();
                GradePrediction g = new GradePrediction("student-por.csv",nodes,0.2,0.001,2000,0.5,60);
                g.testNeuralNetworkNoFile();
                error += g.getMean();
                hits += g.hits();
                meanTime += System.currentTimeMillis() - time;
            }
            error /= 10.0;
            hits /= 10.0;
            meanTime /= 10.0;

            if(meanTime < 1500){
                if(hits > maxHits){
                    nrNodes = nodes;
                    minError = error;
                    maxHits = hits;
                }

            }
            nodes++;
        }
        System.out.println("Best nr of nodes : " + nrNodes);
        System.out.println("Hits : " + maxHits);
        System.out.println("Error : " + minError);
    }

}
