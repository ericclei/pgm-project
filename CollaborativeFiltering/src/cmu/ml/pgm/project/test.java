package cmu.ml.pgm.project;

import java.util.List;

import no.uib.cipr.matrix.Matrix;

/**
 * Created by dexter on 15. 3. 3..
 */
public class test {
    public static void main(String[] args) {
        MatrixFactorizationMovieLens mf
                = new MatrixFactorizationMovieLens("Data/ml-100k/u.user", "Data/ml-100k/u.item",
                "Data/ml-100k/u.data", "Data/ml-100k/u.info");
        
        //mf.printMatrix();
        int latentDim = 10;
        double stepSize = 1e-13;
        int maxIter = 10;
        double eps = 1e-4;
        List<Matrix> result = MatrixFactorization.featureEnrichedMatrixFactorization(mf, latentDim, stepSize, maxIter, eps);
        
        Matrix r = result.get(0);
        for (int i = 0; i < 10; i++) {
        	for (int j = 0; j < 10; j++) 
        		System.out.print(r.get(i, j) + " ");
        	System.out.println();
        }

        System.out.println("done");

    }
}
