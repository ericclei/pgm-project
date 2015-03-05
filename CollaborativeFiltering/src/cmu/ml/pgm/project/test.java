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
        List<Matrix> result = MatrixFactorization.featureEnrichedMatrixFactorization(mf, 10, 1, 10, 1e-6);

        System.out.println(result.get(0));
    }
}
