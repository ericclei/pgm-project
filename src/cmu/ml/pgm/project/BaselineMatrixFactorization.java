package cmu.ml.pgm.project;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;


/**
 * Methods for feature-enriched matrix factorization. Static class.
 * @author eric
 *
 */
public final class BaselineMatrixFactorization {

    private BaselineMatrixFactorization() {}

    /**
     *
     * @param data
     * @param latentDim
     * @param regCoef
     * @param maxIter
     * @param tol
     * @return estimates (R, U, V)
     */
    public static MatrixFactorizationResult factorizeMatrix(MatrixFactorizationMovieLens data,
                                                            int latentDim, double regCoef, int maxIter, double tol) {
        ArrayList<MatrixFactorizationMovieLens.Pair> trainingData = data.getTrainingData();
        int m = data.getNumUsers();
        int n = data.getNumItems();
        Random rand = new Random(1);

        DenseMatrix U = new DenseMatrix(m, latentDim);
        DenseMatrix V = new DenseMatrix(n, latentDim);
        randomlyInitialize(U);
        randomlyInitialize(V);

        for(int t = 0; t < maxIter; t++) {
            MatrixFactorizationMovieLens.Pair chosen = trainingData.get(rand.nextInt(trainingData.size()));
            Vector Ui = getRow(U, chosen.user_id);
            Vector Vj = getRow(V, chosen.item_id);
            double error = chosen.rating - Ui.dot(Vj);

            Vector Uupdate = Vj.scale(-2 * error).add(Ui.scale(2 * regCoef / Ui.size()));
            Vector Vupdate = Ui.scale(-2 * error).add(Vj.scale(2 * regCoef / Vj.size()));
            double difference = Uupdate.norm(Vector.Norm.Two) + Vupdate.norm(Vector.Norm.Two);

//            if(difference < tol) {
//                break;
//            }

            setRow(U, Uupdate, chosen.user_id);
            setRow(V, Vupdate, chosen.item_id);

        }

        return new MatrixFactorizationResult(matrixMult(U, transpose(V)), U, V);
    }

    static Matrix matrixMult(Matrix x, Matrix y) {
        return x.mult(y, new DenseMatrix(x.numRows(), y.numColumns()));
    }

    static Vector getRow(Matrix x, int i) {
        int n = x.numColumns();
        Vector v = new DenseVector(n);
        for (int j = 0; j < n; j++)
            v.set(j, x.get(i, j));
        return v;
    }

    static void setRow(Matrix x, Vector v, int i) {
        for (int j = 0; j < x.numColumns(); j++) {
            x.set(i, j, v.get(j));
        }
    }

    /**
     * U(-1, 1)
     * @param x
     */
    static void randomlyInitialize(Matrix x) {
        Random r = new Random();
        for (int i = 0; i < x.numRows(); i++)
            for (int j = 0; j < x.numColumns(); j++)
                x.set(i, j, 2 * r.nextDouble() - 1);
    }

    static void setAllValues(Matrix x, double val) {
        for (int i = 0; i < x.numRows(); i++)
            for (int j = 0; j < x.numColumns(); j++)
                x.set(i, j, val);
    }

    static Matrix transpose(Matrix x) {
        return x.transpose(new DenseMatrix(x.numColumns(), x.numRows()));
    }


}