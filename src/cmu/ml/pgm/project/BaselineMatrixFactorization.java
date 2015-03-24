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
 * @author dexter
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
                                                            int latentDim, double stepsize, double regCoef, int maxIter, double tol) {
        ArrayList<MatrixFactorizationMovieLens.Pair> trainingData = data.getTrainingData();
        int m = data.getNumUsers();
        int n = data.getNumItems();
        int[] numDataPerUser = data.getNumDataPerUser();
        int[] numDataPerItem = data.getNumDataPerItem();
        Random rand = new Random(1);

        DenseMatrix U = new DenseMatrix(m, latentDim);
        DenseMatrix V = new DenseMatrix(n, latentDim);
        randomlyInitialize(U);
        randomlyInitialize(V);

        for(int t = 1; t <= maxIter; t++) {
            int index = rand.nextInt(trainingData.size());
            MatrixFactorizationMovieLens.Pair chosen = trainingData.get(index);
            Vector Ui = getRow(U, chosen.user_id);
            Vector Vj = getRow(V, chosen.item_id);
            double error = chosen.rating - Ui.dot(Vj);
            double learningRate = stepsize;// / (t * t);
            double scale = 2 * error * learningRate;// * trainingData.size();
            double regularization = -2 * learningRate * regCoef;// * trainingData.size();
            Vector Uupdate = Vj.copy().scale(scale).add(Ui.copy().scale(regularization / numDataPerUser[chosen.user_id]));
            Vector Vupdate = Ui.copy().scale(scale).add(Vj.copy().scale(regularization / numDataPerItem[chosen.item_id]));
            double difference = Uupdate.norm(Vector.Norm.Two) + Vupdate.norm(Vector.Norm.Two);

            if(difference < tol) {
                break;
            }

            addRow(U, Uupdate, chosen.user_id);
            addRow(V, Vupdate, chosen.item_id);
//            if (Double.isNaN(Uupdate.norm(Vector.Norm.Two))) {
//                System.out.println(U);
//                System.out.println("\n" + t + " " + error + " " + scale + " " + regularization);
//                System.exit(-1);
//            }
//            else if (Uupdate.norm(Vector.Norm.Two) > 10) {
//                System.out.println(Uupdate);
//                System.out.println("\n" + t + " " + error + " " + scale + " " + regularization);
//                System.exit(-1);
//            }
//            System.out.println(Uupdate + "\n");
//            System.out.println(V);
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

    static void addRow(Matrix x, Vector v, int i) {
        for (int j = 0; j < x.numColumns(); j++) {
            x.set(i, j, x.get(i,j) + v.get(j));
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