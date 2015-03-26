package cmu.ml.pgm.project;

import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;


/**
 * Methods for feature-enriched matrix factorization. Static class.
 * @author dexter
 *
 */
public final class BayesianMatrixFactorization {

    private BayesianMatrixFactorization() {}

    /**
     *
     * @param data
     * @param latentDim
     * @param maxIter
     * @param tol
     * @return estimates (R, U, V)
     */
    public static MatrixFactorizationResult factorizeMatrix(MatrixFactorizationMovieLens data,
                                                            int latentDim, double stepSize, int maxIter, double tol) {
        throw new UnsupportedOperationException();
    }

    /**
     *
     * @param data
     * @param latentDim
     * @param maxIter
     * @param tol
     * @return estimates (R, U, V, A, B, sigma_R^2, sigma_F^2, sigma_G^2)
     */
    public static MatrixFactorizationResult factorizeMatrixWithFeatures(MatrixFactorizationMovieLens data,
                                                                        int latentDim, double stepSize, int maxIter, double tol) {
        throw new UnsupportedOperationException();

//		Matrix R = data.getRelationMatrix();
//		Matrix F = data.getuFeatureMatrix();
//		Matrix G = data.getiFeatureMatrix();
//		int dF = F.numColumns();
//		int dG = G.numColumns();
//		int m = F.numRows();
//		int n = G.numRows();
//		Matrix U = new DenseMatrix(m, latentDim);
//		Matrix V = new DenseMatrix(n, latentDim);
//		Matrix A = new DenseMatrix(latentDim, dF);
//		Matrix B = new DenseMatrix(latentDim, dG);
//		randomlyInitialize(U);
//		randomlyInitialize(V);
//		randomlyInitialize(A);
//		randomlyInitialize(B);
//		double sigma2R, sigma2F, sigma2G;
//		
//		int t;
//		
//		for (t = 0; t < maxIter; t++) {
//			double overallMaxUpdate = 0;
//			Matrix Rhat = matrixMult(U, transpose(V));
//			sigma2R = frobeniusNorm(R - Rhat)^2 / m / n;
//			sigma2F = frobeniusNorm(F - U*A)^2 / m / df;
//			sigma2G = frobeniusNorm(G - V*B)^2 / n / dg;
//			
//			for (int tA = 0; tA < maxIter; tA++) {
//				Matrix grad = transpose(U) * (F-U*A) / sigma2F;
//				A -= grad * stepSize;
//				double maxUpdate = maxElement(abs(grad * stepSize));
//				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
//				if (maxUpdate < tol)
//					break;
//			}		
//			
//			for (int tB = 0; tB < maxIter; tB++) {
//				Matrix grad = transpose(V) * (G-V*B) / sigma2G;
//				B -= grad * stepSize;
//				double maxUpdate = maxElement(abs(grad * stepSize));
//				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
//				if (maxUpdate < tol)
//					break;
//			}
//			
//			for (int tU = 0; tU < maxIter; tU++) {
//				Matrix grad = (R - U*transpose(V))) * V / sigma2R + 
//					(F - U*A)*transpose(A) / sigma2F;
//				U -= grad * stepSize;
//				double maxUpdate = maxElement(abs(grad * stepSize));
//				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
//				if (maxUpdate < tol)
//					break;
//			}
//			
//			for (int tV = 0; tV < maxIter; tV++) {
//				Matrix grad = transpose(R-U*transpose(V))) * U / sigma2R + 
//					(G - V*B)*transpose(B) / sigma2G;
//				V -= grad * stepSize;
//				double maxUpdate = maxElement(abs(grad * stepSize));
//				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
//				if (maxUpdate < tol)
//					break;
//			}
//			
//			if (overallMaxUpdate < tol)
//				break;
//		}
//
//		Matrix Rhat = matrixMult(U, transpose(V));
//		return new MatrixFactorizationResult(Rhat, U, V, A, sigma2R, sigma2F, sigma2G);
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