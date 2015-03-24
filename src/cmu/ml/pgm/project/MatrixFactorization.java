package cmu.ml.pgm.project;

import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;


/**
 * Methods for feature-enriched matrix factorization. Static class.
 * @author eric
 *
 */
public final class MatrixFactorization {

	private MatrixFactorization() {}

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
		Matrix R = data.getRelationMatrix();
		Matrix F = data.getuFeatureMatrix();
		Matrix G = data.getiFeatureMatrix();
		int dF = F.numColumns();
		int dG = G.numColumns();
		int m = F.numRows();
		int n = G.numRows();
		Matrix U = new DenseMatrix(m, latentDim);
		Matrix V = new DenseMatrix(n, latentDim);
		Matrix A = new DenseMatrix(latentDim, dF);
		Matrix B = new DenseMatrix(latentDim, dG);
		randomlyInitialize(U);
		randomlyInitialize(V);
		randomlyInitialize(A);
		randomlyInitialize(B);
		double sigma2R = 0, sigma2F = 0, sigma2G = 0;

		for (int t = 0; t < maxIter; t++) {
			System.out.printf("t = %d\n", t);
			double overallMaxUpdate = 0;
			Matrix Rhat = times(U, transpose(V));
			sigma2R = squaredFrobeniusNorm(minus(R, Rhat)) / m / n;
			sigma2F = squaredFrobeniusNorm(minus(F, times(U, A))) / m / dF;
			sigma2G = squaredFrobeniusNorm(minus(G, times(V, B))) / n / dG;

			for (int tA = 0; tA < maxIter; tA++) {
				System.out.printf("\ttA = %d\n", tA);
				Matrix grad = times(transpose(U), minus(F, times(U, A)));
				grad = times(grad, 1 / sigma2F);
				Matrix scaledGrad = times(grad, stepSize);
				A = minus(A, scaledGrad);
				double maxUpdate = maxAbsElement(scaledGrad);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}		

			for (int tB = 0; tB < maxIter; tB++) {
				System.out.printf("\ttB = %d\n", tB);
				Matrix grad = times(transpose(V) , minus(G, times(V, B)));
				grad = times(grad, 1 / sigma2G);
				Matrix scaledGrad = times(grad, stepSize);
				B = minus(B, scaledGrad);
				double maxUpdate = maxAbsElement(scaledGrad);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}

			for (int tU = 0; tU < maxIter; tU++) {
				System.out.printf("\ttU = %d\n", tU);
				Matrix grad1 = times(times(minus(R, times(U, transpose(V))), V), 1 / sigma2R);
				Matrix grad2 = times(times(minus(F, times(U, A)), transpose(A)), 1 / sigma2F);
				Matrix grad = plus(grad1, grad2);
				Matrix scaledGrad = times(grad, stepSize);
				U = minus(U, scaledGrad);
				double maxUpdate = maxAbsElement(scaledGrad);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}

			for (int tV = 0; tV < maxIter; tV++) {
				System.out.printf("\ttV = %d\n", tV);
				Matrix grad1 = times(times(minus(R, times(U, transpose(V))), V), 1 / sigma2R);
				Matrix grad2 = times(times(minus(G, times(V, B)), transpose(B)), 1 / sigma2G);
				Matrix grad = plus(grad1, grad2);
				Matrix scaledGrad = times(grad, stepSize);
				V = minus(V, scaledGrad);
				double maxUpdate = maxAbsElement(scaledGrad);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}

			if (overallMaxUpdate < tol)
				break;

			if (t == maxIter - 1)
				System.out.printf("Coordinate descent did not converge. Last max update = %f.\n", 
						overallMaxUpdate);
		}

		Matrix Rhat = times(U, transpose(V));
		return new MatrixFactorizationResult(Rhat, U, V, A, sigma2R, sigma2F, sigma2G);
	}

	static Matrix times(Matrix x, double a) {
		return x.copy().scale(a);
	}

	static Matrix times(Matrix x, Matrix y) {
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

	static Matrix plus(Matrix x, Matrix y) {
		return x.copy().add(y);
	}

	static Matrix minus(Matrix x, Matrix y) {
		return x.copy().add(-1, y);
	}

	static double squaredFrobeniusNorm(Matrix x) {
		return Math.pow(x.norm(Matrix.Norm.Frobenius), 2);
	}

	static double maxAbsElement(Matrix x) {
		double m = 0;
		for (int i = 0; i < x.numRows(); i++) 
			for (int j = 0; j < x.numColumns(); j++)
				m = Math.max(m, Math.abs(x.get(i, j)));
		return m;
	}

}