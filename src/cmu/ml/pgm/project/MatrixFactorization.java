package cmu.ml.pgm.project;

import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.io.MatrixInfo.MatrixField;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;


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
		LinkedSparseMatrix R = data.getRelationMatrix();
		int nObservedRelations = l0Norm(R);
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

		// coordinate descent
		for (int t = 0; t < maxIter; t++) {
//			System.out.printf("t = %d\n", t);
			double overallMaxUpdate = 0;
			Matrix Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat) / nObservedRelations;
			sigma2F = squaredFrobeniusNorm(minus(F, times(U, A))) / m / dF;
			sigma2G = squaredFrobeniusNorm(minus(G, times(V, B))) / n / dG;

			// updates A
			for (int tA = 0; tA < maxIter; tA++) {
//				System.out.printf("\ttA = %d\n", tA);
				Matrix grad = times(transpose(U), minus(F, times(U, A)));
				grad.scale(1 / sigma2F);
				Matrix scaledGrad = times(grad, stepSize);
				A.add(-1, scaledGrad);
				double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}		
			sigma2F = squaredFrobeniusNorm(minus(F, times(U, A))) / m / dF;

			// updates B
			for (int tB = 0; tB < maxIter; tB++) {
//				System.out.printf("\ttB = %d\n", tB);
				Matrix grad = times(transpose(V) , minus(G, times(V, B)));
				grad.scale(1 / sigma2G);
				Matrix scaledGrad = times(grad, stepSize);
				B.add(-1, scaledGrad);
				double maxUpdate = scaledGrad.norm(Matrix.Norm.Maxvalue);
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			sigma2G = squaredFrobeniusNorm(minus(G, times(V, B))) / n / dG;

			// updates U
			for (int tU = 0; tU < maxIter; tU++) {
//				System.out.printf("\ttU = %d\n", tU);
				double maxUpdate = 0;
				for (int i = 0; i < m; i++) {
					// updates U_i
					Vector grad = new DenseVector(latentDim);
					for (int j = 0; j < n; j++) {
						double rij = R.get(i, j);
						double rijHat = Rhat.get(i, j);
						if (rij != 0)
							grad.add(times(getRow(V, j), (rij - rijHat)));
					}
					Vector scaledGrad = times(grad, stepSize);
					addToRow(U, i, scaledGrad);
					maxUpdate = maxAbs(scaledGrad);
				}
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat) / nObservedRelations;
			sigma2F = squaredFrobeniusNorm(minus(F, times(U, A))) / m / dF;

			// updates V
			for (int tV = 0; tV < maxIter; tV++) {
//				System.out.printf("\ttV = %d\n", tV);
				double maxUpdate = 0;
				for (int j = 0; j < n; j++) {
					// updates V_j
					Vector grad = new DenseVector(latentDim);
					for (int i = 0; i < m; i++) {
						double rij = R.get(i, j);
						double rijHat = Rhat.get(i, j);
						if (rij != 0)
							grad.add(times(getRow(U, i), (rij - rijHat)));
					}
					Vector scaledGrad = times(grad, stepSize);
					addToRow(V, j, scaledGrad);
					maxUpdate = maxAbs(scaledGrad);
				}
				overallMaxUpdate = Math.max(overallMaxUpdate, maxUpdate);
				if (maxUpdate < tol)
					break;
			}
			Rhat = times(U, transpose(V));
			sigma2R = sparseSquaredFrobeniusNormOfDiff(R, Rhat) / nObservedRelations;
			sigma2G = squaredFrobeniusNorm(minus(G, times(V, B))) / n / dG;

			if (overallMaxUpdate < tol)
				break;

			if (t == maxIter - 1)
				System.out.printf("Coordinate descent did not converge. Last max update = %f.\n", 
						overallMaxUpdate);
		}

		Matrix Rhat = times(U, transpose(V));
		return new MatrixFactorizationResult(Rhat, U, V, A, B, sigma2R, sigma2F, sigma2G);
	}

	private static double maxAbs(Vector x) {
		double m = 0;
		for (int i = 0; i < x.size(); i++)
			m = Math.max(m, Math.abs(x.get(i)));
		return m;
	}

	private static void addToRow(Matrix x, int i, Vector v) {
		for (int j = 0; j < x.numColumns(); j++) {
			x.set(i, j, x.get(i, j) + v.get(j));
		}
	}

	private static Vector times(Vector x, double a) {
		return x.copy().scale(a);
	}

	static Matrix times(Matrix x, double a) {
		return x.copy().scale(a);
	}

	static Matrix times(Matrix x, Matrix y) {
		return x.mult(y, new DenseMatrix(x.numRows(), y.numColumns()));
	}

	static Vector getColumn(Matrix x, int i) {
		int n = x.numRows();
		Vector v = new DenseVector(n);
		for (int j = 0; j < n; j++)
			v.set(j, x.get(j, i));
		return v;
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

	static double sparseSquaredFrobeniusNormOfDiff(LinkedSparseMatrix x, Matrix y) {
		double norm = 0;
		for (int i = 0; i < x.numRows(); i++)
			for (int j = 0; j < x.numColumns(); j++) {
				double xij = x.get(i, j);
				if (xij != 0) {
					norm += Math.pow((xij - y.get(i, j)), 2);
				}
			}
		
		return norm;
	}
	
	static int l0Norm(Matrix x) {
		int n = 0;
		for (int i = 0; i < x.numRows(); i++)
			for (int j = 0; j < x.numColumns(); j++)
				n += x.get(i, j) == 0 ? 0 : 1;
		return n;
	}

}