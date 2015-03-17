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
public final class MatrixFactorization {

	private MatrixFactorization() {}

	/**
	 * 
	 * @param data
	 * @param latentDim
	 * @param maxIter
	 * @param eps
	 * @return estimates (R, U, V) 
	 */
	public static List<Matrix> matrixFactorization(MatrixFactorizationMovieLens data,
			int latentDim, double stepSize, int maxIter, double eps) {
		return matrixFactorizationSubroutine(data, latentDim, stepSize, maxIter, eps, false).subList(0, 3);
	}

	/**
	 * 
	 * @param data
	 * @param latentDim
	 * @param maxIter
	 * @param eps
	 * @return estimates (R, U, V, A, B) 
	 */
	public static List<Matrix> featureEnrichedMatrixFactorization(MatrixFactorizationMovieLens data,
			int latentDim, double stepSize, int maxIter, double eps) {
		return matrixFactorizationSubroutine(data, latentDim, stepSize, maxIter, eps, true);
	}

	private static List<Matrix> matrixFactorizationSubroutine(MatrixFactorizationMovieLens data,
			int latentDim, double stepSize, int maxIter, double eps, boolean useFeatures) {
		Matrix f = data.getuFeatureMatrix();
		Matrix g = data.getiFeatureMatrix();
		int df = f.numColumns();
		int dg = g.numColumns();
		int m = f.numRows();
		int n = g.numRows();
		Matrix a = new DenseMatrix(df, latentDim);
		Matrix b = new DenseMatrix(dg, latentDim);
		
		if (!useFeatures) {
			df = m;
			dg = n;
			f = Matrices.identity(m);
			g = Matrices.identity(n);
			a = new DenseMatrix(m, latentDim);
			b = new DenseMatrix(n, latentDim);
		}

		randomlyInitialize(a);
		randomlyInitialize(b);

		int t;
		// outer loop over both A and B
		for (t = 0; t < maxIter; t++) {
			System.out.println("t = " + t);
			Matrix u = matrixMult(f, a);
			Matrix v = matrixMult(g, b);
			Matrix r = matrixMult(u, v.transpose(new DenseMatrix(v.numColumns(), v.numRows())));

			double maxUpdateOfAorB = 0;

			// gradient descent on A
			for (int tA = 0; tA < maxIter; tA++) {
				System.out.println("\ttA = " + tA);
				Matrix aNew = new DenseMatrix(a);
				double maxUpdate = 0;
				for (int l = 0; l < latentDim; l++) {
//					System.out.println("l = " + l);
					Vector gradA_l = new DenseVector(df);
					for (int i = 0; i < m; i++) {
						for (int j = 0; j < n; j++) {
							double rij = data.getRelationMatrix().get(i, j);
							if (rij == 0)
								continue;
							Vector fi = getRow(f, i);
							double aVal = (rij - r.get(i, j)) * v.get(j, l);
							gradA_l.add(new DenseVector(fi).scale(-2 * aVal / n / m));
						}
					}
					for (int i = 0; i < df; i++) {
						double update = -stepSize * gradA_l.get(i);
						maxUpdate = Math.max(Math.abs(update), maxUpdate);
						aNew.add(i, l, update);
					}
				}
				maxUpdateOfAorB = Math.max(maxUpdateOfAorB, maxUpdate);
				a = aNew;
				u = matrixMult(f, a);
				r = matrixMult(u, v.transpose(new DenseMatrix(v.numColumns(), v.numRows())));
				if (maxUpdate < eps)
					break;
				if (tA == maxIter - 1)
					System.out.println("gradient descent for A did not converge; last max update = " +
							maxUpdate);
			}

			// gradient descent on B
			for (int tB = 0; tB < maxIter; tB++) {
				System.out.println("\ttB = " + tB);
				Matrix bNew = new DenseMatrix(b);
				double maxUpdate = 0;
				for (int l = 0; l < latentDim; l++) {
					Vector gradB_l = new DenseVector(dg);
					for (int i = 0; i < m; i++) {
						for (int j = 0; j < n; j++) {
							double rij = data.getRelationMatrix().get(i, j);
							if (rij == 0)
								continue;
							Vector gj = getRow(g, j);
							double bVal = (rij - r.get(i, j)) * u.get(i, l);
							gradB_l.add(new DenseVector(gj).scale(-2 * bVal / n / m));
						}
					}
					for (int i = 0; i < dg; i++) {
						double update = -stepSize * gradB_l.get(i);
						maxUpdate = Math.max(Math.abs(update), maxUpdate);
						bNew.add(i, l, update);
					}
				}
				maxUpdateOfAorB = Math.max(maxUpdateOfAorB, maxUpdate);
				b = bNew;
				v = matrixMult(g, b);
				r = matrixMult(u, v.transpose(new DenseMatrix(v.numColumns(), v.numRows())));
				if (maxUpdate < eps) 
					break;
				if (tB == maxIter - 1)
					System.out.println("gradient descent for B did not converge; last max update = " +
							maxUpdate);
			}

			if (maxUpdateOfAorB < eps) {
				System.out.format("t = %d; max update = %f < %f = eps\n", t, maxUpdateOfAorB, eps);;
				break;
			}

			if (t == maxIter - 1)
				System.out.println("coordinate descent did not converge; last max update = " +
						maxUpdateOfAorB);
		}

		Matrix u = matrixMult(f, a);
		Matrix v = matrixMult(g, b);
		Matrix r = matrixMult(u, v.transpose(new DenseMatrix(v.numColumns(), v.numRows())));
		List<Matrix> result = Arrays.asList(r, u, v, a, b);

		return result;
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
}
