package cmu.ml.pgm.project;

import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class MatrixMethods {

	private MatrixMethods() {
	}

	static Matrix removeColumn(Matrix m, int k) {
		Matrix result = new DenseMatrix(m.numRows(), m.numColumns() - 1);
		for (int i = 0; i < m.numRows(); i++)
			for (int j = 0; j < m.numColumns(); j++) {
				if (j < k)
					result.set(i, j, m.get(i, j));
				if (j > k)
					result.set(i, j - 1, m.get(i, j));
			}
		return result;
	}

	static double maxAbs(Vector x) {
		double m = 0;
		for (int i = 0; i < x.size(); i++)
			m = Math.max(m, Math.abs(x.get(i)));
		return m;
	}

	static void addToColumn(Matrix x, int i, Vector v) {
		for (int j = 0; j < x.numRows(); j++) {
			x.set(j, i, x.get(j, i) + v.get(j));
		}
	}

	static void addToRow(Matrix x, int i, Vector v) {
		for (int j = 0; j < x.numColumns(); j++) {
			x.set(i, j, x.get(i, j) + v.get(j));
		}
	}

	static Vector times(Vector x, double a) {
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
	 * U(0, 1)
	 * 
	 * @param x
	 */
	static void randomlyInitialize(Matrix x) {
		Random r = new Random();
		for (int i = 0; i < x.numRows(); i++)
			for (int j = 0; j < x.numColumns(); j++)
				x.set(i, j, r.nextDouble());
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

	static double sparseSquaredFrobeniusNormOfDiff(Matrix x, Matrix y) {
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

	static void bound(Matrix x, double min, double max) {
		for (int i = 0; i < x.numRows(); i++)
			for (int j = 0; j < x.numColumns(); j++)
				x.set(i, j, Math.max(min, Math.min(max, x.get(i, j))));
	}
}
