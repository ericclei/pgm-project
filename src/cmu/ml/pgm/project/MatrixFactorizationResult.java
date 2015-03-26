package cmu.ml.pgm.project;

import java.util.List;

import no.uib.cipr.matrix.Matrix;

public class MatrixFactorizationResult {
	private Matrix R;
	private Matrix[] U, V, A, B;
	private double sigma2R, sigma2F, sigma2G;
	private List<Matrix> intermediateR;
	public MatrixFactorizationResult(Matrix r, Matrix u, Matrix v, Matrix a, Matrix b,
			double sigma2r, double sigma2f, double sigma2g, List<Matrix> iR) {
		R = r;
		U = new Matrix[]{u};
		V = new Matrix[]{v};
		A = new Matrix[]{a};
		B = new Matrix[]{b};
		sigma2R = sigma2r;
		sigma2F = sigma2f;
		sigma2G = sigma2g;
		intermediateR = iR;
	}

	public MatrixFactorizationResult(Matrix r, Matrix u, Matrix v) {
		R = r;
		U = new Matrix[]{u};
		V = new Matrix[]{v};
	}

	public MatrixFactorizationResult(Matrix r, Matrix[] u, Matrix[] v) {
		R = r;
		U = u;
		V = v;
	}

	public List<Matrix> getIntermediateR() {
		return intermediateR;
	}

	public Matrix getR() {
		return R;
	}
	public Matrix[] getU() {
		return U;
	}
	public Matrix[] getV() {
		return V;
	}
	public Matrix[] getA() {
		return A;
	}
	public Matrix[] getB() {
		return B;
	}
	public double getSigma2R() {
		return sigma2R;
	}
	public double getSigma2F() {
		return sigma2F;
	}
	public double getSigma2G() {
		return sigma2G;
	}
}
