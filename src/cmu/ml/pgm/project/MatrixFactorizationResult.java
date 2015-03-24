package cmu.ml.pgm.project;

import no.uib.cipr.matrix.Matrix;

public class MatrixFactorizationResult {
	private Matrix R, U, V, A;
	private double sigma2R, sigma2F, sigma2G;
	public MatrixFactorizationResult(Matrix r, Matrix u, Matrix v, Matrix a,
			double sigma2r, double sigma2f, double sigma2g) {
		R = r;
		U = u;
		V = v;
		A = a;
		sigma2R = sigma2r;
		sigma2F = sigma2f;
		sigma2G = sigma2g;
	}

	public MatrixFactorizationResult(Matrix r, Matrix u, Matrix v) {
		R = r;
		U = u;
		V = v;
	}

	public Matrix getR() {
		return R;
	}
	public Matrix getU() {
		return U;
	}
	public Matrix getV() {
		return V;
	}
	public Matrix getA() {
		return A;
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
