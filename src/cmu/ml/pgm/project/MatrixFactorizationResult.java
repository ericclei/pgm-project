package cmu.ml.pgm.project;

import java.util.List;

import no.uib.cipr.matrix.Matrix;

public class MatrixFactorizationResult {
	private Matrix R;
	private Matrix[] U, V, An, Bn, Ab, Bb;
	private double sigma2R, sigma2F, sigma2G;
	private List<Matrix> intermediateR;
	public MatrixFactorizationResult(Matrix r, Matrix u, Matrix v, Matrix an, Matrix bn, Matrix ab, Matrix bb,
			double sigma2r, double sigma2f, double sigma2g, List<Matrix> iR) {
		R = r;
		U = new Matrix[]{u};
		V = new Matrix[]{v};
		An = new Matrix[]{an};
		Bn = new Matrix[]{bn};
		Ab = new Matrix[]{ab};
		Bb = new Matrix[]{bb};
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
	public Matrix[] getAn() {
		return An;
	}
	public Matrix[] getBn() {
		return Bn;
	}
	public Matrix[] getAb() {
		return Ab;
	}
	public Matrix[] getBb() {
		return Bb;
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
